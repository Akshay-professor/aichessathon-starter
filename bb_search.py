"""Iterative deepening alpha-beta over the bitboard engine.

Search state is allocated once by `State` and threaded through every jitted function as
explicit array parameters. numba captures module-level arrays as *readonly* constants, so
anything the search writes has to be an argument; only the immutable tables below stay global.
The signatures are long as a result, and composing the shared prefix once keeps them honest.

Time is enforced from outside. The search is `nogil=True`, so a plain Python timer thread keeps
running while it executes and can set `control[STOP]`; the search polls that flag every few
thousand nodes. The same mechanism kills a ponder thread, which is why it is built this way now
rather than bolted on later. The invariant that makes shared state safe is that **only one
search runs at a time**: a ponder thread is stopped and joined before a real search begins.

"""

import numpy as np
from numba import njit
from numpy.typing import NDArray

from bb_board import (
    FLAG_EP,
    MAX_MOVES,
    MAX_PLY,
    META_EP,
    META_HALFMOVE,
    META_SIDE,
    in_check,
    make_move,
    move_from,
    move_promotion,
    move_to,
)
from bb_eval import evaluate
from bb_movegen import generate_legal, generate_pseudo
from bb_tables import KING, PAWN, QUEEN, U64, ZERO, ZOBRIST_EP, ZOBRIST_SIDE

MATE = 30000
MATE_THRESHOLD = 29000
INFINITY = 1 << 20

TT_BITS = 22
TT_SIZE = 1 << TT_BITS
TT_MASK = U64(TT_SIZE - 1)

EXACT = 0
LOWER = 1
UPPER = 2

STOP = 0
SOFT_STOP = 1
NODES = 2
BEST_MOVE = 3
BEST_SCORE = 4
COMPLETED_DEPTH = 5
CONTEMPT = 6
CONTROL_FIELDS = 8

CHECK_INTERVAL = 2048
MAX_GAME_KEYS = 512

FUTILITY_MARGIN = 95
DELTA_MARGIN = 200

# Read-only tables may stay global: numba freezes them as constants, which is what we want.
ORDER_VALUE = np.array([0, 100, 320, 330, 500, 900, 20000], dtype=np.int64)

LMR: NDArray[np.int64] = np.zeros((64, 64), dtype=np.int64)
for _depth in range(2, 64):
    for _index in range(1, 64):
        if _depth >= 3 and _index >= 3:
            LMR[_depth][_index] = int(0.75 + np.log(_depth) * np.log(_index) / 2.25)

# Every jitted search function takes the same array bundle, in this order.
ARRAYS = (
    "uint64[:, :], int8[:, :], int64[:, :], uint64[:], "  # position stack
    "int32[:, :], int64[:, :], int32[:, :], int64[:, :, :], "  # moves, ordering, heuristics
    "uint64[:], int32[:], int64[:], int8[:], int8[:], "  # transposition table
    "uint64[:], int64[:]"  # game history, control
)


class State:
    """Every buffer the search writes. Allocated once; the jitted code mutates it in place."""

    def __init__(self) -> None:
        self.bb: NDArray[np.uint64] = np.zeros((MAX_PLY, 8), dtype=U64)
        self.mailbox: NDArray[np.int8] = np.zeros((MAX_PLY, 64), dtype=np.int8)
        self.meta: NDArray[np.int64] = np.zeros((MAX_PLY, 4), dtype=np.int64)
        self.zkey: NDArray[np.uint64] = np.zeros(MAX_PLY, dtype=U64)
        self.moves: NDArray[np.int32] = np.zeros((MAX_PLY, MAX_MOVES), dtype=np.int32)
        self.scores: NDArray[np.int64] = np.zeros((MAX_PLY, MAX_MOVES), dtype=np.int64)
        self.killers: NDArray[np.int32] = np.zeros((MAX_PLY, 2), dtype=np.int32)
        self.history: NDArray[np.int64] = np.zeros((2, 64, 64), dtype=np.int64)
        self.tt_key: NDArray[np.uint64] = np.zeros(TT_SIZE, dtype=U64)
        self.tt_move: NDArray[np.int32] = np.zeros(TT_SIZE, dtype=np.int32)
        self.tt_score: NDArray[np.int64] = np.zeros(TT_SIZE, dtype=np.int64)
        self.tt_depth: NDArray[np.int8] = np.zeros(TT_SIZE, dtype=np.int8)
        self.tt_flag: NDArray[np.int8] = np.zeros(TT_SIZE, dtype=np.int8)
        self.game_keys: NDArray[np.uint64] = np.zeros(MAX_GAME_KEYS, dtype=U64)
        self.control: NDArray[np.int64] = np.zeros(CONTROL_FIELDS, dtype=np.int64)

    def run(self, max_depth: int, game_count: int) -> None:
        """Search this state. The only entry point; results land in `self.control`."""
        search(
            self.bb, self.mailbox, self.meta, self.zkey,
            self.moves, self.scores, self.killers, self.history,
            self.tt_key, self.tt_move, self.tt_score, self.tt_depth, self.tt_flag,
            self.game_keys, self.control,
            max_depth, game_count,
        )


@njit(f"boolean({ARRAYS}, int64, int64)", cache=False, nogil=True)
def _repeated(
    bb: NDArray[np.uint64], mailbox: NDArray[np.int8], meta: NDArray[np.int64],
    zkey: NDArray[np.uint64], moves: NDArray[np.int32], scores: NDArray[np.int64],
    killers: NDArray[np.int32], history: NDArray[np.int64],
    tt_key: NDArray[np.uint64], tt_move: NDArray[np.int32], tt_score: NDArray[np.int64],
    tt_depth: NDArray[np.int8], tt_flag: NDArray[np.int8],
    game_keys: NDArray[np.uint64], control: NDArray[np.int64],
    ply: int, game_count: int,
) -> bool:
    """True if this position already occurred, in the search path or earlier in the game.

    The FEN we are handed carries no history, so the game keys recorded by the driver are the
    only way to know we are about to repeat. The referee claims threefold by itself.
    """
    halfmove = meta[ply, META_HALFMOVE]
    if halfmove < 4:
        return False
    key = zkey[ply]
    back = 2
    while back <= halfmove:
        index = ply - back
        if index >= 0:
            if zkey[index] == key:
                return True
        else:
            # game_keys holds every position of the game in order, ending at the root, so an
            # ancestor below the search stack sits at (game_count - 1) + (ply - back).
            game_index = game_count - 1 + index
            if game_index < 0:
                return False
            if game_keys[game_index] == key:
                return True
        back += 2
    return False


@njit(f"void({ARRAYS}, int64, int64, int32, int64)", cache=False, nogil=True)
def _score_moves(
    bb: NDArray[np.uint64], mailbox: NDArray[np.int8], meta: NDArray[np.int64],
    zkey: NDArray[np.uint64], moves: NDArray[np.int32], scores: NDArray[np.int64],
    killers: NDArray[np.int32], history: NDArray[np.int64],
    tt_key: NDArray[np.uint64], tt_move: NDArray[np.int32], tt_score: NDArray[np.int64],
    tt_depth: NDArray[np.int8], tt_flag: NDArray[np.int8],
    game_keys: NDArray[np.uint64], control: NDArray[np.int64],
    ply: int, count: int, hash_move: int, side: int,
) -> None:
    """Hash move, then captures by MVV-LVA, then promotions, killers and history."""
    for index in range(count):
        move = moves[ply, index]
        origin = move_from(move)
        target = move_to(move)
        if move == hash_move:
            scores[ply, index] = 1 << 40
            continue
        victim = mailbox[ply, target]
        if victim != 0 or (move & FLAG_EP) != 0:
            attacker = mailbox[ply, origin]
            gain = ORDER_VALUE[victim] if victim != 0 else ORDER_VALUE[PAWN]
            scores[ply, index] = (1 << 36) + gain * 64 - ORDER_VALUE[attacker]
        elif move_promotion(move) == QUEEN:
            scores[ply, index] = 1 << 35
        elif move == killers[ply, 0]:
            scores[ply, index] = 1 << 34
        elif move == killers[ply, 1]:
            scores[ply, index] = (1 << 34) - 1
        else:
            scores[ply, index] = history[side, origin, target]


@njit("int32(int32[:, :], int64[:, :], int64, int64, int64)", cache=False, nogil=True)
def _pick_move(
    moves: NDArray[np.int32], scores: NDArray[np.int64], ply: int, start: int, count: int
) -> int:
    """Selection sort one move at a time: most cutoffs land before the list is exhausted."""
    best = start
    for index in range(start + 1, count):
        if scores[ply, index] > scores[ply, best]:
            best = index
    if best != start:
        swap_move = moves[ply, start]
        moves[ply, start] = moves[ply, best]
        moves[ply, best] = swap_move
        swap_score = scores[ply, start]
        scores[ply, start] = scores[ply, best]
        scores[ply, best] = swap_score
    return int(moves[ply, start])


@njit(f"int64({ARRAYS}, int64, int64, int64)", cache=False, nogil=True)
def quiescence(
    bb: NDArray[np.uint64], mailbox: NDArray[np.int8], meta: NDArray[np.int64],
    zkey: NDArray[np.uint64], moves: NDArray[np.int32], scores: NDArray[np.int64],
    killers: NDArray[np.int32], history: NDArray[np.int64],
    tt_key: NDArray[np.uint64], tt_move: NDArray[np.int32], tt_score: NDArray[np.int64],
    tt_depth: NDArray[np.int8], tt_flag: NDArray[np.int8],
    game_keys: NDArray[np.uint64], control: NDArray[np.int64],
    alpha: int, beta: int, ply: int,
) -> int:
    """Captures and queen promotions only, so the evaluation is never read mid-exchange."""
    control[NODES] += 1
    if control[NODES] % CHECK_INTERVAL == 0 and control[STOP] != 0:
        return 0
    if ply + 2 >= MAX_PLY:
        return evaluate(bb, meta, ply)

    us = meta[ply, META_SIDE]
    checked = in_check(bb, ply, us)
    stand_pat = -MATE + ply
    if not checked:
        stand_pat = evaluate(bb, meta, ply)
        if stand_pat >= beta:
            return stand_pat
        if stand_pat > alpha:
            alpha = stand_pat

    count = generate_pseudo(bb, mailbox, meta, ply, moves[ply])
    _score_moves(bb, mailbox, meta, zkey, moves, scores, killers, history,
                 tt_key, tt_move, tt_score, tt_depth, tt_flag, game_keys, control,
                 ply, count, 0, us)

    best = stand_pat
    legal = 0
    for index in range(count):
        move = _pick_move(moves, scores, ply, index, count)
        target = move_to(move)
        is_capture = mailbox[ply, target] != 0 or (move & FLAG_EP) != 0
        if not checked and not is_capture and move_promotion(move) != QUEEN:
            continue
        if not checked and is_capture:
            victim = mailbox[ply, target]
            gain = ORDER_VALUE[victim] if victim != 0 else ORDER_VALUE[PAWN]
            if stand_pat + gain + DELTA_MARGIN < alpha:
                continue
        make_move(bb, mailbox, meta, zkey, ply, move)
        if in_check(bb, ply + 1, us):
            continue
        legal += 1
        score = -quiescence(bb, mailbox, meta, zkey, moves, scores, killers, history,
                            tt_key, tt_move, tt_score, tt_depth, tt_flag, game_keys, control,
                            -beta, -alpha, ply + 1)
        if score > best:
            best = score
            if score > alpha:
                alpha = score
                if alpha >= beta:
                    break
    if checked and legal == 0:
        return -MATE + ply
    return best


@njit(f"int64({ARRAYS}, int64, int64, int64, int64, boolean, int64)", cache=False, nogil=True)
def negamax(
    bb: NDArray[np.uint64], mailbox: NDArray[np.int8], meta: NDArray[np.int64],
    zkey: NDArray[np.uint64], moves: NDArray[np.int32], scores: NDArray[np.int64],
    killers: NDArray[np.int32], history: NDArray[np.int64],
    tt_key: NDArray[np.uint64], tt_move: NDArray[np.int32], tt_score: NDArray[np.int64],
    tt_depth: NDArray[np.int8], tt_flag: NDArray[np.int8],
    game_keys: NDArray[np.uint64], control: NDArray[np.int64],
    depth: int, alpha: int, beta: int, ply: int, allow_null: bool, game_count: int,
) -> int:
    control[NODES] += 1
    if control[NODES] % CHECK_INTERVAL == 0 and control[STOP] != 0:
        return 0

    is_pv = beta - alpha > 1
    us = meta[ply, META_SIDE]

    contempt = int(control[CONTEMPT])
    draw_score = contempt if (ply & 1) == 0 else -contempt

    if ply > 0:
        if meta[ply, META_HALFMOVE] >= 100:
            return draw_score
        if _repeated(bb, mailbox, meta, zkey, moves, scores, killers, history,
                     tt_key, tt_move, tt_score, tt_depth, tt_flag, game_keys, control,
                     ply, game_count):
            return draw_score
        if alpha < -MATE + ply:
            alpha = -MATE + ply
        if beta > MATE - ply - 1:
            beta = MATE - ply - 1
        if alpha >= beta:
            return alpha

    slot = zkey[ply] & TT_MASK
    hash_move = 0
    if tt_key[slot] == zkey[ply]:
        hash_move = tt_move[slot]
        if ply > 0 and not is_pv and tt_depth[slot] >= depth:
            stored = int(tt_score[slot])
            if stored > MATE_THRESHOLD:
                stored -= ply
            elif stored < -MATE_THRESHOLD:
                stored += ply
            flag = tt_flag[slot]
            if flag == EXACT:
                return stored
            if flag == LOWER and stored >= beta:
                return stored
            if flag == UPPER and stored <= alpha:
                return stored

    checked = in_check(bb, ply, us)
    if checked:
        depth += 1
    if depth <= 0 or ply + 2 >= MAX_PLY:
        return quiescence(bb, mailbox, meta, zkey, moves, scores, killers, history,
                          tt_key, tt_move, tt_score, tt_depth, tt_flag, game_keys, control,
                          alpha, beta, ply)

    static = 0
    if not checked:
        static = evaluate(bb, meta, ply)

    if not is_pv and not checked and -MATE_THRESHOLD < beta < MATE_THRESHOLD:
        if depth <= 4 and static - FUTILITY_MARGIN * depth >= beta:
            return static
        has_pieces = False
        for piece in range(2, KING):
            if (bb[ply, 1 + piece] & bb[ply, us]) != ZERO:
                has_pieces = True
        if allow_null and depth >= 3 and static >= beta and has_pieces:
            reduction = 2 + depth // 4
            # A null move is a pass: the same position with the other side to move.
            for slot_index in range(8):
                bb[ply + 1, slot_index] = bb[ply, slot_index]
            for square in range(64):
                mailbox[ply + 1, square] = mailbox[ply, square]
            key = zkey[ply] ^ ZOBRIST_SIDE
            if meta[ply, META_EP] >= 0:
                key ^= ZOBRIST_EP[meta[ply, META_EP] & 7]
            meta[ply + 1, META_SIDE] = 1 - us
            meta[ply + 1, 1] = meta[ply, 1]
            meta[ply + 1, META_EP] = -1
            meta[ply + 1, META_HALFMOVE] = meta[ply, META_HALFMOVE] + 1
            zkey[ply + 1] = key
            score = -negamax(bb, mailbox, meta, zkey, moves, scores, killers, history,
                             tt_key, tt_move, tt_score, tt_depth, tt_flag, game_keys, control,
                             depth - 1 - reduction, -beta, -beta + 1, ply + 1, False,
                             game_count)
            if control[STOP] != 0:
                return 0
            if score >= beta:
                return beta

    count = generate_legal(bb, mailbox, meta, zkey, ply, moves[ply])
    if count == 0:
        return -MATE + ply if checked else draw_score

    _score_moves(bb, mailbox, meta, zkey, moves, scores, killers, history,
                 tt_key, tt_move, tt_score, tt_depth, tt_flag, game_keys, control,
                 ply, count, hash_move, us)

    best_score = -INFINITY
    best_move = 0
    original_alpha = alpha

    for index in range(count):
        move = _pick_move(moves, scores, ply, index, count)
        target = move_to(move)
        is_capture = mailbox[ply, target] != 0 or (move & FLAG_EP) != 0
        is_quiet = not is_capture and move_promotion(move) == 0

        make_move(bb, mailbox, meta, zkey, ply, move)
        gives_check = in_check(bb, ply + 1, 1 - us)

        reduction = 0
        if is_quiet and not checked and not gives_check and depth >= 3 and index >= 3:
            capped_depth = depth if depth < 64 else 63
            capped_index = index if index < 64 else 63
            reduction = LMR[capped_depth, capped_index]
            if is_pv and reduction > 0:
                reduction -= 1
            if reduction > depth - 2:
                reduction = depth - 2
            if reduction < 0:
                reduction = 0

        if index == 0:
            score = -negamax(bb, mailbox, meta, zkey, moves, scores, killers, history,
                             tt_key, tt_move, tt_score, tt_depth, tt_flag, game_keys, control,
                             depth - 1, -beta, -alpha, ply + 1, True, game_count)
        else:
            score = -negamax(bb, mailbox, meta, zkey, moves, scores, killers, history,
                             tt_key, tt_move, tt_score, tt_depth, tt_flag, game_keys, control,
                             depth - 1 - reduction, -alpha - 1, -alpha, ply + 1, True,
                             game_count)
            if score > alpha and reduction > 0:
                score = -negamax(bb, mailbox, meta, zkey, moves, scores, killers, history,
                                 tt_key, tt_move, tt_score, tt_depth, tt_flag, game_keys,
                                 control, depth - 1, -alpha - 1, -alpha, ply + 1, True,
                                 game_count)
            if alpha < score < beta:
                score = -negamax(bb, mailbox, meta, zkey, moves, scores, killers, history,
                                 tt_key, tt_move, tt_score, tt_depth, tt_flag, game_keys,
                                 control, depth - 1, -beta, -alpha, ply + 1, True, game_count)

        if control[STOP] != 0:
            return 0

        if score > best_score:
            best_score = score
            best_move = move
            if ply == 0:
                control[BEST_MOVE] = move
                control[BEST_SCORE] = score
            if score > alpha:
                alpha = score
                if alpha >= beta:
                    if is_quiet:
                        if move != killers[ply, 0]:
                            killers[ply, 1] = killers[ply, 0]
                            killers[ply, 0] = move
                        history[us, move_from(move), move_to(move)] += depth * depth
                    break

    flag = EXACT
    if best_score <= original_alpha:
        flag = UPPER
    elif best_score >= beta:
        flag = LOWER
    stored_depth = depth if depth < 127 else 127
    if tt_key[slot] != zkey[ply] or stored_depth >= tt_depth[slot] or flag == EXACT:
        to_store = best_score
        if to_store > MATE_THRESHOLD:
            to_store += ply
        elif to_store < -MATE_THRESHOLD:
            to_store -= ply
        tt_key[slot] = zkey[ply]
        tt_move[slot] = best_move
        tt_score[slot] = to_store
        tt_depth[slot] = stored_depth
        tt_flag[slot] = flag
    return best_score


@njit(f"void({ARRAYS}, int64, int64)", cache=False, nogil=True)
def search(
    bb: NDArray[np.uint64], mailbox: NDArray[np.int8], meta: NDArray[np.int64],
    zkey: NDArray[np.uint64], moves: NDArray[np.int32], scores: NDArray[np.int64],
    killers: NDArray[np.int32], history: NDArray[np.int64],
    tt_key: NDArray[np.uint64], tt_move: NDArray[np.int32], tt_score: NDArray[np.int64],
    tt_depth: NDArray[np.int8], tt_flag: NDArray[np.int8],
    game_keys: NDArray[np.uint64], control: NDArray[np.int64],
    max_depth: int, game_count: int,
) -> None:
    """Iterative deepening with aspiration windows. Results land in `control`.

    A depth is only committed once it completes. An aborted iteration leaves control[BEST_MOVE]
    holding whatever the root had already proved better, which is the standard trade: a partial
    result at depth N is worth more than a complete one at N-1, but only when it improved.
    """
    control[NODES] = 0
    control[COMPLETED_DEPTH] = 0

    for ply in range(MAX_PLY):
        killers[ply, 0] = 0
        killers[ply, 1] = 0
    # Age the history rather than clearing it: ordering from the last move is still informative.
    for side in range(2):
        for origin in range(64):
            for target in range(64):
                history[side, origin, target] //= 8

    score = 0
    for depth in range(1, max_depth + 1):
        if depth <= 4:
            score = negamax(bb, mailbox, meta, zkey, moves, scores, killers, history,
                            tt_key, tt_move, tt_score, tt_depth, tt_flag, game_keys, control,
                            depth, -INFINITY, INFINITY, 0, True, game_count)
        else:
            window = 25
            while True:
                alpha = score - window
                beta = score + window
                score = negamax(bb, mailbox, meta, zkey, moves, scores, killers, history,
                                tt_key, tt_move, tt_score, tt_depth, tt_flag, game_keys,
                                control, depth, alpha, beta, 0, True, game_count)
                if control[STOP] != 0:
                    break
                if score <= alpha or score >= beta:
                    window *= 4
                    if window > 2000:
                        score = negamax(bb, mailbox, meta, zkey, moves, scores, killers,
                                        history, tt_key, tt_move, tt_score, tt_depth,
                                        tt_flag, game_keys, control, depth, -INFINITY,
                                        INFINITY, 0, True, game_count)
                        break
                else:
                    break
        if control[STOP] != 0:
            break
        control[COMPLETED_DEPTH] = depth
        if score > MATE_THRESHOLD or score < -MATE_THRESHOLD:
            break
        if control[SOFT_STOP] != 0:
            break
