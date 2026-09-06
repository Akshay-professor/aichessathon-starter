"""AI Chessathon submission entrypoint.

Phase 1: an alpha-beta searcher built on python-chess. This is a deliberate stopgap. It is
strong enough to be worth having live on the ladder while the jitted bitboard engine is built,
and it doubles as the fallback path once that engine lands.

The structure that matters and will survive every later phase:

* get_move never raises and never returns an illegal move. Every failure mode the platform
  recognises (illegal, crash, flag) loses the game outright, so the entrypoint is a seatbelt
  around the search rather than the search itself.
* The time budget is derived from the 300-ply adjudication cap, not from a guess at game
  length. A game that runs to the cap gives each side 150 moves out of 120s + 0.5s/move, which
  is about 1.3s per move. Budgeting as if the game lasts 40 moves is how agents flag here.
* Positions we have been asked about are remembered. The referee claims threefold draws by
  itself, so an agent that does not track repetitions can draw a won game without being told.
"""

import threading
import time
from typing import Final

import chess
import numpy as np

import bb_board
import bb_search

PLY_CAP: Final = 300
INCREMENT_MS: Final = 500.0
RESERVE_MS: Final = 250.0
NODES_PER_CLOCK_CHECK: Final = 2048

MATE: Final = 30_000
MATE_THRESHOLD: Final = 29_000
INFINITY: Final = 1 << 20
MAX_PLY: Final = 64

PIECE_MG: Final = {
    chess.PAWN: 82,
    chess.KNIGHT: 337,
    chess.BISHOP: 365,
    chess.ROOK: 477,
    chess.QUEEN: 1025,
    chess.KING: 0,
}
PIECE_EG: Final = {
    chess.PAWN: 94,
    chess.KNIGHT: 281,
    chess.BISHOP: 297,
    chess.ROOK: 512,
    chess.QUEEN: 936,
    chess.KING: 0,
}
# Phase weights sum to 24 in the opening; the evaluation interpolates on the remaining material.
PHASE_WEIGHT: Final = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 1,
    chess.ROOK: 2,
    chess.QUEEN: 4,
    chess.KING: 0,
}
TOTAL_PHASE: Final = 24

# Tables are written from white's point of view with a8 first, so square s indexes at s ^ 56.
# fmt: off
PAWN_MG_PST: Final = (
      0,   0,   0,   0,   0,   0,   0,   0,
     60,  70,  70,  70,  70,  70,  70,  60,
     20,  25,  35,  45,  45,  35,  25,  20,
      8,  12,  18,  32,  32,  18,  12,   8,
      2,   6,  10,  26,  26,   8,   6,   2,
      4,   0,  -4,   6,   6,  -8,   2,   4,
      4,   8,   8, -20, -20,  10,  10,   4,
      0,   0,   0,   0,   0,   0,   0,   0,
)
PAWN_EG_PST: Final = (
      0,   0,   0,   0,   0,   0,   0,   0,
    150, 145, 135, 120, 120, 135, 145, 150,
     90,  88,  76,  62,  62,  76,  88,  90,
     42,  36,  30,  24,  24,  30,  36,  42,
     18,  16,  10,   8,   8,  10,  16,  18,
      6,   6,   2,   4,   4,   2,   6,   6,
     10,   8,   8,  10,  10,   8,   8,  10,
      0,   0,   0,   0,   0,   0,   0,   0,
)
KNIGHT_PST: Final = (
    -60, -40, -25, -20, -20, -25, -40, -60,
    -35, -15,  10,  15,  15,  10, -15, -35,
    -20,  12,  28,  32,  32,  28,  12, -20,
    -15,  10,  30,  36,  36,  30,  10, -15,
    -16,   8,  28,  34,  34,  28,   8, -16,
    -22,   6,  20,  26,  26,  22,   8, -22,
    -35, -14,   4,  10,  10,   6, -12, -35,
    -60, -32, -22, -16, -16, -22, -32, -60,
)
BISHOP_PST: Final = (
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   4,   0,   0,   0,   0,   4, -10,
    -10,  10,  12,  12,  12,  12,  10, -10,
    -10,   4,  12,  18,  18,  12,   4, -10,
    -10,   6,  10,  18,  18,  10,   6, -10,
    -10,  12,  12,  12,  12,  12,  12, -10,
    -10,  14,   4,   4,   4,   4,  14, -10,
    -20, -10, -14, -12, -12, -14, -10, -20,
)
ROOK_PST: Final = (
      6,   8,  10,  12,  12,  10,   8,   6,
     14,  20,  22,  24,  24,  22,  20,  14,
     -2,   4,   6,   8,   8,   6,   4,  -2,
     -6,   0,   2,   4,   4,   2,   0,  -6,
     -8,  -2,   0,   2,   2,   0,  -2,  -8,
    -10,  -2,   0,   2,   2,   0,  -2, -10,
    -12,  -2,   0,   4,   4,   0,  -2, -12,
     -6,  -4,   4,  10,  10,   4,  -4,  -6,
)
QUEEN_PST: Final = (
    -20, -10, -10,  -4,  -4, -10, -10, -20,
    -10,   0,   4,   0,   0,   4,   0, -10,
    -10,   4,   6,   6,   6,   6,   4, -10,
     -4,   0,   6,   8,   8,   6,   0,  -4,
     -4,   2,   6,   8,   8,   6,   2,  -4,
    -10,   6,   6,   6,   6,   6,   6, -10,
    -10,   0,   6,   0,   0,   4,   0, -10,
    -20, -10, -10,  -4,  -4, -10, -10, -20,
)
KING_MG_PST: Final = (
    -60, -70, -70, -80, -80, -70, -70, -60,
    -50, -60, -60, -70, -70, -60, -60, -50,
    -40, -50, -50, -60, -60, -50, -50, -40,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
     20,  20,  -6,  -6,  -6,  -6,  20,  20,
     20,  30,  10,   0,   0,  10,  34,  22,
)
KING_EG_PST: Final = (
    -50, -30, -20, -20, -20, -20, -30, -50,
    -30, -10,  10,  16,  16,  10, -10, -30,
    -20,  10,  26,  32,  32,  26,  10, -20,
    -20,  16,  32,  40,  40,  32,  16, -20,
    -20,  16,  32,  40,  40,  32,  16, -20,
    -20,  10,  26,  32,  32,  26,  10, -20,
    -30, -16,   6,  12,  12,   6, -16, -30,
    -50, -34, -24, -20, -20, -24, -34, -50,
)
# fmt: on
PST_MG: Final = {
    chess.PAWN: PAWN_MG_PST,
    chess.KNIGHT: KNIGHT_PST,
    chess.BISHOP: BISHOP_PST,
    chess.ROOK: ROOK_PST,
    chess.QUEEN: QUEEN_PST,
    chess.KING: KING_MG_PST,
}
PST_EG: Final = {
    chess.PAWN: PAWN_EG_PST,
    chess.KNIGHT: KNIGHT_PST,
    chess.BISHOP: BISHOP_PST,
    chess.ROOK: ROOK_PST,
    chess.QUEEN: QUEEN_PST,
    chess.KING: KING_EG_PST,
}

BISHOP_PAIR: Final = 30
DOUBLED_PAWN: Final = -12
ISOLATED_PAWN: Final = -14
PASSED_PAWN_BY_RANK: Final = (0, 6, 12, 24, 44, 76, 120, 0)
ROOK_OPEN_FILE: Final = 22
ROOK_SEMI_OPEN_FILE: Final = 10
TEMPO: Final = 12

# MVV-LVA: victim value dominates, attacker value breaks ties in the aggressor's favour.
MVV_LVA_VICTIM: Final = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

FILE_MASKS: Final = tuple(chess.BB_FILES)
NEIGHBOUR_FILE_MASKS: Final = tuple(
    (chess.BB_FILES[index - 1] if index > 0 else 0)
    | (chess.BB_FILES[index + 1] if index < 7 else 0)
    for index in range(8)
)


def _passed_masks(color: chess.Color) -> tuple[int, ...]:
    """Squares ahead of each square on its own and adjacent files, from `color`'s side."""
    masks = []
    for square in range(64):
        file_index = chess.square_file(square)
        rank_index = chess.square_rank(square)
        span = FILE_MASKS[file_index] | NEIGHBOUR_FILE_MASKS[file_index]
        ahead = 0
        ranks = range(rank_index + 1, 8) if color == chess.WHITE else range(0, rank_index)
        for rank in ranks:
            ahead |= chess.BB_RANKS[rank]
        masks.append(span & ahead)
    return tuple(masks)


PASSED_MASKS: Final = {
    chess.WHITE: _passed_masks(chess.WHITE),
    chess.BLACK: _passed_masks(chess.BLACK),
}


class Timeout(Exception):
    """Raised inside the search when the move budget is spent."""


def evaluate(board: chess.Board) -> int:
    """Static evaluation in centipawns, from the side to move's point of view."""
    middlegame = 0
    endgame = 0
    phase = 0

    pawns = {
        chess.WHITE: board.pieces_mask(chess.PAWN, chess.WHITE),
        chess.BLACK: board.pieces_mask(chess.PAWN, chess.BLACK),
    }

    for color in (chess.WHITE, chess.BLACK):
        sign = 1 if color == chess.WHITE else -1
        own_pawns = pawns[color]
        enemy_pawns = pawns[not color]

        for piece_type in chess.PIECE_TYPES:
            squares = board.pieces_mask(piece_type, color)
            phase += PHASE_WEIGHT[piece_type] * chess.popcount(squares)
            for square in chess.scan_forward(squares):
                index = square ^ 56 if color == chess.WHITE else square
                middlegame += sign * (PIECE_MG[piece_type] + PST_MG[piece_type][index])
                endgame += sign * (PIECE_EG[piece_type] + PST_EG[piece_type][index])

        if chess.popcount(board.pieces_mask(chess.BISHOP, color)) >= 2:
            middlegame += sign * BISHOP_PAIR
            endgame += sign * BISHOP_PAIR

        for square in chess.scan_forward(own_pawns):
            file_index = chess.square_file(square)
            if chess.popcount(own_pawns & FILE_MASKS[file_index]) > 1:
                middlegame += sign * DOUBLED_PAWN
                endgame += sign * DOUBLED_PAWN
            if not own_pawns & NEIGHBOUR_FILE_MASKS[file_index]:
                middlegame += sign * ISOLATED_PAWN
                endgame += sign * ISOLATED_PAWN
            if not enemy_pawns & PASSED_MASKS[color][square]:
                rank = chess.square_rank(square)
                relative = rank if color == chess.WHITE else 7 - rank
                bonus = PASSED_PAWN_BY_RANK[relative]
                middlegame += sign * (bonus // 2)
                endgame += sign * bonus

        for square in chess.scan_forward(board.pieces_mask(chess.ROOK, color)):
            file_mask = FILE_MASKS[chess.square_file(square)]
            if not (own_pawns | enemy_pawns) & file_mask:
                middlegame += sign * ROOK_OPEN_FILE
            elif not own_pawns & file_mask:
                middlegame += sign * ROOK_SEMI_OPEN_FILE

    phase = min(phase, TOTAL_PHASE)
    score = (middlegame * phase + endgame * (TOTAL_PHASE - phase)) // TOTAL_PHASE
    if board.turn == chess.BLACK:
        score = -score
    return score + TEMPO


EXACT: Final = 0
LOWER: Final = 1
UPPER: Final = 2

TT_LIMIT: Final = 1 << 20
FUTILITY_MARGIN: Final = 95
DELTA_MARGIN: Final = 200
NON_PAWN_TYPES: Final = (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)


class Searcher:
    """Negamax with alpha-beta, iterative deepening and the usual ordering heuristics.

    The transposition table and history scores persist between moves in a game, which is worth
    a ply on its own: the platform keeps the process alive until the game ends.
    """

    def __init__(self) -> None:
        self.tt: dict[object, tuple[int, int, int, chess.Move | None]] = {}
        self.killers: list[list[chess.Move | None]] = [[None, None] for _ in range(MAX_PLY + 8)]
        self.history: dict[tuple[bool, int, int], int] = {}
        self.seen: dict[object, int] = {}
        self.nodes = 0
        self.deadline = 0.0
        self.best_move: chess.Move | None = None
        self.best_score = 0

    def prepare(self, deadline: float) -> None:
        self.nodes = 0
        self.deadline = deadline
        self.best_move = None
        self.best_score = 0
        if len(self.tt) > TT_LIMIT:
            self.tt.clear()
        for scores in self.killers:
            scores[0] = None
            scores[1] = None

    def _tick(self) -> None:
        self.nodes += 1
        if self.nodes % NODES_PER_CLOCK_CHECK == 0 and time.monotonic() >= self.deadline:
            raise Timeout

    def _score_move(
        self, board: chess.Board, move: chess.Move, ply: int, tt_move: chess.Move | None
    ) -> int:
        if move == tt_move:
            return 1 << 24
        if board.is_capture(move):
            victim = board.piece_type_at(move.to_square)
            victim_value = MVV_LVA_VICTIM[victim] if victim is not None else 100
            attacker = board.piece_type_at(move.from_square)
            attacker_value = MVV_LVA_VICTIM[attacker] if attacker is not None else 0
            return (1 << 22) + victim_value * 16 - attacker_value
        if move.promotion is not None:
            return (1 << 21) + move.promotion
        killers = self.killers[ply]
        if move == killers[0]:
            return 1 << 20
        if move == killers[1]:
            return (1 << 20) - 1
        return self.history.get((board.turn, move.from_square, move.to_square), 0)

    def _ordered(
        self, board: chess.Board, moves: list[chess.Move], ply: int, tt_move: chess.Move | None
    ) -> list[chess.Move]:
        moves.sort(key=lambda move: self._score_move(board, move, ply, tt_move), reverse=True)
        return moves

    def _repeated(self, key: object) -> bool:
        return self.seen.get(key, 0) > 0

    def _has_non_pawn_material(self, board: chess.Board) -> bool:
        return any(board.pieces_mask(piece, board.turn) for piece in NON_PAWN_TYPES)

    def quiescence(self, board: chess.Board, alpha: int, beta: int, ply: int) -> int:
        """Search captures only, so the evaluation is never read mid-exchange."""
        self._tick()
        in_check = board.is_check()
        if not in_check:
            stand_pat = evaluate(board)
            if stand_pat >= beta:
                return stand_pat
            alpha = max(alpha, stand_pat)
        else:
            stand_pat = -MATE + ply

        if ply >= MAX_PLY:
            return stand_pat

        moves = [
            move
            for move in board.legal_moves
            if in_check or board.is_capture(move) or move.promotion is not None
        ]
        if not moves:
            return -MATE + ply if in_check else stand_pat

        best = stand_pat
        for move in self._ordered(board, moves, ply, None):
            if not in_check and board.is_capture(move):
                victim = board.piece_type_at(move.to_square)
                gain = MVV_LVA_VICTIM[victim] if victim is not None else 100
                if stand_pat + gain + DELTA_MARGIN < alpha:
                    continue
            board.push(move)
            score = -self.quiescence(board, -beta, -alpha, ply + 1)
            board.pop()
            if score > best:
                best = score
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break
        return best

    def negamax(
        self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int, allow_null: bool
    ) -> int:
        self._tick()
        is_pv = beta - alpha > 1
        key = board._transposition_key()

        if ply > 0:
            if self._repeated(key) or board.halfmove_clock >= 100:
                return 0
            alpha = max(alpha, -MATE + ply)
            beta = min(beta, MATE - ply - 1)
            if alpha >= beta:
                return alpha

        tt_move: chess.Move | None = None
        entry = self.tt.get(key)
        if entry is not None:
            stored_depth, stored_score, flag, tt_move = entry
            if ply > 0 and stored_depth >= depth and not is_pv:
                if flag == EXACT:
                    return stored_score
                if flag == LOWER and stored_score >= beta:
                    return stored_score
                if flag == UPPER and stored_score <= alpha:
                    return stored_score

        in_check = board.is_check()
        if in_check:
            depth += 1
        if depth <= 0 or ply >= MAX_PLY:
            return self.quiescence(board, alpha, beta, ply)

        static = evaluate(board) if not in_check else 0

        if not is_pv and not in_check and abs(beta) < MATE_THRESHOLD:
            if depth <= 4 and static - FUTILITY_MARGIN * depth >= beta:
                return static
            if (
                allow_null
                and depth >= 3
                and static >= beta
                and self._has_non_pawn_material(board)
            ):
                reduction = 2 + depth // 4
                board.push(chess.Move.null())
                self.seen[key] = self.seen.get(key, 0) + 1
                score = -self.negamax(
                    board, depth - 1 - reduction, -beta, -beta + 1, ply + 1, False
                )
                self.seen[key] -= 1
                board.pop()
                if score >= beta:
                    return beta

        moves = list(board.legal_moves)
        if not moves:
            return -MATE + ply if in_check else 0

        self.seen[key] = self.seen.get(key, 0) + 1
        best_score = -INFINITY
        best_move: chess.Move | None = None
        original_alpha = alpha
        try:
            for index, move in enumerate(self._ordered(board, moves, ply, tt_move)):
                is_quiet = not board.is_capture(move) and move.promotion is None
                board.push(move)
                gives_check = board.is_check()

                reduction = 0
                if is_quiet and depth >= 3 and index >= 3 and not in_check and not gives_check:
                    reduction = 1 if index < 6 else 2
                    reduction = min(reduction, depth - 2)

                if index == 0:
                    score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1, True)
                else:
                    score = -self.negamax(
                        board, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1, True
                    )
                    if alpha < score < beta:
                        score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1, True)
                board.pop()

                if score > best_score:
                    best_score = score
                    best_move = move
                    if ply == 0:
                        self.best_move = move
                        self.best_score = score
                if score > alpha:
                    alpha = score
                if alpha >= beta:
                    if is_quiet:
                        killers = self.killers[ply]
                        if move != killers[0]:
                            killers[1] = killers[0]
                            killers[0] = move
                        slot = (board.turn, move.from_square, move.to_square)
                        self.history[slot] = self.history.get(slot, 0) + depth * depth
                    break
        finally:
            self.seen[key] -= 1

        if best_score <= original_alpha:
            flag = UPPER
        elif best_score >= beta:
            flag = LOWER
        else:
            flag = EXACT
        self.tt[key] = (depth, best_score, flag, best_move)
        return best_score

    def search_root(self, board: chess.Board, depth: int) -> None:
        """One iteration of iterative deepening. Updates best_move as soon as it improves."""
        self.negamax(board, depth, -INFINITY, INFINITY, 0, True)


_SEARCHER = Searcher()
_GAME_SEEN: dict[object, int] = {}
_CALLS = 0


def _budget_ms(time_left_ms: int, plies_played: int) -> float:
    """Milliseconds to spend on this move.

    Sized against the 300-ply adjudication cap rather than a guessed game length. A game that
    runs to the cap costs us 150 moves, so the honest average here is about 1.3s, not the 3s
    that dividing the base clock by 40 suggests.
    """
    moves_left = max(10.0, (PLY_CAP - plies_played) / 2.0)
    soft = time_left_ms / moves_left + INCREMENT_MS * 0.75
    hard = max(0.0, time_left_ms - RESERVE_MS) * 0.35
    return max(15.0, min(soft, hard))


def _fallback_move(board: chess.Board) -> chess.Move:
    """A legal move chosen without the search, for when the search cannot be trusted."""
    best = None
    best_gain = -1
    for move in board.legal_moves:
        gain = 0
        if board.is_capture(move):
            victim = board.piece_type_at(move.to_square)
            gain = MVV_LVA_VICTIM[victim] if victim is not None else 100
        if gain > best_gain:
            best_gain = gain
            best = move
    if best is None:
        raise ValueError("no legal moves")
    return best


def _think(board: chess.Board, budget: float) -> chess.Move | None:
    """The reference python-chess search, kept as the fallback when the engine cannot answer."""
    started = time.monotonic()
    deadline = started + budget / 1000.0

    key = board._transposition_key()
    _GAME_SEEN[key] = _GAME_SEEN.get(key, 0) + 1

    # The search gets its own board. A Timeout unwinds through negamax with moves still
    # pushed, so the caller's board must never be the one the search walks.
    scratch = board.copy(stack=False)
    searcher = _SEARCHER
    searcher.prepare(deadline)
    searcher.seen = dict(_GAME_SEEN)

    best: chess.Move | None = None
    best_score = -INFINITY
    depth = 0

    for depth in range(1, MAX_PLY):
        searcher.best_move = None
        try:
            searcher.search_root(scratch, depth)
        except Timeout:
            # A partial iteration is only trusted when it beat the last completed one.
            if searcher.best_move is not None and searcher.best_score > best_score:
                best = searcher.best_move
            break
        if searcher.best_move is not None:
            best = searcher.best_move
            best_score = searcher.best_score
        if abs(best_score) > MATE_THRESHOLD:
            break
        # Starting another iteration is only worth it with half the budget still unspent.
        if (time.monotonic() - started) * 1000.0 > budget * 0.5:
            break

    elapsed = (time.monotonic() - started) * 1000.0
    print(
        f"fallback depth {depth} score {best_score} "
        f"nodes {searcher.nodes} {elapsed:.0f}ms/{budget:.0f}ms"
    )
    return best


MAX_SEARCH_DEPTH: Final = 48
# Inside this many plies of the 300-ply cap, the referee's material adjudication starts to
# decide the game, so a draw stops being worth zero. See _contempt.
ADJUDICATION_HORIZON: Final = 60
ADJUDICATION_CONTEMPT: Final = 150
MATERIAL_MARGIN: Final = 50
FALLBACK_SHARE: Final = 0.10
# An iteration started just under the budget can run several times as long as the one before
# it, so new iterations stop early and the hard abort lands on the budget itself. Overshooting
# the budget is how agents flag, and a flag is a loss.
SOFT_SHARE: Final = 0.45

_STATE = bb_search.State()
_POSITION = bb_board.Position()
_HISTORY_POSITION = bb_board.Position()

# The FEN carries no history, but the referee claims threefold and fifty-move draws on the
# whole game. We are only shown our own turns, so the positions in between are reconstructed
# by replaying the one legal move that leads from our last position to the one we are handed.
_GAME_BOARD: chess.Board | None = None
_HISTORY_KEYS: list[int] = []


def _position_key(board: chess.Board) -> int:
    """The engine's own zobrist for an arbitrary board."""
    bb_board.set_fen(_HISTORY_POSITION, board.fen())
    return int(_HISTORY_POSITION.zkey[0])


def _sync_history(board: chess.Board) -> None:
    """Extend the recorded game with the opponent's reply, or start again if we lost track."""
    global _GAME_BOARD
    if _GAME_BOARD is not None:
        target = board._transposition_key()
        for move in _GAME_BOARD.legal_moves:
            _GAME_BOARD.push(move)
            if _GAME_BOARD._transposition_key() == target:
                _HISTORY_KEYS.append(_position_key(_GAME_BOARD))
                return
            _GAME_BOARD.pop()
    _GAME_BOARD = board.copy(stack=False)
    _HISTORY_KEYS.clear()
    _HISTORY_KEYS.append(_position_key(_GAME_BOARD))


def _commit_move(move: chess.Move) -> None:
    """Record the move we are about to play, so the next call can extend from it."""
    if _GAME_BOARD is not None and _GAME_BOARD.is_legal(move):
        _GAME_BOARD.push(move)
        _HISTORY_KEYS.append(_position_key(_GAME_BOARD))


def _contempt(board: chess.Board, plies_played: int) -> int:
    """What a draw is worth to us, in centipawns, from our own point of view.

    Normally zero. Near the ply cap it is not: the referee adjudicates 300 plies on material
    alone, so if we are ahead a repetition throws away a win the referee would have given us,
    and if we are behind a repetition rescues half a point from a loss. The evaluation cannot
    see this, because nothing in the position says the game is about to be scored on material.
    """
    if PLY_CAP - plies_played > ADJUDICATION_HORIZON:
        return 0
    balance = 0
    for piece, value in MVV_LVA_VICTIM.items():
        if piece == chess.KING:
            continue
        balance += value * (
            len(board.pieces(piece, board.turn)) - len(board.pieces(piece, not board.turn))
        )
    if balance > MATERIAL_MARGIN:
        return -ADJUDICATION_CONTEMPT
    if balance < -MATERIAL_MARGIN:
        return ADJUDICATION_CONTEMPT
    return 0


def _raise_flag(control: "object", index: int) -> None:
    control[index] = 1  # type: ignore[index]


def _engine_move(fen: str, budget_ms: float, time_left_ms: int) -> str | None:
    """Search with the jitted engine and return its move in UCI, or None if it produced none.

    The clock is enforced from outside. `bb_search.search` is compiled nogil, so these timer
    threads keep running while it executes: one asks it to stop starting new iterations, the
    other aborts it outright.
    """
    bb_board.set_fen(_POSITION, fen)
    _STATE.bb[0] = _POSITION.bb[0]
    _STATE.mailbox[0] = _POSITION.mailbox[0]
    _STATE.meta[0] = _POSITION.meta[0]
    _STATE.zkey[0] = _POSITION.zkey[0]

    recorded = _HISTORY_KEYS[-bb_search.MAX_GAME_KEYS :]
    for index, key in enumerate(recorded):
        _STATE.game_keys[index] = np.uint64(key)

    control = _STATE.control
    control[bb_search.STOP] = 0
    control[bb_search.SOFT_STOP] = 0
    control[bb_search.BEST_MOVE] = 0

    soft_ms = budget_ms * SOFT_SHARE
    soft = threading.Timer(soft_ms / 1000.0, _raise_flag, (control, bb_search.SOFT_STOP))
    hard = threading.Timer(budget_ms / 1000.0, _raise_flag, (control, bb_search.STOP))
    soft.daemon = True
    hard.daemon = True

    started = time.monotonic()
    soft.start()
    hard.start()
    try:
        _STATE.run(MAX_SEARCH_DEPTH, len(recorded))
    finally:
        soft.cancel()
        hard.cancel()

    elapsed = (time.monotonic() - started) * 1000.0
    nodes = int(control[bb_search.NODES])
    rate = nodes / elapsed if elapsed > 0 else 0.0
    print(
        f"depth {int(control[bb_search.COMPLETED_DEPTH])} "
        f"score {int(control[bb_search.BEST_SCORE])} nodes {nodes} "
        f"{elapsed:.0f}ms/{budget_ms:.0f}ms {rate:.0f}knps"
    )

    move = int(control[bb_search.BEST_MOVE])
    return bb_board.move_uci(move) if move != 0 else None


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation.

    This never raises and never returns a move that is not legal in `fen`. Illegal moves,
    crashes and flag falls each lose the game outright, so a defect anywhere below has to cost
    one bad move rather than the point. Three layers, in order: the jitted engine, the
    python-chess reference search, then any legal move.
    """
    try:
        board = chess.Board(fen)
        fallback = _fallback_move(board)
    except (ValueError, IndexError) as error:
        print(f"cannot read {fen!r}: {error!r}")
        return "0000"

    try:
        _sync_history(board)
        plies_played = len(_HISTORY_KEYS) - 1
        _STATE.control[bb_search.CONTEMPT] = _contempt(board, plies_played)
    except Exception as error:  # history is an optimisation; never let it end the game
        print(f"history tracking failed: {error!r}")
        plies_played = 0

    budget = _budget_ms(time_left_ms, plies_played)
    chosen: chess.Move | None = None

    try:
        uci = _engine_move(fen, budget, time_left_ms)
        if uci is not None:
            candidate = chess.Move.from_uci(uci)
            if board.is_legal(candidate):
                chosen = candidate
            else:
                print(f"engine returned illegal {uci!r}; falling back")
    except Exception as error:
        print(f"engine failed, falling back: {error!r}")

    if chosen is None:
        # The engine has already spent its budget, so the fallback gets a small slice of what
        # is left. A worse move is survivable; a flag is not.
        try:
            spare = max(0.0, time_left_ms - RESERVE_MS) * FALLBACK_SHARE
            move = _think(board, spare)
            if move is not None and board.is_legal(move):
                chosen = move
        except Exception as error:
            print(f"fallback search failed: {error!r}")

    if chosen is None:
        chosen = fallback

    try:
        _commit_move(chosen)
    except Exception as error:
        print(f"could not record {chosen.uci()}: {error!r}")

    return chosen.uci()


# Compile every jitted signature and touch the search tables now, inside the 60s init budget.
# Work deferred to the first get_move would come out of the match clock instead.
_HISTORY_KEYS.append(_position_key(chess.Board()))
_engine_move(chess.STARTING_FEN, 40.0, 120_000)
_HISTORY_KEYS.clear()
_GAME_BOARD = None
