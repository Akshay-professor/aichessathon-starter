"""Legal move generation.

Generation is deliberately in two stages: pseudo-legal moves, then a filter that plays each one
and asks whether our king is attacked. That is slower than deriving pins and checkers up front,
and it is the right first version, because it is simple enough to be obviously correct and
`tools/perft.py` proves it. The faster generator replaces it only once perft is exact, with
perft standing as the regression net.
"""

import numpy as np
from numba import njit
from numpy.typing import NDArray

from bb_board import (
    CASTLE_BK,
    CASTLE_BQ,
    CASTLE_WK,
    CASTLE_WQ,
    FLAG_CASTLE,
    FLAG_DOUBLE,
    FLAG_EP,
    META_CASTLE,
    META_EP,
    META_SIDE,
    encode,
    in_check,
    is_attacked,
    lsb,
    make_move,
)
from bb_tables import (
    BISHOP,
    KING,
    KING_ATTACKS,
    KNIGHT,
    KNIGHT_ATTACKS,
    ONE,
    PAWN,
    PAWN_ATTACKS,
    QUEEN,
    ROOK,
    U64,
    WHITE,
    ZERO,
    bishop_attacks,
    queen_attacks,
    rook_attacks,
)

RANK_3 = U64(0x0000000000FF0000)
RANK_6 = U64(0x0000FF0000000000)
PROMOTION_RANKS = U64(0xFF000000000000FF)

PROMOTION_PIECES = np.array([QUEEN, ROOK, BISHOP, KNIGHT], dtype=np.int64)


@njit("int64(uint64[:, :], int8[:, :], int64[:, :], int64, int32[:])", cache=False)
def generate_pseudo(
    bb: NDArray[np.uint64],
    mailbox: NDArray[np.int8],
    meta: NDArray[np.int64],
    ply: int,
    moves: NDArray[np.int32],
) -> int:
    """Fill `moves` with pseudo-legal moves and return how many. King safety is not checked."""
    us = meta[ply, META_SIDE]
    them = 1 - us
    mine = bb[ply, us]
    theirs = bb[ply, them]
    occupied = mine | theirs
    empty = ~occupied
    count = 0

    pawns = bb[ply, 1 + PAWN] & mine
    if us == WHITE:
        single = (pawns << U64(1 * 8)) & empty
        double = ((single & RANK_3) << U64(8)) & empty
        push_step = 8
    else:
        single = (pawns >> U64(8)) & empty
        double = ((single & RANK_6) >> U64(8)) & empty
        push_step = -8

    remaining = single
    while remaining != ZERO:
        target = lsb(remaining)
        remaining &= remaining - ONE
        origin = target - push_step
        if (ONE << U64(target)) & PROMOTION_RANKS:
            for index in range(4):
                moves[count] = encode(origin, target, PROMOTION_PIECES[index], 0)
                count += 1
        else:
            moves[count] = encode(origin, target, 0, 0)
            count += 1

    remaining = double
    while remaining != ZERO:
        target = lsb(remaining)
        remaining &= remaining - ONE
        moves[count] = encode(target - 2 * push_step, target, 0, FLAG_DOUBLE)
        count += 1

    ep_square = meta[ply, META_EP]
    remaining = pawns
    while remaining != ZERO:
        origin = lsb(remaining)
        remaining &= remaining - ONE
        captures = PAWN_ATTACKS[us][origin] & theirs
        while captures != ZERO:
            target = lsb(captures)
            captures &= captures - ONE
            if (ONE << U64(target)) & PROMOTION_RANKS:
                for index in range(4):
                    moves[count] = encode(origin, target, PROMOTION_PIECES[index], 0)
                    count += 1
            else:
                moves[count] = encode(origin, target, 0, 0)
                count += 1
        if ep_square >= 0 and (PAWN_ATTACKS[us][origin] & (ONE << U64(ep_square))) != ZERO:
            moves[count] = encode(origin, ep_square, 0, FLAG_EP)
            count += 1

    for piece in range(KNIGHT, KING + 1):
        pieces = bb[ply, 1 + piece] & mine
        while pieces != ZERO:
            origin = lsb(pieces)
            pieces &= pieces - ONE
            if piece == KNIGHT:
                targets = KNIGHT_ATTACKS[origin]
            elif piece == BISHOP:
                targets = bishop_attacks(origin, occupied)
            elif piece == ROOK:
                targets = rook_attacks(origin, occupied)
            elif piece == QUEEN:
                targets = queen_attacks(origin, occupied)
            else:
                targets = KING_ATTACKS[origin]
            targets &= ~mine
            while targets != ZERO:
                target = lsb(targets)
                targets &= targets - ONE
                moves[count] = encode(origin, target, 0, 0)
                count += 1

    rights = meta[ply, META_CASTLE]
    if rights != 0:
        home = 0 if us == WHITE else 56
        king_side = CASTLE_WK if us == WHITE else CASTLE_BK
        queen_side = CASTLE_WQ if us == WHITE else CASTLE_BQ
        origin = home + 4
        if (rights & king_side) != 0:
            gap = (ONE << U64(home + 5)) | (ONE << U64(home + 6))
            if (occupied & gap) == ZERO:
                safe = not is_attacked(bb, ply, origin, them, occupied)
                safe = safe and not is_attacked(bb, ply, home + 5, them, occupied)
                safe = safe and not is_attacked(bb, ply, home + 6, them, occupied)
                if safe:
                    moves[count] = encode(origin, home + 6, 0, FLAG_CASTLE)
                    count += 1
        if (rights & queen_side) != 0:
            gap = (ONE << U64(home + 1)) | (ONE << U64(home + 2)) | (ONE << U64(home + 3))
            if (occupied & gap) == ZERO:
                safe = not is_attacked(bb, ply, origin, them, occupied)
                safe = safe and not is_attacked(bb, ply, home + 3, them, occupied)
                safe = safe and not is_attacked(bb, ply, home + 2, them, occupied)
                if safe:
                    moves[count] = encode(origin, home + 2, 0, FLAG_CASTLE)
                    count += 1

    return count


@njit(
    "int64(uint64[:, :], int8[:, :], int64[:, :], uint64[:], int64, int32[:])",
    cache=False,
)
def generate_legal(
    bb: NDArray[np.uint64],
    mailbox: NDArray[np.int8],
    meta: NDArray[np.int64],
    zkey: NDArray[np.uint64],
    ply: int,
    moves: NDArray[np.int32],
) -> int:
    """Pseudo-legal moves, minus the ones that leave our own king attacked."""
    us = meta[ply, META_SIDE]
    total = generate_pseudo(bb, mailbox, meta, ply, moves)
    kept = 0
    for index in range(total):
        candidate = moves[index]
        make_move(bb, mailbox, meta, zkey, ply, candidate)
        if not in_check(bb, ply + 1, us):
            moves[kept] = candidate
            kept += 1
    return kept
