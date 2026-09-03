"""Tapered evaluation over the bitboard arrays.

Everything is int centipawns from the side to move's point of view, and every table is built at
import. The evaluation is deliberately conventional — material, piece-square tables, pawn
structure, mobility, king shelter — because its job right now is to be a good enough reference
to measure a tuned or learned replacement against. Phase 4 replaces the numbers, not the shape:
the weights live in arrays so a Texel run can overwrite them without touching the search.
"""

import numpy as np
from numba import njit
from numpy.typing import NDArray

from bb_board import META_SIDE, lsb, popcount
from bb_tables import (
    BISHOP,
    BLACK,
    KING,
    KNIGHT,
    KNIGHT_ATTACKS,
    ONE,
    PAWN,
    QUEEN,
    ROOK,
    U64,
    WHITE,
    ZERO,
    bishop_attacks,
    queen_attacks,
    rook_attacks,
)

PIECE_MG = np.array([0, 82, 337, 365, 477, 1025, 0], dtype=np.int32)
PIECE_EG = np.array([0, 94, 281, 297, 512, 936, 0], dtype=np.int32)
PHASE_WEIGHT = np.array([0, 0, 1, 1, 2, 4, 0], dtype=np.int32)
TOTAL_PHASE = 24

MOBILITY_MG = np.array([0, 0, 4, 5, 2, 1, 0], dtype=np.int32)
MOBILITY_EG = np.array([0, 0, 4, 5, 4, 2, 0], dtype=np.int32)

BISHOP_PAIR = 30
DOUBLED_PAWN = -12
ISOLATED_PAWN = -14
ROOK_OPEN_FILE = 22
ROOK_SEMI_OPEN_FILE = 10
SHIELD_PAWN = 11
TEMPO = 12

PASSED_BY_RANK = np.array([0, 6, 12, 24, 44, 76, 120, 0], dtype=np.int32)

# Tables read a8 first, so a white piece on square s indexes at s ^ 56 and a black piece at s.
# fmt: off
_PAWN_MG = (
      0,   0,   0,   0,   0,   0,   0,   0,
     60,  70,  70,  70,  70,  70,  70,  60,
     20,  25,  35,  45,  45,  35,  25,  20,
      8,  12,  18,  32,  32,  18,  12,   8,
      2,   6,  10,  26,  26,   8,   6,   2,
      4,   0,  -4,   6,   6,  -8,   2,   4,
      4,   8,   8, -20, -20,  10,  10,   4,
      0,   0,   0,   0,   0,   0,   0,   0,
)
_PAWN_EG = (
      0,   0,   0,   0,   0,   0,   0,   0,
    150, 145, 135, 120, 120, 135, 145, 150,
     90,  88,  76,  62,  62,  76,  88,  90,
     42,  36,  30,  24,  24,  30,  36,  42,
     18,  16,  10,   8,   8,  10,  16,  18,
      6,   6,   2,   4,   4,   2,   6,   6,
     10,   8,   8,  10,  10,   8,   8,  10,
      0,   0,   0,   0,   0,   0,   0,   0,
)
_KNIGHT = (
    -60, -40, -25, -20, -20, -25, -40, -60,
    -35, -15,  10,  15,  15,  10, -15, -35,
    -20,  12,  28,  32,  32,  28,  12, -20,
    -15,  10,  30,  36,  36,  30,  10, -15,
    -16,   8,  28,  34,  34,  28,   8, -16,
    -22,   6,  20,  26,  26,  22,   8, -22,
    -35, -14,   4,  10,  10,   6, -12, -35,
    -60, -32, -22, -16, -16, -22, -32, -60,
)
_BISHOP = (
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   4,   0,   0,   0,   0,   4, -10,
    -10,  10,  12,  12,  12,  12,  10, -10,
    -10,   4,  12,  18,  18,  12,   4, -10,
    -10,   6,  10,  18,  18,  10,   6, -10,
    -10,  12,  12,  12,  12,  12,  12, -10,
    -10,  14,   4,   4,   4,   4,  14, -10,
    -20, -10, -14, -12, -12, -14, -10, -20,
)
_ROOK = (
      6,   8,  10,  12,  12,  10,   8,   6,
     14,  20,  22,  24,  24,  22,  20,  14,
     -2,   4,   6,   8,   8,   6,   4,  -2,
     -6,   0,   2,   4,   4,   2,   0,  -6,
     -8,  -2,   0,   2,   2,   0,  -2,  -8,
    -10,  -2,   0,   2,   2,   0,  -2, -10,
    -12,  -2,   0,   4,   4,   0,  -2, -12,
     -6,  -4,   4,  10,  10,   4,  -4,  -6,
)
_QUEEN = (
    -20, -10, -10,  -4,  -4, -10, -10, -20,
    -10,   0,   4,   0,   0,   4,   0, -10,
    -10,   4,   6,   6,   6,   6,   4, -10,
     -4,   0,   6,   8,   8,   6,   0,  -4,
     -4,   2,   6,   8,   8,   6,   2,  -4,
    -10,   6,   6,   6,   6,   6,   6, -10,
    -10,   0,   6,   0,   0,   4,   0, -10,
    -20, -10, -10,  -4,  -4, -10, -10, -20,
)
_KING_MG = (
    -60, -70, -70, -80, -80, -70, -70, -60,
    -50, -60, -60, -70, -70, -60, -60, -50,
    -40, -50, -50, -60, -60, -50, -50, -40,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
     20,  20,  -6,  -6,  -6,  -6,  20,  20,
     20,  30,  10,   0,   0,  10,  34,  22,
)
_KING_EG = (
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

PST_MG: NDArray[np.int32] = np.zeros((7, 64), dtype=np.int32)
PST_EG: NDArray[np.int32] = np.zeros((7, 64), dtype=np.int32)
for _piece, _mg, _eg in (
    (PAWN, _PAWN_MG, _PAWN_EG),
    (KNIGHT, _KNIGHT, _KNIGHT),
    (BISHOP, _BISHOP, _BISHOP),
    (ROOK, _ROOK, _ROOK),
    (QUEEN, _QUEEN, _QUEEN),
    (KING, _KING_MG, _KING_EG),
):
    PST_MG[_piece] = np.array(_mg, dtype=np.int32)
    PST_EG[_piece] = np.array(_eg, dtype=np.int32)

FILE_BB: NDArray[np.uint64] = np.array(
    [U64(0x0101010101010101) << U64(f) for f in range(8)], dtype=U64
)
ADJACENT_FILES: NDArray[np.uint64] = np.array(
    [
        (FILE_BB[f - 1] if f > 0 else ZERO) | (FILE_BB[f + 1] if f < 7 else ZERO)
        for f in range(8)
    ],
    dtype=U64,
)


def _passed_masks(color: int) -> NDArray[np.uint64]:
    """Squares in front of a square on its own and adjacent files, from `color`'s side."""
    masks = np.zeros(64, dtype=U64)
    for square in range(64):
        file_index, rank_index = square % 8, square // 8
        span = FILE_BB[file_index] | ADJACENT_FILES[file_index]
        ahead = ZERO
        ranks = range(rank_index + 1, 8) if color == WHITE else range(0, rank_index)
        for rank in ranks:
            ahead |= U64(0xFF) << U64(rank * 8)
        masks[square] = span & ahead
    return masks


PASSED_MASK: NDArray[np.uint64] = np.stack([_passed_masks(WHITE), _passed_masks(BLACK)])

# Three files around the king, on the two ranks in front of it, for a cheap shelter term.
SHIELD_ZONE: NDArray[np.uint64] = np.zeros((2, 64), dtype=U64)
for _color in (WHITE, BLACK):
    for _sq in range(64):
        _file, _rank = _sq % 8, _sq // 8
        _files = FILE_BB[_file] | ADJACENT_FILES[_file]
        _zone = ZERO
        for _step in (1, 2):
            _target = _rank + _step if _color == WHITE else _rank - _step
            if 0 <= _target < 8:
                _zone |= U64(0xFF) << U64(_target * 8)
        SHIELD_ZONE[_color][_sq] = _files & _zone


@njit("int64(uint64[:, :], int64[:, :], int64)", cache=False)
def evaluate(bb: NDArray[np.uint64], meta: NDArray[np.int64], ply: int) -> int:
    """Static score in centipawns, positive for the side to move."""
    middlegame = 0
    endgame = 0
    phase = 0
    occupied = bb[ply, WHITE] | bb[ply, BLACK]

    for color in range(2):
        sign = 1 if color == WHITE else -1
        mine = bb[ply, color]
        own_pawns = bb[ply, 1 + PAWN] & mine
        enemy_pawns = bb[ply, 1 + PAWN] & bb[ply, 1 - color]

        for piece in range(PAWN, KING + 1):
            pieces = bb[ply, 1 + piece] & mine
            phase += PHASE_WEIGHT[piece] * popcount(pieces)
            while pieces != ZERO:
                square = lsb(pieces)
                pieces &= pieces - ONE
                index = square ^ 56 if color == WHITE else square
                middlegame += sign * (PIECE_MG[piece] + PST_MG[piece][index])
                endgame += sign * (PIECE_EG[piece] + PST_EG[piece][index])

                if piece == KNIGHT:
                    attacks = KNIGHT_ATTACKS[square]
                elif piece == BISHOP:
                    attacks = bishop_attacks(square, occupied)
                elif piece == ROOK:
                    attacks = rook_attacks(square, occupied)
                elif piece == QUEEN:
                    attacks = queen_attacks(square, occupied)
                else:
                    attacks = ZERO

                if piece >= KNIGHT and piece <= QUEEN:
                    moves = popcount(attacks & ~mine)
                    middlegame += sign * MOBILITY_MG[piece] * moves
                    endgame += sign * MOBILITY_EG[piece] * moves

                if piece == ROOK:
                    file_mask = FILE_BB[square & 7]
                    if (own_pawns | enemy_pawns) & file_mask == ZERO:
                        middlegame += sign * ROOK_OPEN_FILE
                    elif own_pawns & file_mask == ZERO:
                        middlegame += sign * ROOK_SEMI_OPEN_FILE

                elif piece == PAWN:
                    file_index = square & 7
                    if popcount(own_pawns & FILE_BB[file_index]) > 1:
                        middlegame += sign * DOUBLED_PAWN
                        endgame += sign * DOUBLED_PAWN
                    if own_pawns & ADJACENT_FILES[file_index] == ZERO:
                        middlegame += sign * ISOLATED_PAWN
                        endgame += sign * ISOLATED_PAWN
                    if enemy_pawns & PASSED_MASK[color][square] == ZERO:
                        rank = square >> 3
                        relative = rank if color == WHITE else 7 - rank
                        bonus = PASSED_BY_RANK[relative]
                        middlegame += sign * (bonus // 2)
                        endgame += sign * bonus

                elif piece == KING:
                    shelter = popcount(own_pawns & SHIELD_ZONE[color][square])
                    middlegame += sign * SHIELD_PAWN * shelter

        if popcount(bb[ply, 1 + BISHOP] & mine) >= 2:
            middlegame += sign * BISHOP_PAIR
            endgame += sign * BISHOP_PAIR

    if phase > TOTAL_PHASE:
        phase = TOTAL_PHASE
    score = (middlegame * phase + endgame * (TOTAL_PHASE - phase)) // TOTAL_PHASE
    if meta[ply, META_SIDE] == BLACK:
        score = -score
    return score + TEMPO
