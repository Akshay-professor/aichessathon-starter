"""Tapered evaluation over the bitboard arrays.

Everything is int centipawns from the side to move's point of view, and every table is built at
import. The evaluation is deliberately conventional — material, piece-square tables, pawn
structure, mobility, king shelter — because its job right now is to be a good enough reference
to measure a tuned or learned replacement against. Phase 4 replaces the numbers, not the shape:
the weights live in arrays so a Texel run can overwrite them without touching the search.
"""

from pathlib import Path

import numpy as np
from numba import njit
from numpy.typing import NDArray

from bb_board import META_SIDE, lsb, popcount
from bb_tables import (
    BISHOP,
    BLACK,
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

PIECE_MG = np.array([0, 82, 337, 365, 477, 1025, 0], dtype=np.int32)
PIECE_EG = np.array([0, 94, 281, 297, 512, 936, 0], dtype=np.int32)
PHASE_WEIGHT = np.array([0, 0, 1, 1, 2, 4, 0], dtype=np.int32)
TOTAL_PHASE = 24

MOBILITY_MG = np.array([0, 0, 4, 5, 2, 1, 0], dtype=np.int32)
MOBILITY_EG = np.array([0, 0, 4, 5, 4, 2, 0], dtype=np.int32)

# Scalar terms, indexed by the constants below. Every one has a separate middlegame and
# endgame weight even where the two currently agree, so tuning can move them apart.
BISHOP_PAIR = 0
DOUBLED_PAWN = 1
ISOLATED_PAWN = 2
ROOK_OPEN_FILE = 3
ROOK_SEMI_OPEN_FILE = 4
SHIELD_PAWN = 5
KING_ZONE_ATTACKS = 6
KING_FILE_OPEN = 7
KING_ADJACENT_OPEN = 8
SCALAR_COUNT = 9

SCALAR_MG = np.array([30, -12, -14, 22, 10, 11, -3, -20, -8], dtype=np.int32)
SCALAR_EG = np.array([30, -12, -14, 0, 0, 0, -1, 0, 0], dtype=np.int32)

# King danger by the number of distinct enemy pieces bearing on the king zone. Bucketing the
# count keeps the response non-linear in attackers - two attackers are far worse than twice one
# - while the score stays linear in these weights, which is what makes it tunable.
KING_DANGER_MG = np.array([0, -2, -8, -20, -40, -65, -90, -110], dtype=np.int32)
KING_DANGER_EG = np.array([0, -1, -3, -7, -12, -18, -25, -30], dtype=np.int32)
KING_DANGER_BUCKETS = 8

TEMPO = 12

PASSED_MG = np.array([0, 3, 6, 12, 22, 38, 60, 0], dtype=np.int32)
PASSED_EG = np.array([0, 6, 12, 24, 44, 76, 120, 0], dtype=np.int32)

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

KING_ZONE: NDArray[np.uint64] = np.array(
    [KING_ATTACKS[sq] | (ONE << U64(sq)) for sq in range(64)], dtype=U64
)

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


def _load_tuned_weights() -> bool:
    """Overwrite the default numbers from weights/eval.npz, if it shipped with us.

    This has to run before the evaluation below is decorated: numba captures the arrays it
    reads as compile-time constants, so a load after that point would be silently ignored.
    """
    path = Path(__file__).resolve().parent / "weights" / "eval.npz"
    if not path.exists():
        return False
    # A weights file that does not match the current tables is ignored rather than trusted.
    # Anything raised here happens at import, and a failed import loses every game, so this
    # path is deliberately unable to fail.
    try:
        return _apply_weights(path)
    except Exception as error:
        print(f"ignoring {path.name}: {error!r}")
        return False


def _apply_weights(path: Path) -> bool:
    applied = False
    with np.load(path) as stored:
        for name, table in (
            ("piece_mg", PIECE_MG), ("piece_eg", PIECE_EG),
            ("pst_mg", PST_MG), ("pst_eg", PST_EG),
            ("mobility_mg", MOBILITY_MG), ("mobility_eg", MOBILITY_EG),
            ("scalar_mg", SCALAR_MG), ("scalar_eg", SCALAR_EG),
            ("passed_mg", PASSED_MG), ("passed_eg", PASSED_EG),
            ("king_danger_mg", KING_DANGER_MG), ("king_danger_eg", KING_DANGER_EG),
        ):
            if name not in stored:
                continue
            values = stored[name]
            if values.shape != table.shape:
                print(f"ignoring {name}: shape {values.shape}, expected {table.shape}")
                continue
            table[...] = values
            applied = True
    return applied


TUNED = _load_tuned_weights()


@njit("int64(uint64[:, :], int64[:, :], int64)", cache=False)
def evaluate(bb: NDArray[np.uint64], meta: NDArray[np.int64], ply: int) -> int:
    """Static score in centipawns, positive for the side to move."""
    middlegame = 0
    endgame = 0
    phase = 0
    occupied = bb[ply, WHITE] | bb[ply, BLACK]

    king_square_white = lsb(bb[ply, 1 + KING] & bb[ply, WHITE])
    king_square_black = lsb(bb[ply, 1 + KING] & bb[ply, BLACK])
    zone_white = KING_ZONE[king_square_white]
    zone_black = KING_ZONE[king_square_black]
    # How many distinct enemy pieces bear on each king, and how many squares of its zone
    # they cover between them. Accumulated during the piece loop, scored after it.
    attackers_white = 0
    attackers_black = 0
    zone_hits_white = 0
    zone_hits_black = 0

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
                elif piece == PAWN:
                    attacks = PAWN_ATTACKS[color][square]
                else:
                    attacks = ZERO

                if piece != KING:
                    if color == WHITE:
                        hits = popcount(attacks & zone_black)
                        if hits > 0:
                            attackers_black += 1
                            zone_hits_black += hits
                    else:
                        hits = popcount(attacks & zone_white)
                        if hits > 0:
                            attackers_white += 1
                            zone_hits_white += hits

                if piece >= KNIGHT and piece <= QUEEN:
                    moves = popcount(attacks & ~mine)
                    middlegame += sign * MOBILITY_MG[piece] * moves
                    endgame += sign * MOBILITY_EG[piece] * moves

                if piece == ROOK:
                    file_mask = FILE_BB[square & 7]
                    if (own_pawns | enemy_pawns) & file_mask == ZERO:
                        middlegame += sign * SCALAR_MG[ROOK_OPEN_FILE]
                        endgame += sign * SCALAR_EG[ROOK_OPEN_FILE]
                    elif own_pawns & file_mask == ZERO:
                        middlegame += sign * SCALAR_MG[ROOK_SEMI_OPEN_FILE]
                        endgame += sign * SCALAR_EG[ROOK_SEMI_OPEN_FILE]

                elif piece == PAWN:
                    file_index = square & 7
                    if popcount(own_pawns & FILE_BB[file_index]) > 1:
                        middlegame += sign * SCALAR_MG[DOUBLED_PAWN]
                        endgame += sign * SCALAR_EG[DOUBLED_PAWN]
                    if own_pawns & ADJACENT_FILES[file_index] == ZERO:
                        middlegame += sign * SCALAR_MG[ISOLATED_PAWN]
                        endgame += sign * SCALAR_EG[ISOLATED_PAWN]
                    if enemy_pawns & PASSED_MASK[color][square] == ZERO:
                        rank = square >> 3
                        relative = rank if color == WHITE else 7 - rank
                        middlegame += sign * PASSED_MG[relative]
                        endgame += sign * PASSED_EG[relative]

                elif piece == KING:
                    shelter = popcount(own_pawns & SHIELD_ZONE[color][square])
                    middlegame += sign * SCALAR_MG[SHIELD_PAWN] * shelter
                    endgame += sign * SCALAR_EG[SHIELD_PAWN] * shelter

        if popcount(bb[ply, 1 + BISHOP] & mine) >= 2:
            middlegame += sign * SCALAR_MG[BISHOP_PAIR]
            endgame += sign * SCALAR_EG[BISHOP_PAIR]

    # King danger, applied once per side now that every attacker has been counted.
    for color in range(2):
        sign = 1 if color == WHITE else -1
        own_pawns = bb[ply, 1 + PAWN] & bb[ply, color]
        if color == WHITE:
            count = attackers_white
            hits = zone_hits_white
            king_square = king_square_white
        else:
            count = attackers_black
            hits = zone_hits_black
            king_square = king_square_black
        if count >= KING_DANGER_BUCKETS:
            count = KING_DANGER_BUCKETS - 1
        middlegame += sign * KING_DANGER_MG[count]
        endgame += sign * KING_DANGER_EG[count]
        middlegame += sign * SCALAR_MG[KING_ZONE_ATTACKS] * hits
        endgame += sign * SCALAR_EG[KING_ZONE_ATTACKS] * hits

        king_file = king_square & 7
        if own_pawns & FILE_BB[king_file] == ZERO:
            middlegame += sign * SCALAR_MG[KING_FILE_OPEN]
            endgame += sign * SCALAR_EG[KING_FILE_OPEN]
        adjacent_open = 0
        if king_file > 0 and own_pawns & FILE_BB[king_file - 1] == ZERO:
            adjacent_open += 1
        if king_file < 7 and own_pawns & FILE_BB[king_file + 1] == ZERO:
            adjacent_open += 1
        middlegame += sign * SCALAR_MG[KING_ADJACENT_OPEN] * adjacent_open
        endgame += sign * SCALAR_EG[KING_ADJACENT_OPEN] * adjacent_open

    if phase > TOTAL_PHASE:
        phase = TOTAL_PHASE
    score = (middlegame * phase + endgame * (TOTAL_PHASE - phase)) // TOTAL_PHASE
    if meta[ply, META_SIDE] == BLACK:
        score = -score
    return score + TEMPO
