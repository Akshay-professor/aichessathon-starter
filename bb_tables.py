"""Precomputed tables and sliding-attack kernels for the bitboard engine.

Everything here is built at import, inside the platform's 60 second init budget, and never
allocated again. Bitboards are `uint64`; squares are `int64` with a1 = 0 and h8 = 63.

Sliding attacks use hyperbola quintessence for files, diagonals and anti-diagonals, and a
first-rank lookup for ranks. That split is not an optimisation, it is a correctness boundary:
the byte-swap trick reverses rank order, which is exactly what a file or a diagonal needs and
exactly what a rank does not. Magic bitboards would be faster; this is a few lines, has no
constants to get wrong, and can be replaced later if profiling asks for it.
"""

import numpy as np
from numba import njit
from numpy.typing import NDArray

U64 = np.uint64
ONE = U64(1)
ZERO = U64(0)
FULL = U64(0xFFFFFFFFFFFFFFFF)

WHITE = 0
BLACK = 1

PAWN = 1
KNIGHT = 2
BISHOP = 3
ROOK = 4
QUEEN = 5
KING = 6

KNIGHT_DELTAS = ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2))
KING_DELTAS = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))
BISHOP_DIRS = ((1, 1), (1, -1), (-1, -1), (-1, 1))
ROOK_DIRS = ((0, 1), (1, 0), (0, -1), (-1, 0))


def _square(file_index: int, rank_index: int) -> int:
    return rank_index * 8 + file_index


def _on_board(file_index: int, rank_index: int) -> bool:
    return 0 <= file_index < 8 and 0 <= rank_index < 8


def _step_table(deltas: tuple[tuple[int, int], ...]) -> NDArray[np.uint64]:
    table = np.zeros(64, dtype=U64)
    for square in range(64):
        file_index, rank_index = square % 8, square // 8
        bits = ZERO
        for delta_file, delta_rank in deltas:
            target_file, target_rank = file_index + delta_file, rank_index + delta_rank
            if _on_board(target_file, target_rank):
                bits |= ONE << U64(_square(target_file, target_rank))
        table[square] = bits
    return table


def _ray(square: int, direction: tuple[int, int], include_edge: bool) -> np.uint64:
    """Every square along `direction` from `square`, optionally dropping the final square."""
    file_index, rank_index = square % 8, square // 8
    delta_file, delta_rank = direction
    collected: list[int] = []
    while True:
        file_index += delta_file
        rank_index += delta_rank
        if not _on_board(file_index, rank_index):
            break
        collected.append(_square(file_index, rank_index))
    if not include_edge and collected:
        collected.pop()
    bits = ZERO
    for target in collected:
        bits |= ONE << U64(target)
    return bits


def _line_mask(square: int, directions: tuple[tuple[int, int], ...]) -> np.uint64:
    bits = ZERO
    for direction in directions:
        bits |= _ray(square, direction, include_edge=True)
    return bits


KNIGHT_ATTACKS: NDArray[np.uint64] = _step_table(KNIGHT_DELTAS)
KING_ATTACKS: NDArray[np.uint64] = _step_table(KING_DELTAS)

PAWN_ATTACKS: NDArray[np.uint64] = np.zeros((2, 64), dtype=U64)
for _sq in range(64):
    PAWN_ATTACKS[WHITE][_sq] = _step_table(((-1, 1), (1, 1)))[_sq]
    PAWN_ATTACKS[BLACK][_sq] = _step_table(((-1, -1), (1, -1)))[_sq]

# Full lines through a square, excluding the square itself. Rank masks are unused by the
# hyperbola path but kept for evaluation and pin logic.
FILE_LINE: NDArray[np.uint64] = np.array(
    [_line_mask(sq, ((0, 1), (0, -1))) for sq in range(64)], dtype=U64
)
RANK_LINE: NDArray[np.uint64] = np.array(
    [_line_mask(sq, ((1, 0), (-1, 0))) for sq in range(64)], dtype=U64
)
DIAG_LINE: NDArray[np.uint64] = np.array(
    [_line_mask(sq, ((1, 1), (-1, -1))) for sq in range(64)], dtype=U64
)
ANTI_LINE: NDArray[np.uint64] = np.array(
    [_line_mask(sq, ((1, -1), (-1, 1))) for sq in range(64)], dtype=U64
)


def _first_rank_attacks() -> NDArray[np.uint64]:
    """attacks[file][inner occupancy] along a single rank, as bits 0..7."""
    table = np.zeros((8, 64), dtype=U64)
    for file_index in range(8):
        for inner in range(64):
            occupancy = inner << 1
            bits = 0
            for step in (1, -1):
                cursor = file_index + step
                while 0 <= cursor < 8:
                    bits |= 1 << cursor
                    if occupancy & (1 << cursor):
                        break
                    cursor += step
            table[file_index][inner] = U64(bits)
    return table


FIRST_RANK_ATTACKS: NDArray[np.uint64] = _first_rank_attacks()

# BETWEEN[a][b] is the open path between two aligned squares, LINE[a][b] the whole line through
# them. Both are zero when the squares do not share a line. Used for check evasion and pins.
BETWEEN: NDArray[np.uint64] = np.zeros((64, 64), dtype=U64)
LINE: NDArray[np.uint64] = np.zeros((64, 64), dtype=U64)
for _a in range(64):
    for _dir in BISHOP_DIRS + ROOK_DIRS:
        _path = ZERO
        _file, _rank = _a % 8, _a // 8
        while True:
            _file += _dir[0]
            _rank += _dir[1]
            if not _on_board(_file, _rank):
                break
            _b = _square(_file, _rank)
            BETWEEN[_a][_b] = _path
            _path |= ONE << U64(_b)
    for _b in range(64):
        if _a == _b:
            continue
        for _mask in (FILE_LINE, RANK_LINE, DIAG_LINE, ANTI_LINE):
            if _mask[_a] & (ONE << U64(_b)):
                LINE[_a][_b] = (_mask[_a] | (ONE << U64(_a))) & (_mask[_b] | (ONE << U64(_b)))
                break

_RNG = np.random.default_rng(0x5EED_C0DE)
ZOBRIST_PIECE: NDArray[np.uint64] = _RNG.integers(0, 1 << 63, size=(2, 7, 64), dtype=np.uint64)
ZOBRIST_CASTLE: NDArray[np.uint64] = _RNG.integers(0, 1 << 63, size=16, dtype=np.uint64)
ZOBRIST_EP: NDArray[np.uint64] = _RNG.integers(0, 1 << 63, size=8, dtype=np.uint64)
ZOBRIST_SIDE = U64(_RNG.integers(0, 1 << 63, dtype=np.uint64))


@njit("uint64(uint64)", cache=False, inline="always")
def byteswap(bits: np.uint64) -> np.uint64:
    """Reverse byte order, which reverses rank order. The heart of hyperbola quintessence."""
    bits = ((bits >> U64(8)) & U64(0x00FF00FF00FF00FF)) | (
        (bits & U64(0x00FF00FF00FF00FF)) << U64(8)
    )
    bits = ((bits >> U64(16)) & U64(0x0000FFFF0000FFFF)) | (
        (bits & U64(0x0000FFFF0000FFFF)) << U64(16)
    )
    return (bits >> U64(32)) | (bits << U64(32))


@njit("uint64(int64, uint64, uint64)", cache=False, inline="always")
def line_attacks(square: int, occupied: np.uint64, mask: np.uint64) -> np.uint64:
    """o ^ (o - 2r) along a line, mirrored to cover the negative direction."""
    piece = ONE << U64(square)
    blockers = occupied & mask
    forward = blockers - (piece + piece)
    reverse = byteswap(byteswap(blockers) - (byteswap(piece) + byteswap(piece)))
    return (forward ^ reverse) & mask


@njit("uint64(int64, uint64)", cache=False, inline="always")
def rank_attacks(square: int, occupied: np.uint64) -> np.uint64:
    """Ranks cannot use the byte-swap trick, so they get a direct lookup instead."""
    rank_base = U64(square & ~7)  # 8 * (square // 8)
    inner = (occupied >> (rank_base + ONE)) & U64(63)
    return U64(FIRST_RANK_ATTACKS[square & 7][inner] << rank_base)


@njit("uint64(int64, uint64)", cache=False, inline="always")
def bishop_attacks(square: int, occupied: np.uint64) -> np.uint64:
    return line_attacks(square, occupied, DIAG_LINE[square]) | line_attacks(
        square, occupied, ANTI_LINE[square]
    )


@njit("uint64(int64, uint64)", cache=False, inline="always")
def rook_attacks(square: int, occupied: np.uint64) -> np.uint64:
    return line_attacks(square, occupied, FILE_LINE[square]) | rank_attacks(square, occupied)


@njit("uint64(int64, uint64)", cache=False, inline="always")
def queen_attacks(square: int, occupied: np.uint64) -> np.uint64:
    return bishop_attacks(square, occupied) | rook_attacks(square, occupied)
