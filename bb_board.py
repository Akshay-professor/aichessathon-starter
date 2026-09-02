"""Position state, FEN handling and move application for the bitboard engine.

The whole game state lives in preallocated arrays indexed by search ply, so the search never
allocates. Moves are applied copy-make: a position is 8 bitboards, a 64 byte mailbox and five
scalars, which is cheap enough to copy and removes every unmake bug at a stroke. Unmake is the
single largest source of defects in a hand-written engine and we simply do not have one.

Array layout, all indexed `[ply]`:
    bb[ply, 0]        white occupancy          bb[ply, 1]  black occupancy
    bb[ply, 1 + type] one bitboard per piece type, PAWN=1 through KING=6
    mailbox[ply, sq]  piece type on that square, 0 for empty; colour comes from occupancy
    meta[ply]         side to move, castling rights, en passant square, halfmove clock
    zkey[ply]         zobrist hash
"""

import numpy as np
from numba import njit
from numpy.typing import NDArray

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
    ZOBRIST_CASTLE,
    ZOBRIST_EP,
    ZOBRIST_PIECE,
    ZOBRIST_SIDE,
    bishop_attacks,
    rook_attacks,
)

MAX_PLY = 160
MAX_MOVES = 256

META_SIDE = 0
META_CASTLE = 1
META_EP = 2
META_HALFMOVE = 3
META_FIELDS = 4

CASTLE_WK = 1
CASTLE_WQ = 2
CASTLE_BK = 4
CASTLE_BQ = 8

FLAG_EP = 1 << 15
FLAG_CASTLE = 1 << 16
FLAG_DOUBLE = 1 << 17

A1, E1, H1 = 0, 4, 7
A8, E8, H8 = 56, 60, 63

# Rights surviving a move that touches this square, applied to both from and to.
CASTLE_KEEP: NDArray[np.int64] = np.full(64, 15, dtype=np.int64)
CASTLE_KEEP[A1] = 15 & ~CASTLE_WQ
CASTLE_KEEP[H1] = 15 & ~CASTLE_WK
CASTLE_KEEP[E1] = 15 & ~(CASTLE_WK | CASTLE_WQ)
CASTLE_KEEP[A8] = 15 & ~CASTLE_BQ
CASTLE_KEEP[H8] = 15 & ~CASTLE_BK
CASTLE_KEEP[E8] = 15 & ~(CASTLE_BK | CASTLE_BQ)

DEBRUIJN = U64(0x03F79D71B4CB0A89)
DEBRUIJN_INDEX: NDArray[np.int64] = np.zeros(64, dtype=np.int64)
for _i in range(64):
    DEBRUIJN_INDEX[(DEBRUIJN << U64(_i)) >> U64(58)] = _i

PIECE_LETTERS = ".pnbrqk"


class Position:
    """The array bundle a search operates on. One instance per search tree."""

    def __init__(self) -> None:
        self.bb: NDArray[np.uint64] = np.zeros((MAX_PLY, 8), dtype=U64)
        self.mailbox: NDArray[np.int8] = np.zeros((MAX_PLY, 64), dtype=np.int8)
        self.meta: NDArray[np.int64] = np.zeros((MAX_PLY, META_FIELDS), dtype=np.int64)
        self.zkey: NDArray[np.uint64] = np.zeros(MAX_PLY, dtype=U64)


@njit("int64(uint64)", cache=False, inline="always")
def lsb(bits: np.uint64) -> int:
    """Index of the least significant set bit. Undefined for zero, never called with it."""
    isolated = bits & (~bits + ONE)
    return int(DEBRUIJN_INDEX[(isolated * DEBRUIJN) >> U64(58)])


@njit("int64(uint64)", cache=False, inline="always")
def popcount(bits: np.uint64) -> int:
    bits = bits - ((bits >> U64(1)) & U64(0x5555555555555555))
    bits = (bits & U64(0x3333333333333333)) + ((bits >> U64(2)) & U64(0x3333333333333333))
    bits = (bits + (bits >> U64(4))) & U64(0x0F0F0F0F0F0F0F0F)
    return int((bits * U64(0x0101010101010101)) >> U64(56))


@njit("uint64(uint64[:, :], int64, int64, int64, uint64)", cache=False)
def attackers_to(
    bb: NDArray[np.uint64],
    ply: int,
    square: int,
    by_color: int,
    occupied: np.uint64,
) -> np.uint64:
    """Every piece of `by_color` bearing on `square` for the given occupancy."""
    them = bb[ply, by_color]
    # A pawn attacks `square` from where a pawn of the opposite colour on `square` would strike.
    attacks = PAWN_ATTACKS[1 - by_color][square] & bb[ply, 1 + PAWN]
    attacks |= KNIGHT_ATTACKS[square] & bb[ply, 1 + KNIGHT]
    attacks |= KING_ATTACKS[square] & bb[ply, 1 + KING]
    attacks |= bishop_attacks(square, occupied) & (bb[ply, 1 + BISHOP] | bb[ply, 1 + QUEEN])
    attacks |= rook_attacks(square, occupied) & (bb[ply, 1 + ROOK] | bb[ply, 1 + QUEEN])
    return U64(attacks & them)


@njit("boolean(uint64[:, :], int64, int64, int64, uint64)", cache=False)
def is_attacked(
    bb: NDArray[np.uint64],
    ply: int,
    square: int,
    by_color: int,
    occupied: np.uint64,
) -> bool:
    return bool(attackers_to(bb, ply, square, by_color, occupied) != ZERO)


@njit("int64(uint64[:, :], int64, int64)", cache=False, inline="always")
def king_square(bb: NDArray[np.uint64], ply: int, color: int) -> int:
    return lsb(bb[ply, 1 + KING] & bb[ply, color])


@njit("boolean(uint64[:, :], int64, int64)", cache=False)
def in_check(bb: NDArray[np.uint64], ply: int, color: int) -> bool:
    occupied = bb[ply, WHITE] | bb[ply, BLACK]
    return is_attacked(bb, ply, king_square(bb, ply, color), 1 - color, occupied)


@njit("int32(int64, int64, int64, int64)", cache=False, inline="always")
def encode(origin: int, target: int, promotion: int, flags: int) -> int:
    return origin | (target << 6) | (promotion << 12) | flags


@njit("int64(int32)", cache=False, inline="always")
def move_from(move: int) -> int:
    return move & 63


@njit("int64(int32)", cache=False, inline="always")
def move_to(move: int) -> int:
    return (move >> 6) & 63


@njit("int64(int32)", cache=False, inline="always")
def move_promotion(move: int) -> int:
    return (move >> 12) & 7


@njit("void(uint64[:, :], int8[:, :], int64[:, :], uint64[:], int64, int32)", cache=False)
def make_move(
    bb: NDArray[np.uint64],
    mailbox: NDArray[np.int8],
    meta: NDArray[np.int64],
    zkey: NDArray[np.uint64],
    ply: int,
    move: int,
) -> None:
    """Apply `move` at `ply`, writing the resulting position at `ply + 1`."""
    for slot in range(8):
        bb[ply + 1, slot] = bb[ply, slot]
    for square in range(64):
        mailbox[ply + 1, square] = mailbox[ply, square]

    us = meta[ply, META_SIDE]
    them = 1 - us
    rights = meta[ply, META_CASTLE]
    ep_square = meta[ply, META_EP]
    key = zkey[ply]

    origin = move_from(move)
    target = move_to(move)
    promotion = move_promotion(move)
    moved = np.int64(mailbox[ply, origin])
    captured = np.int64(mailbox[ply, target])

    # Undo the old en passant and castling contributions before recomputing them.
    if ep_square >= 0:
        key ^= ZOBRIST_EP[ep_square & 7]
    key ^= ZOBRIST_CASTLE[rights]

    origin_bit = ONE << U64(origin)
    target_bit = ONE << U64(target)

    bb[ply + 1, us] ^= origin_bit
    bb[ply + 1, 1 + moved] ^= origin_bit
    mailbox[ply + 1, origin] = 0
    key ^= ZOBRIST_PIECE[us, moved, origin]

    if captured != 0:
        bb[ply + 1, them] ^= target_bit
        bb[ply + 1, 1 + captured] ^= target_bit
        key ^= ZOBRIST_PIECE[them, captured, target]

    if (move & FLAG_EP) != 0:
        victim = target - 8 if us == WHITE else target + 8
        victim_bit = ONE << U64(victim)
        bb[ply + 1, them] ^= victim_bit
        bb[ply + 1, 1 + PAWN] ^= victim_bit
        mailbox[ply + 1, victim] = 0
        key ^= ZOBRIST_PIECE[them, PAWN, victim]

    placed = promotion if promotion != 0 else moved
    bb[ply + 1, us] |= target_bit
    bb[ply + 1, 1 + placed] |= target_bit
    mailbox[ply + 1, target] = np.int8(placed)
    key ^= ZOBRIST_PIECE[us, placed, target]

    if (move & FLAG_CASTLE) != 0:
        if target > origin:
            rook_from = origin + 3
            rook_to = origin + 1
        else:
            rook_from = origin - 4
            rook_to = origin - 1
        rook_bits = (ONE << U64(rook_from)) | (ONE << U64(rook_to))
        bb[ply + 1, us] ^= rook_bits
        bb[ply + 1, 1 + ROOK] ^= rook_bits
        mailbox[ply + 1, rook_from] = 0
        mailbox[ply + 1, rook_to] = np.int8(ROOK)
        key ^= ZOBRIST_PIECE[us, ROOK, rook_from]
        key ^= ZOBRIST_PIECE[us, ROOK, rook_to]

    rights = rights & CASTLE_KEEP[origin] & CASTLE_KEEP[target]
    key ^= ZOBRIST_CASTLE[rights]

    if (move & FLAG_DOUBLE) != 0:
        new_ep = (origin + target) // 2
        meta[ply + 1, META_EP] = new_ep
        key ^= ZOBRIST_EP[new_ep & 7]
    else:
        meta[ply + 1, META_EP] = -1

    if moved == PAWN or captured != 0:
        meta[ply + 1, META_HALFMOVE] = 0
    else:
        meta[ply + 1, META_HALFMOVE] = meta[ply, META_HALFMOVE] + 1

    meta[ply + 1, META_SIDE] = them
    meta[ply + 1, META_CASTLE] = rights
    zkey[ply + 1] = key ^ ZOBRIST_SIDE


@njit("uint64(uint64[:, :], int64[:, :], int64)", cache=False)
def compute_zobrist(
    bb: NDArray[np.uint64], meta: NDArray[np.int64], ply: int
) -> np.uint64:
    """Full hash from scratch. Only used to prove the incremental update in make_move."""
    key = ZERO
    for color in range(2):
        for piece in range(PAWN, KING + 1):
            pieces = bb[ply, 1 + piece] & bb[ply, color]
            while pieces != ZERO:
                square = lsb(pieces)
                pieces &= pieces - ONE
                key ^= ZOBRIST_PIECE[color, piece, square]
    key ^= ZOBRIST_CASTLE[meta[ply, META_CASTLE]]
    if meta[ply, META_EP] >= 0:
        key ^= ZOBRIST_EP[meta[ply, META_EP] & 7]
    if meta[ply, META_SIDE] == BLACK:
        key ^= ZOBRIST_SIDE
    return U64(key)


def set_fen(position: Position, fen: str) -> None:
    """Load a FEN into ply 0. Pure Python: this runs once per move, never in the search."""
    board, side, castling, ep_field = fen.split()[:4]
    halfmove = int(fen.split()[4]) if len(fen.split()) > 4 else 0

    position.bb[0, :] = ZERO
    position.mailbox[0, :] = 0

    rank = 7
    file_index = 0
    for character in board:
        if character == "/":
            rank -= 1
            file_index = 0
        elif character.isdigit():
            file_index += int(character)
        else:
            piece = PIECE_LETTERS.index(character.lower())
            color = WHITE if character.isupper() else BLACK
            square = rank * 8 + file_index
            bit = ONE << U64(square)
            position.bb[0, color] |= bit
            position.bb[0, 1 + piece] |= bit
            position.mailbox[0, square] = piece
            file_index += 1

    rights = 0
    rights |= CASTLE_WK if "K" in castling else 0
    rights |= CASTLE_WQ if "Q" in castling else 0
    rights |= CASTLE_BK if "k" in castling else 0
    rights |= CASTLE_BQ if "q" in castling else 0

    position.meta[0, META_SIDE] = WHITE if side == "w" else BLACK
    position.meta[0, META_CASTLE] = rights
    position.meta[0, META_EP] = -1 if ep_field == "-" else _square_index(ep_field)
    position.meta[0, META_HALFMOVE] = halfmove
    position.zkey[0] = compute_zobrist(position.bb, position.meta, 0)


def _square_index(name: str) -> int:
    return (ord(name[0]) - ord("a")) + (int(name[1]) - 1) * 8


def square_name(square: int) -> str:
    return chr(ord("a") + (square % 8)) + str(square // 8 + 1)


def move_uci(move: int) -> str:
    """UCI text for an encoded move, including the promotion suffix."""
    text = square_name(move & 63) + square_name((move >> 6) & 63)
    promotion = (move >> 12) & 7
    if promotion:
        text += PIECE_LETTERS[promotion]
    return text
