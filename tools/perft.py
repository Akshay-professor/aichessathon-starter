"""Perft: count leaf nodes and compare against published values.

A move generator that is right in 99.9% of positions still loses games, because one illegal
move is a forfeit. Perft is the only cheap way to know the generator is exact rather than
plausible, so nothing downstream of it is built until this is clean.

    python -m tools.perft            # the standard suite
    python -m tools.perft --divide "<fen>" --depth 3
"""

import argparse
import time

import numpy as np
from numba import njit
from numpy.typing import NDArray

from bb_board import MAX_MOVES, MAX_PLY, Position, make_move, move_uci, set_fen
from bb_movegen import generate_legal

# fen, then the expected node count at depth 1, 2, 3, ...
SUITE: list[tuple[str, str, list[int]]] = [
    (
        "startpos",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        [20, 400, 8902, 197281, 4865609, 119060324],
    ),
    (
        "kiwipete",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        [48, 2039, 97862, 4085603, 193690690],
    ),
    (
        "endgame",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        [14, 191, 2812, 43238, 674624, 11030083],
    ),
    (
        "promotion",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        [6, 264, 9467, 422333, 15833292],
    ),
    (
        "mirrored",
        "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1",
        [6, 264, 9467, 422333, 15833292],
    ),
    (
        "talkchess",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 0 1",
        [44, 1486, 62379, 2103487, 89941194],
    ),
    (
        "steven",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 1",
        [46, 2079, 89890, 3894594, 164075551],
    ),
]


@njit(
    "int64(uint64[:, :], int8[:, :], int64[:, :], uint64[:], int32[:, :], int64, int64)",
    cache=False,
)
def perft(
    bb: NDArray[np.uint64],
    mailbox: NDArray[np.int8],
    meta: NDArray[np.int64],
    zkey: NDArray[np.uint64],
    buffers: NDArray[np.int32],
    ply: int,
    depth: int,
) -> int:
    if depth == 0:
        return 1
    count = generate_legal(bb, mailbox, meta, zkey, ply, buffers[ply])
    if depth == 1:
        return count
    total = 0
    for index in range(count):
        move = buffers[ply, index]
        make_move(bb, mailbox, meta, zkey, ply, move)
        total += perft(bb, mailbox, meta, zkey, buffers, ply + 1, depth - 1)
    return total


def _fresh() -> tuple[Position, NDArray[np.int32]]:
    return Position(), np.zeros((MAX_PLY, MAX_MOVES), dtype=np.int32)


def run_suite(max_depth: int) -> bool:
    position, buffers = _fresh()
    ok = True
    for name, fen, expected in SUITE:
        for depth, want in enumerate(expected[:max_depth], start=1):
            set_fen(position, fen)
            started = time.perf_counter()
            got = int(
                perft(position.bb, position.mailbox, position.meta, position.zkey,
                      buffers, 0, depth)
            )
            elapsed = time.perf_counter() - started
            status = "ok " if got == want else "BAD"
            rate = got / elapsed / 1e6 if elapsed > 0 else 0.0
            print(
                f"{status} {name:10s} depth {depth}  {got:>12,}"
                f"  expected {want:>12,}  {elapsed:7.2f}s  {rate:6.2f} Mnps"
            )
            if got != want:
                ok = False
                break
    return ok


def divide(fen: str, depth: int) -> None:
    position, buffers = _fresh()
    set_fen(position, fen)
    count = generate_legal(
        position.bb, position.mailbox, position.meta, position.zkey, 0, buffers[0]
    )
    roots = [int(buffers[0, index]) for index in range(count)]
    total = 0
    for move in sorted(roots, key=move_uci):
        set_fen(position, fen)
        make_move(position.bb, position.mailbox, position.meta, position.zkey, 0, move)
        nodes = int(
            perft(position.bb, position.mailbox, position.meta, position.zkey,
                  buffers, 1, depth - 1)
        )
        total += nodes
        print(f"{move_uci(move)}: {nodes}")
    print(f"total: {total}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Perft the bitboard move generator.")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--divide")
    arguments = parser.parse_args()
    if arguments.divide:
        divide(arguments.divide, arguments.depth)
        return
    raise SystemExit(0 if run_suite(arguments.depth) else "perft mismatch")


if __name__ == "__main__":
    main()
