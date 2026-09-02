"""Differential fuzzing of the bitboard engine against python-chess.

Perft proves the generator on a handful of famously awkward positions. This proves it on
positions nobody chose: random games, every node checked. Three invariants at every position:

* the legal move set matches python-chess exactly
* the incremental zobrist key in make_move matches a hash recomputed from scratch
* the board arrays still describe the same position python-chess has

    python -m tools.fuzz --games 2000
"""

import argparse
import random

import chess
import numpy as np

from bb_board import (
    MAX_MOVES,
    Position,
    compute_zobrist,
    make_move,
    move_uci,
    set_fen,
)
from bb_movegen import generate_legal


def _describe(position: Position, ply: int) -> tuple[int, ...]:
    """Occupancy fingerprint, so a divergence in the arrays is caught even without a move."""
    return tuple(int(position.bb[ply, slot]) for slot in range(8))


def _reference_fingerprint(board: chess.Board) -> tuple[int, ...]:
    return (
        int(board.occupied_co[chess.WHITE]),
        int(board.occupied_co[chess.BLACK]),
        int(board.pawns),
        int(board.knights),
        int(board.bishops),
        int(board.rooks),
        int(board.queens),
        int(board.kings),
    )


def fuzz(games: int, seed: int, max_plies: int) -> int:
    rng = random.Random(seed)
    position = Position()
    moves = np.zeros(MAX_MOVES, dtype=np.int32)
    failures = 0
    positions = 0

    for game in range(games):
        board = chess.Board()
        set_fen(position, board.fen())

        for _ply in range(max_plies):
            if board.is_game_over(claim_draw=False):
                break
            positions += 1

            count = int(
                generate_legal(
                    position.bb, position.mailbox, position.meta, position.zkey, 0, moves
                )
            )
            mine = sorted(move_uci(int(moves[index])) for index in range(count))
            reference = sorted(move.uci() for move in board.legal_moves)

            if mine != reference:
                failures += 1
                print(f"game {game}: move set diverged at {board.fen()}")
                print(f"  extra:   {sorted(set(mine) - set(reference))}")
                print(f"  missing: {sorted(set(reference) - set(mine))}")
                break

            if _describe(position, 0) != _reference_fingerprint(board):
                failures += 1
                print(f"game {game}: board arrays diverged at {board.fen()}")
                break

            recomputed = int(compute_zobrist(position.bb, position.meta, 0))
            if recomputed != int(position.zkey[0]):
                failures += 1
                print(f"game {game}: zobrist diverged at {board.fen()}")
                break

            chosen = int(moves[rng.randrange(count)])
            make_move(position.bb, position.mailbox, position.meta, position.zkey,
                      0, chosen)
            board.push(chess.Move.from_uci(move_uci(chosen)))

            # Copy the freshly made position back down to ply 0 for the next iteration.
            position.bb[0, :] = position.bb[1, :]
            position.mailbox[0, :] = position.mailbox[1, :]
            position.meta[0, :] = position.meta[1, :]
            position.zkey[0] = position.zkey[1]

        if (game + 1) % 250 == 0:
            print(f"{game + 1} games, {positions:,} positions, {failures} divergences")

    print(f"\n{games} games, {positions:,} positions checked, {failures} divergences")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Differential test against python-chess.")
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-plies", type=int, default=300)
    arguments = parser.parse_args()
    failures = fuzz(arguments.games, arguments.seed, arguments.max_plies)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
