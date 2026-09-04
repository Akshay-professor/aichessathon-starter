"""Generate labelled training positions by playing the engine against itself.

Texel tuning needs positions paired with the result of the game they came from. We have no
network and may not ship or run another engine, so the label source is our own play. The games
start from randomised openings because rated games start from curated positions we have never
seen, and tuning on the standard opening alone would fit the wrong distribution.

Only quiet positions are kept: nothing in check, nothing where the move played was a capture.
An evaluation tuned on positions that are mid-exchange learns the noise of the exchange.

    python -m tools.selfplay --games 2000 --out data/quiet.txt
"""

import argparse
import random
import time
from pathlib import Path

import chess

from bb_board import Position, set_fen
from bb_search import State
from harness.rules import PLY_CAP

PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
OPENING_PLIES = 8
SKIP_PLIES = 10
RESULTS = {"1-0": "1.0", "0-1": "0.0", "1/2-1/2": "0.5"}


def _material(board: chess.Board) -> int:
    return sum(
        value * (len(board.pieces(piece, chess.WHITE)) - len(board.pieces(piece, chess.BLACK)))
        for piece, value in PIECE_VALUES.items()
    )


def _adjudicate(board: chess.Board) -> str:
    """The referee's rule: 300 plies without a result is decided on material, else drawn."""
    balance = _material(board)
    if balance > 0:
        return "1-0"
    if balance < 0:
        return "0-1"
    return "1/2-1/2"


def _play(
    state: State, position: Position, depth: int, rng: random.Random
) -> tuple[str, list[str]]:
    board = chess.Board()
    for _ in range(OPENING_PLIES):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(rng.choice(moves))

    quiet: list[str] = []
    while True:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            return outcome.result(), quiet
        if len(board.move_stack) >= PLY_CAP:
            return _adjudicate(board), quiet

        set_fen(position, board.fen())
        state.bb[0] = position.bb[0]
        state.mailbox[0] = position.mailbox[0]
        state.meta[0] = position.meta[0]
        state.zkey[0] = position.zkey[0]
        state.game_keys[0] = position.zkey[0]
        state.control[:] = 0
        state.run(depth, 1)

        encoded = int(state.control[3])
        move = chess.Move(encoded & 63, (encoded >> 6) & 63, promotion=(encoded >> 12) & 7 or None)
        if move not in board.legal_moves:
            return _adjudicate(board), quiet

        is_capture = board.is_capture(move)
        if len(board.move_stack) >= SKIP_PLIES and not board.is_check() and not is_capture:
            quiet.append(board.fen())
        board.push(move)


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-play labelled positions for tuning.")
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("data/quiet.txt"))
    arguments = parser.parse_args()

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(arguments.seed)
    state = State()
    position = Position()

    started = time.perf_counter()
    written = 0
    with arguments.out.open("w") as handle:
        for game in range(arguments.games):
            result, quiet = _play(state, position, arguments.depth, rng)
            label = RESULTS[result]
            for fen in quiet:
                handle.write(f"{fen} {label}\n")
                written += 1
            if (game + 1) % 25 == 0:
                rate = (game + 1) / (time.perf_counter() - started)
                print(f"{game + 1}/{arguments.games} games, {written:,} positions, "
                      f"{rate:.1f} games/sec", flush=True)

    print(f"wrote {written:,} positions to {arguments.out}")


if __name__ == "__main__":
    main()
