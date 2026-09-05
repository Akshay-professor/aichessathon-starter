"""Texel tuning: fit the evaluation's weights to the results of games it played.

The evaluation is a sum of weighted features interpolated between a middlegame and an endgame
score, which makes it *linear in its own weights*. So tuning it is not a search problem, it is a
regression: extract the feature vector for each position, then fit the weights so the score,
squashed through a sigmoid, predicts how the game actually ended.

The feature extractor has to mirror bb_eval.evaluate exactly or the fit optimises the wrong
function, so `--verify` reconstructs the evaluation from the features and the current weights
and checks it against the real thing before any tuning happens.

    python -m tools.tune --verify
    python -m tools.tune --data data/quiet.txt --out weights/eval.npz
"""

import argparse
import time
from pathlib import Path

import numpy as np
from numba import njit
from numpy.typing import NDArray

import bb_eval
from bb_board import Position, lsb, popcount, set_fen
from bb_eval import (
    ADJACENT_FILES,
    BISHOP_PAIR,
    DOUBLED_PAWN,
    FILE_BB,
    ISOLATED_PAWN,
    PASSED_MASK,
    PHASE_WEIGHT,
    ROOK_OPEN_FILE,
    ROOK_SEMI_OPEN_FILE,
    SCALAR_COUNT,
    SHIELD_PAWN,
    SHIELD_ZONE,
    TOTAL_PHASE,
)
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
    WHITE,
    ZERO,
    bishop_attacks,
    queen_attacks,
    rook_attacks,
)

# Feature layout. Everything is from white's point of view: a white piece adds, a black one
# subtracts, so a symmetric position has an all-zero vector.
MATERIAL_BASE = 0          # 7 slots, indexed by piece type
PST_BASE = 7               # 7 * 64 slots
MOBILITY_BASE = PST_BASE + 7 * 64
SCALAR_BASE = MOBILITY_BASE + 7
PASSED_BASE = SCALAR_BASE + SCALAR_COUNT
FEATURE_COUNT = PASSED_BASE + 8


@njit("int64(uint64[:, :], int64, int16[:])", cache=False)
def extract(bb: NDArray[np.uint64], ply: int, out: NDArray[np.int16]) -> int:
    """Fill `out` with the feature counts for this position and return its phase."""
    for index in range(FEATURE_COUNT):
        out[index] = 0

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

                out[MATERIAL_BASE + piece] += sign
                out[PST_BASE + piece * 64 + index] += sign

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

                if KNIGHT <= piece <= QUEEN:
                    out[MOBILITY_BASE + piece] += sign * popcount(attacks & ~mine)

                if piece == ROOK:
                    file_mask = FILE_BB[square & 7]
                    if (own_pawns | enemy_pawns) & file_mask == ZERO:
                        out[SCALAR_BASE + ROOK_OPEN_FILE] += sign
                    elif own_pawns & file_mask == ZERO:
                        out[SCALAR_BASE + ROOK_SEMI_OPEN_FILE] += sign

                elif piece == PAWN:
                    file_index = square & 7
                    if popcount(own_pawns & FILE_BB[file_index]) > 1:
                        out[SCALAR_BASE + DOUBLED_PAWN] += sign
                    if own_pawns & ADJACENT_FILES[file_index] == ZERO:
                        out[SCALAR_BASE + ISOLATED_PAWN] += sign
                    if enemy_pawns & PASSED_MASK[color][square] == ZERO:
                        rank = square >> 3
                        relative = rank if color == WHITE else 7 - rank
                        out[PASSED_BASE + relative] += sign

                elif piece == KING:
                    out[SCALAR_BASE + SHIELD_PAWN] += sign * popcount(
                        own_pawns & SHIELD_ZONE[color][square]
                    )

        if popcount(bb[ply, 1 + BISHOP] & mine) >= 2:
            out[SCALAR_BASE + BISHOP_PAIR] += sign

    return phase if phase < TOTAL_PHASE else TOTAL_PHASE


def current_weights() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The evaluation's present numbers, laid out to match the feature vector."""
    middlegame = np.zeros(FEATURE_COUNT)
    endgame = np.zeros(FEATURE_COUNT)
    for piece in range(PAWN, KING + 1):
        middlegame[MATERIAL_BASE + piece] = bb_eval.PIECE_MG[piece]
        endgame[MATERIAL_BASE + piece] = bb_eval.PIECE_EG[piece]
        middlegame[PST_BASE + piece * 64 : PST_BASE + piece * 64 + 64] = bb_eval.PST_MG[piece]
        endgame[PST_BASE + piece * 64 : PST_BASE + piece * 64 + 64] = bb_eval.PST_EG[piece]
        middlegame[MOBILITY_BASE + piece] = bb_eval.MOBILITY_MG[piece]
        endgame[MOBILITY_BASE + piece] = bb_eval.MOBILITY_EG[piece]
    middlegame[SCALAR_BASE : SCALAR_BASE + SCALAR_COUNT] = bb_eval.SCALAR_MG
    endgame[SCALAR_BASE : SCALAR_BASE + SCALAR_COUNT] = bb_eval.SCALAR_EG
    middlegame[PASSED_BASE : PASSED_BASE + 8] = bb_eval.PASSED_MG
    endgame[PASSED_BASE : PASSED_BASE + 8] = bb_eval.PASSED_EG
    return middlegame, endgame


def save_weights(
    middlegame: NDArray[np.float64], endgame: NDArray[np.float64], out: Path
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    rounded_mg = np.rint(middlegame).astype(np.int32)
    rounded_eg = np.rint(endgame).astype(np.int32)
    piece_mg = np.zeros(7, dtype=np.int32)
    piece_eg = np.zeros(7, dtype=np.int32)
    pst_mg = np.zeros((7, 64), dtype=np.int32)
    pst_eg = np.zeros((7, 64), dtype=np.int32)
    mobility_mg = np.zeros(7, dtype=np.int32)
    mobility_eg = np.zeros(7, dtype=np.int32)
    for piece in range(PAWN, KING + 1):
        piece_mg[piece] = rounded_mg[MATERIAL_BASE + piece]
        piece_eg[piece] = rounded_eg[MATERIAL_BASE + piece]
        pst_mg[piece] = rounded_mg[PST_BASE + piece * 64 : PST_BASE + piece * 64 + 64]
        pst_eg[piece] = rounded_eg[PST_BASE + piece * 64 : PST_BASE + piece * 64 + 64]
        mobility_mg[piece] = rounded_mg[MOBILITY_BASE + piece]
        mobility_eg[piece] = rounded_eg[MOBILITY_BASE + piece]
    np.savez(
        out,
        piece_mg=piece_mg, piece_eg=piece_eg,
        pst_mg=pst_mg, pst_eg=pst_eg,
        mobility_mg=mobility_mg, mobility_eg=mobility_eg,
        scalar_mg=rounded_mg[SCALAR_BASE : SCALAR_BASE + SCALAR_COUNT],
        scalar_eg=rounded_eg[SCALAR_BASE : SCALAR_BASE + SCALAR_COUNT],
        passed_mg=rounded_mg[PASSED_BASE : PASSED_BASE + 8],
        passed_eg=rounded_eg[PASSED_BASE : PASSED_BASE + 8],
    )


def verify(samples: int = 400) -> bool:
    """Reconstruct the evaluation from the features and check it matches the real one.

    If this disagrees, the tuner is optimising a different function from the one that plays,
    and every result downstream of it is meaningless.
    """
    import random

    import chess

    middlegame, endgame = current_weights()
    position = Position()
    features = np.zeros(FEATURE_COUNT, dtype=np.int16)
    rng = random.Random(0)
    mismatches = 0

    for _ in range(samples):
        board = chess.Board()
        for _ in range(rng.randrange(0, 80)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        if board.is_game_over():
            continue

        set_fen(position, board.fen())
        phase = extract(position.bb, 0, features)
        counts = features.astype(np.float64)
        raw_mg = float(middlegame @ counts)
        raw_eg = float(endgame @ counts)
        white_score = (raw_mg * phase + raw_eg * (TOTAL_PHASE - phase)) // TOTAL_PHASE
        rebuilt = white_score if board.turn == chess.WHITE else -white_score
        rebuilt += bb_eval.TEMPO

        actual = int(bb_eval.evaluate(position.bb, position.meta, 0))
        if int(rebuilt) != actual:
            mismatches += 1
            if mismatches <= 3:
                print(f"mismatch {board.fen()}: rebuilt {int(rebuilt)} vs actual {actual}")

    print(f"{samples} positions, {mismatches} mismatches between features and the evaluation")
    return mismatches == 0


def load_dataset(path: Path, limit: int) -> tuple[NDArray[np.int16], NDArray[np.float32],
                                                  NDArray[np.float32], NDArray[np.float32]]:
    """Read 'fen result' lines into features, phase, side-to-move sign and result."""
    position = Position()

    # A million rows of 476 int16 is nearly a gigabyte, so it is filled in place rather than
    # accumulated in a list and copied.
    with path.open() as handle:
        total = min(sum(1 for _ in handle), limit)

    rows = np.zeros((total, FEATURE_COUNT), dtype=np.int16)
    phases = np.zeros(total, dtype=np.float32)
    stm = np.zeros(total, dtype=np.float32)
    results = np.zeros(total, dtype=np.float32)

    with path.open() as handle:
        for count, line in enumerate(handle):
            if count >= total:
                break
            fen, _, label = line.rpartition(" ")
            set_fen(position, fen)
            phases[count] = extract(position.bb, 0, rows[count]) / TOTAL_PHASE
            stm[count] = 1.0 if fen.split()[1] == "w" else -1.0
            results[count] = float(label)

    return rows, phases, stm, results


def _scores(
    features: NDArray[np.int16], phase: NDArray[np.float32], stm: NDArray[np.float32],
    middlegame: NDArray[np.float32], endgame: NDArray[np.float32], tempo: float,
    batch: int = 65536,
) -> NDArray[np.float32]:
    out = np.empty(len(features), dtype=np.float32)
    for start in range(0, len(features), batch):
        stop = min(start + batch, len(features))
        block = features[start:stop].astype(np.float32)
        raw_mg = block @ middlegame
        raw_eg = block @ endgame
        out[start:stop] = raw_mg * phase[start:stop] + raw_eg * (1.0 - phase[start:stop])
    return out + stm * tempo


def _sigmoid(scores: NDArray[np.float32], scale: float) -> NDArray[np.float32]:
    squashed = 1.0 / (1.0 + np.exp(-scale * scores / 400.0 * np.log(10.0)))
    return np.asarray(squashed, dtype=np.float32)


def _loss(
    features: NDArray[np.int16], phase: NDArray[np.float32], stm: NDArray[np.float32],
    results: NDArray[np.float32], middlegame: NDArray[np.float32],
    endgame: NDArray[np.float32], tempo: float, scale: float,
) -> float:
    predicted = _sigmoid(_scores(features, phase, stm, middlegame, endgame, tempo), scale)
    return float(np.mean((predicted - results) ** 2))


def calibrate(
    features: NDArray[np.int16], phase: NDArray[np.float32], stm: NDArray[np.float32],
    results: NDArray[np.float32], middlegame: NDArray[np.float32],
    endgame: NDArray[np.float32], tempo: float,
) -> float:
    """Find the sigmoid scale that best maps the current evaluation onto observed results."""
    best_scale, best_loss = 1.0, float("inf")
    for scale in np.arange(0.4, 2.01, 0.05):
        value = _loss(features, phase, stm, results, middlegame, endgame, tempo, float(scale))
        if value < best_loss:
            best_scale, best_loss = float(scale), value
    return best_scale


def tune(
    features: NDArray[np.int16], phase: NDArray[np.float32], stm: NDArray[np.float32],
    results: NDArray[np.float32], epochs: int, learning_rate: float, decay: float,
    batch: int = 65536,
) -> tuple[NDArray[np.float32], NDArray[np.float32], float]:
    """Adam on the squared error between the squashed score and the game result."""
    start_mg, start_eg = current_weights()
    middlegame = start_mg.astype(np.float32)
    endgame = start_eg.astype(np.float32)
    anchor_mg = middlegame.copy()
    anchor_eg = endgame.copy()
    tempo = float(bb_eval.TEMPO)

    split = int(len(features) * 0.9)
    train = slice(0, split)
    valid = slice(split, len(features))

    scale = calibrate(features[train], phase[train], stm[train], results[train],
                      middlegame, endgame, tempo)
    print(f"sigmoid scale K = {scale:.2f}")

    moment_mg = np.zeros_like(middlegame)
    moment_eg = np.zeros_like(endgame)
    velocity_mg = np.zeros_like(middlegame)
    velocity_eg = np.zeros_like(endgame)
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    constant = scale * np.log(10.0) / 400.0

    best_valid = _loss(features[valid], phase[valid], stm[valid], results[valid],
                       middlegame, endgame, tempo, scale)
    best = (middlegame.copy(), endgame.copy(), tempo)
    start_train = _loss(features[train], phase[train], stm[train], results[train],
                        middlegame, endgame, tempo, scale)
    print(f"epoch   0  train {start_train:.6f}  validation {best_valid:.6f}")

    count = split
    for epoch in range(1, epochs + 1):
        grad_mg = np.zeros_like(middlegame)
        grad_eg = np.zeros_like(endgame)
        grad_tempo = 0.0
        for start in range(0, count, batch):
            stop = min(start + batch, count)
            block = features[start:stop].astype(np.float32)
            block_phase = phase[start:stop]
            raw = block @ middlegame * block_phase + block @ endgame * (1.0 - block_phase)
            raw = raw + stm[start:stop] * tempo
            predicted = 1.0 / (1.0 + np.exp(-constant * raw))
            outer = 2.0 * (predicted - results[start:stop]) * predicted * (1.0 - predicted)
            outer = outer * constant
            grad_mg += block.T @ (outer * block_phase)
            grad_eg += block.T @ (outer * (1.0 - block_phase))
            grad_tempo += float(np.sum(outer * stm[start:stop]))
        grad_mg /= count
        grad_eg /= count
        grad_tempo /= count
        # Pull gently back toward the hand-set numbers: material and piece-square tables are
        # partly redundant, so the fit has flat directions it would otherwise wander along.
        grad_mg += decay * (middlegame - anchor_mg)
        grad_eg += decay * (endgame - anchor_eg)

        moment_mg = beta1 * moment_mg + (1 - beta1) * grad_mg
        moment_eg = beta1 * moment_eg + (1 - beta1) * grad_eg
        velocity_mg = beta2 * velocity_mg + (1 - beta2) * grad_mg**2
        velocity_eg = beta2 * velocity_eg + (1 - beta2) * grad_eg**2
        correction1 = 1 - beta1**epoch
        correction2 = 1 - beta2**epoch
        middlegame -= learning_rate * (moment_mg / correction1) / (
            np.sqrt(velocity_mg / correction2) + epsilon
        )
        endgame -= learning_rate * (moment_eg / correction1) / (
            np.sqrt(velocity_eg / correction2) + epsilon
        )
        tempo -= learning_rate * grad_tempo / (abs(grad_tempo) + epsilon) * 0.1

        if epoch % 25 == 0 or epoch == epochs:
            current = _loss(features[valid], phase[valid], stm[valid], results[valid],
                            middlegame, endgame, tempo, scale)
            fitted = _loss(features[train], phase[train], stm[train], results[train],
                           middlegame, endgame, tempo, scale)
            marker = ""
            if current < best_valid:
                best_valid = current
                best = (middlegame.copy(), endgame.copy(), tempo)
                marker = "  *"
            print(f"epoch {epoch:3d}  train {fitted:.6f}  validation {current:.6f}{marker}",
                  flush=True)

    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Texel-tune the evaluation weights.")
    parser.add_argument("--data", type=Path, default=Path("data/quiet.txt"))
    parser.add_argument("--out", type=Path, default=Path("weights/eval.npz"))
    parser.add_argument("--limit", type=int, default=2_000_000)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1.5)
    parser.add_argument("--decay", type=float, default=2e-5)
    parser.add_argument("--verify", action="store_true")
    arguments = parser.parse_args()

    if arguments.verify:
        raise SystemExit(0 if verify() else "features do not match the evaluation")

    if not verify(200):
        raise SystemExit("features do not match the evaluation; refusing to tune")

    started = time.perf_counter()
    features, phase, stm, results = load_dataset(arguments.data, arguments.limit)
    print(f"{len(features):,} positions loaded in {time.perf_counter() - started:.1f}s")

    middlegame, endgame, tempo = tune(
        features, phase, stm, results,
        arguments.epochs, arguments.learning_rate, arguments.decay,
    )
    save_weights(middlegame.astype(np.float64), endgame.astype(np.float64), arguments.out)
    print(f"wrote {arguments.out}  (tempo {tempo:.1f})")


if __name__ == "__main__":
    main()
