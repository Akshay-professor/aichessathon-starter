# How this agent works

One function, `get_move(fen, time_left_ms) -> str`. Everything below it is a bitboard chess
engine compiled by numba, written for a single core and a clock that averages about 1.3 seconds
a move.

## Why it is shaped this way

Strength at a fixed time control is roughly search depth times evaluation quality, and depth is
logarithmic in node rate. The obvious way to write this agent is a negamax over `python-chess`,
which is what the starter template hands you. Measured on this machine, that costs 39µs per
legal-move generation and 25µs per evaluation, so about 15,000 nodes a second, which buys depth
4 to 6 in the time available.

Putting the whole search into numba `nopython` mode instead — move generation, make-move,
evaluation, quiescence and the transposition table, with no Python objects on the hot path —
measures at 0.8 to 1.4 million nodes a second and reaches depth 12 to 18. No evaluation
improvement recovers six plies, so the node rate is where the effort went.

## The modules

| File | What it holds |
|---|---|
| `agent.py` | the entrypoint: time management, game history, pondering, and the fallbacks |
| `bb_tables.py` | attack tables, zobrist keys, sliding-attack kernels |
| `bb_board.py` | position arrays, FEN, incremental zobrist, copy-make |
| `bb_movegen.py` | pseudo-legal generation and the legality filter |
| `bb_eval.py` | tapered evaluation |
| `bb_search.py` | iterative deepening, PVS, quiescence, transposition table, pruning |

Three structural decisions worth explaining:

**Copy-make, not make/unmake.** A position is eight bitboards, a 64-byte mailbox and four
scalars. Copying that is nearly free, and it means there is no unmake path — the single largest
source of defects in a hand-written engine simply does not exist here.

**Hyperbola quintessence for sliding attacks, with a lookup table for ranks.** The byte-swap
trick reverses rank order, which is exactly what a file or a diagonal needs and exactly what a
rank cannot use. That split is a correctness boundary, not an optimisation. Magic bitboards
would be faster; this has no magic constants to get wrong and is fast enough.

**Search state is passed as explicit array parameters.** numba captures a module-level array as
a *readonly* constant and refuses to write to it, so every buffer the search mutates has to be
an argument. The signatures are long as a result.

## Time

The clock is enforced from outside the jitted code. `bb_search.search` compiles with
`nogil=True`, so an ordinary Python timer thread keeps running alongside it and raises a stop
flag that the search polls every 2048 nodes. One timer stops new iterations at 45% of the
budget, another aborts outright at the budget itself.

The budget comes from the 300-ply adjudication cap rather than a guess at game length. A game
that runs to the cap costs 150 moves out of 120s plus 0.5s a move, so the honest average is
about 1.3 seconds, not the 3 seconds that dividing the base clock by 40 suggests.

## Three things the position alone does not say

**Repetition.** The referee claims threefold and fifty-move draws over the whole game, but we
are only shown our own turns. The positions in between are reconstructed by replaying the one
legal move that leads from our last recorded position to the one we are handed.

**Adjudication.** Three hundred plies without a result is decided on material alone. Within 60
plies of the cap a draw stops being worth zero: ahead on material it is worth -150, because a
repetition throws away a win the referee would hand us; behind, it is worth +150, because a
repetition rescues half a point from a loss.

**The opponent's clock.** The process keeps its core after `get_move` returns, and pondering is
allowed. The ponder search runs on the position we just moved into, with the opponent to move,
filling a shared transposition table for every reply they might choose rather than betting on
one prediction. Measured at 36% of our own clock returned at equal depth.

## Never losing to ourselves

Illegal moves, crashes and flag falls each lose the game outright, so `get_move` cannot raise
and cannot return an illegal move. It validates the engine's choice and falls back, in order, to
a `python-chess` reference search and then to any legal move. This has already earned its place:
during development a timeout unwinding through the search left moves pushed on the board, and
the validation caught it on the first tactical test.

## How it is verified

- `python -m tools.perft` — exact node counts on seven standard positions. Clean to depth 5,
  484 million nodes.
- `python -m tools.fuzz` — every legal move set, board state and incremental zobrist key checked
  against `python-chess` at every node of thousands of random games.
- `python -m tools.import_time` — cold import against the 60s init budget, ceiling 40s. Every
  game pays the numba compile in full: the filesystem is read-only, and `/tmp` starts empty for
  each game, so no compile cache can survive between them.

A move generator that is right in 99.9% of positions still forfeits games, so nothing downstream
of perft and the fuzzer was built until both were clean.

## Things that were tried and rejected

**Static exchange evaluation.** Implemented and verified against a brute-force reference over
125,468 checks with no mismatches, then measured in the search and reverted. Nodes and time to
reach a fixed depth across five positions:

| Variant | Nodes | Time |
|---|---|---|
| no SEE | 5,491,488 | 6.34s |
| ordering, losing captures last | 6,868,097 | 8.19s |
| ordering, losing captures below killers | 5,877,790 | 7.02s |
| quiescence pruning only | 6,547,092 | 7.36s |
| lazy, only when the attacker outvalues the victim | 5,824,758 | 6.71s |

Every integration was worse. The likely reason is that this engine's nodes are unusually cheap —
the evaluation costs 608ns — so a swap-off that recomputes sliding attacks costs a meaningful
fraction of a whole node, where in an engine with an expensive evaluation it would be noise.
That predicts SEE becomes worth revisiting once the evaluation gets heavier, which is the one
condition under which it should be tried again.
