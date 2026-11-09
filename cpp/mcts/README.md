# nash_mcts: C++ PUCT tree search

A pybind11 extension that runs AlphaZero-style batched MCTS for the Nash-Eval policy + value network.
C++ handles move generation (the vendored `cpp/third_party/chess-library/chess.hpp`), the `(8, 8, 36)`
input encoding, the tree, selection, virtual loss, backup and a light terminal solver. Python only
evaluates the network. The Python driver is `core/mcts.py`; the tests are in `tests/mcts_test.py`.

## Build

You need C++17, CMake ≥ 3.15 and pybind11 (pip). There are no other dependencies.

```
python cpp/mcts/build.py            # cmake + build with 2 jobs (override with NASH_MCTS_JOBS)
```

This produces `cpp/mcts/nash_mcts.cpython-*.so`. Intermediate files go in `cpp/mcts/build/`, and both
are git-ignored. `core/mcts.py` adds `cpp/mcts` to `sys.path` when the module isn't importable. The
tests skip when the extension hasn't been built.

## API

```python
import nash_mcts
nash_mcts.POLICY_SIZE            # 4672
nash_mcts.INPUT_SHAPE            # (8, 8, 36)
nash_mcts.encode(fen)            # (8, 8, 36) float32, identical to fast_fen_to_example(fen)
nash_mcts.encode_batch(fens)     # (k, 8, 8, 36)
nash_mcts.legal_moves(fen)       # [(uci, move_to_index(...)), ...]

tree = nash_mcts.MCTS(c_puct=2.0, fpu_reduction=0.25, dirichlet_alpha=0.3,
                      dirichlet_epsilon=0.25, virtual_loss=1, seed=0)
tree.reset(fen, history_uci=[], add_noise=True)  # root = fen after history_uci (history is used for repetition)
x = tree.gather(n)               # up to n selections -> (k, 8, 8, 36) distinct leaves; releases the GIL
tree.backup(policy_logits, wdl)  # (k, 4672) logits, (k, 3) W/D/L for the side to move at each leaf
tree.cancel()                    # drop a pending batch and its virtual loss
tree.root_stats()                # [(uci, policy_index, visits, Q, prior)], Q = root side's expected score
tree.root_move_proven()          # 'win' / 'draw' / 'loss' / 'unknown' per root move (root side's view)
tree.root_visits(); tree.root_value(); tree.root_proven(); tree.root_terminal(); tree.root_fen()
tree.select_move(temperature=0.0)  # argmax visits at T=0, otherwise samples N^(1/T)
tree.advance(uci)                # re-roots the tree on a child and keeps its subtree
tree.noise_enabled               # read/write property
tree.pending_paths(); tree.num_pending(); tree.total_virtual_loss(); tree.num_nodes(); tree.last_gather_stats()
```

`core/mcts.py` provides `run_mcts`, `select_mcts_move`, `search`, `MultiMCTS` (several trees, one
network call per round), `visit_distribution` (the policy training target), `play_game_mcts` and
`self_play_game_mcts`.

## Design

- **Values.** WDL index 0 means the side to move at the leaf wins. A leaf's value is `W + D/2` for that
  side. Each edge stores `W` from the view of the player who made the move, so the value flips (`1 - v`) at every ply.
- **Selection.** PUCT: `Q + c_puct * P * sqrt(N_parent) / (1 + N)`. Priors are a softmax of the policy
  logits over legal moves. Unvisited children use FPU `= (1 - parent Q) - fpu_reduction`.
- **Batching.** Each `gather` selection adds virtual loss along its path. A path that reaches an
  already-pending leaf counts as a collision. Its virtual loss stays until `backup` (or `cancel`) removes it,
  so later selections in the same batch spread out. A collision adds no visit and no value, so `root_visits`
  grows only by the number of leaves and terminals. A batch larger than the number of legal moves is fine.
- **Terminals and the solver.** Mate, stalemate, insufficient material (python-chess rules), the
  fifty-move rule (halfmove clock ≥ 100) and threefold repetition are terminal. Repetition counts
  positions in the tree path plus the `history_uci` positions inside the halfmove-clock window, and the
  key matches python-chess's `_transposition_key`. Terminal results propagate as proven win, draw or loss.
  A proven-lost child for the opponent is chosen at once, and proven-losing moves are avoided. At the
  root only mate and stalemate are terminal.
- **Noise.** Dirichlet noise is added to the root priors on first expansion and after `advance`, when
  `noise_enabled` is set.
- **Tree reuse.** `advance` copies the chosen subtree into a fresh contiguous node array, breadth-first.
  The old tree is freed.

## Encoding parity quirks (replicated from `fast_fen_to_example`)

- Board orientation: for White to move the tensor is the vertically flipped board, and for Black to
  move it is the board as-is. The hero is always the side to move.
- Bishop planes are split by tensor `(row + col)` parity, not by real square colour.
- Pawn pushes stop at the first occupied square, and a blocked single push also blocks the double push.
  Knight and king planes mark every target square. Slider rays include the first blocker of either colour.
- `encode(fen)` takes castling rights and the en-passant square literally from the FEN text, as the
  Python version does. Tree leaves are encoded from `board.fen()` semantics: cleaned castling rights,
  and an ep square only when an ep capture is legal.
- Material is summed in double precision in FEN order, divided by 40.2, then cast to float32.

Parity was checked on 15k random game positions (bit-exact) and on hand-picked edge cases (see `tests/mcts_test.py`).

## Benchmark

`python cpp/mcts/bench.py` uses a stub evaluator (no network) and 4000 simulations per search. It was run on the shared
4-core box under load (load average ~4–5), so treat the numbers as rough.

| position   | batch | sims/s total | sims/s C++ only | nodes   |
|------------|------:|-------------:|----------------:|--------:|
| start      | 1     | 58,134       | 91,455          | 89,510  |
| start      | 16    | 106,669      | 136,356         | 89,510  |
| start      | 64    | 117,764      | 149,095         | 89,510  |
| start      | 256   | 63,107       | 93,527          | 89,387  |
| middlegame | 1     | 41,251       | 62,338          | 144,986 |
| middlegame | 16    | 133,155      | 165,866         | 144,986 |
| middlegame | 64    | 104,557      | 130,759         | 171,092 |
| middlegame | 256   | 60,221       | 78,004          | 181,133 |
| endgame    | 1     | 68,944       | 128,327         | 65,378  |
| endgame    | 16    | 230,260      | 325,463         | 65,378  |
| endgame    | 64    | 210,320      | 299,804         | 65,369  |
| endgame    | 256   | 124,187      | 193,518         | 64,778  |

Encoding cost per position over 500 random positions:

| encoder | time per position | speedup |
|---|---|---|
| `fast_fen_to_example` | 239 µs | baseline |
| `nash_mcts.encode` | 2.9 µs | 81x |
| `encode_batch` | 4.0 µs | 60x |

With a real network the network call dominates; the tree costs about 7–15 µs per simulation.

## Limitations

- A tree isn't thread-safe. Use one tree per thread, or drive several trees with `MultiMCTS`.
- Repetition-based proven draws depend on the path (the usual graph-history interaction caveat).
- With tree reuse the root can already hold ≥ `num_simulations` visits, in which case no new search runs.
- Very large batches (256) lose throughput to collisions. Batches of 16–64 per tree work best.
- The search hasn't been tried with a real TensorFlow model yet. TF isn't installed on the dev box, so the
  Keras path is covered only by a stub with `predict`/`layers`.
