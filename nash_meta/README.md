# Nash meta-game training loop

Three processes run asynchronously around one run directory. None of them waits on another; they only see each
other's files.

```
            ┌──────────────── RUN/pool/gen_XXXX.keras ────────────────┐
            │                                                          ▼
   trainer.py (GPU)                                              arena.py (GPU + CPU workers)
   one training path, starting                                   actor games: 800-node MCTS vs 800-node MCTS,
   from the pretrained distill net                               both players drawn from the Nash σ (with
            ▲                                                    replacement), preset openings, root noise
            │                                                    league: every new checkpoint plays the pool
   RUN/labeled/shard_*.npz                                       (both colours, fixed openings) → MaxEnt Nash
            │                                                    → new σ
   labeler.py (CPU, Stockfish)  ◄──────── RUN/games/actor_*.jsonl ┘
   every position of every actor game → Stockfish plays it out to the end;
   every position of that subgame is labeled (SF move, subgame result)
```

* **arena.py** explores. Actor games sample both players from σ, the current Nash mixture over the pool, so
  exploration goes where the meta-game says the strong strategies are. When the trainer publishes a checkpoint,
  the arena plays it against every pool member with both colours on the evaluation openings (no noise; league
  games take priority over actor games), then recomputes the MaxEnt Nash (`core/nash.py`) in a separate process
  so play never stalls. The pool is capped at `--pool-size` (oldest evicted). Actor games pause while the
  labeler's backlog exceeds `--max-backlog`, so every generated position gets its subgame.
* **labeler.py** turns each actor-game position into a full Stockfish game played to its end, and labels every
  position of that subgame with Stockfish's move and the subgame's final result for the side to move.
* **trainer.py** trains continuously from the pretrained weights on a sliding window of the newest labeled
  positions and publishes a generation every `--checkpoint-positions` new positions. It pauses only when it has
  drawn more than `--replay` samples per ingested position.

## Pretraining → Nash

The 10-epoch Lichess distill run (`weights-distill-6x64-v2`) is the pretraining phase: its epoch checkpoints seed
the pool, the round robin between them (`self_play_agent/checkpoint_round_robin.py`, same evaluation openings)
seeds the league results and the first σ, and the trainer starts from its final weights.

```bash
RUN=out/meta
python nash_meta/arena.py   --run $RUN --init-models 'out/v2/epochs/epoch_*.keras' \
                            --init-tournament out/v2/tournament --openings data/evals.npz
python nash_meta/labeler.py --run $RUN --stockfish sf/stockfish-bin --engines 6 --nodes 10000
python nash_meta/trainer.py --run $RUN --init out/v2/distill_6x64.keras
```

Every process resumes from the run directory when restarted.

## Files

| Path | Writer | Contents |
|---|---|---|
| `pool/gen_XXXX.keras` | trainer | published generations (written to `tmp/` then renamed) |
| `league.json` | arena | pool, σ and its history (Nash, Elo per recomputation), pairwise results, throughput |
| `games/actor_*.jsonl` | arena | actor games: opening, moves, White's expected score per ply, result |
| `labeled/shard_*.npz` | labeler | `fens`, `idx` (SF move), `wdl` (one-hot result, side to move), `subgame`, `ply`, `roots` |
| `labeler.json` | labeler | backlog, subgames/s, positions/s, result counts |
| `trainer.json`, `trainer_last.keras` | trainer | steps, ingested positions, generations, train/val curves; resume checkpoint |
