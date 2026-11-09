# v3 vs Leela's distilled 11258-96x8-se, both in our C++ MCTS

Leela net: Dietrich Kappe's distilled `11258-96x8-se` (8 blocks x 96 filters with SE, the same trunk as v3; its dense
policy head adds ~3.8M parameters), from https://github.com/dkappe/leela-chess-weights/releases/tag/11258-96x8-se.
Converted with `lc0 leela2onnx` (lc0 v0.32.1, GPL-3.0) and run through `play/leela.py`, which encodes Leela's
112-plane input with history and maps its 1858-move policy onto our 4672 indices (lc0's PolicyTemperature 1.359).
`play/leela_check.py` checks the adapter against lc0 itself: on 306 positions (real games plus castling, promotion
and en passant cases) the priors match to 0.0007 and the value exactly.

Same setup as games/sf_match_v3: 10 balanced openings x both colours (`pick_openings(evals.npz, 10, seed=7)`), SF 17.1
with 1 thread, Elo-limited SF at 1 s/move, our side limited to 800 new simulations per move. The openings come from the
Lichess eval positions v3 was trained on, which can only help v3 on its first move or two.

| Match | Leela 96x8 | v3 8x96 |
|---|---|---|
| vs SF Elo 2000 | 20/20 (20W) | 19.5/20 |
| vs SF Elo 2400 | 15.5/20 (12W 7D 1L) | 13.5/20 |
| vs SF Elo 2700 | 11/20 (7W 8D 5L) | 16/20 |
| Elo (MLE over the three) | ~2700 | ~2755 |
| vs full-strength SF @ 10k nodes | 1.5/20 (0W 3D 17L) | 2.5/20 |

Head-to-head (v3's score): 800 nodes each 13/20 (8W 10D 2L); 1 node each (pure policy) 10/20 and 54/100 (33W 42D 25L).
Our search parameters were tuned for v3, not for Leela's net.
