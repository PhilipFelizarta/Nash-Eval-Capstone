"""
Round robin between policy/value checkpoints, each playing with the C++ MCTS (nash_mcts), then the
MaxEnt Nash equilibrium of the meta-game (core/nash.py), as in the original tournament pipeline.

	python self_play_agent/checkpoint_round_robin.py --models 'out/epochs/epoch_*.keras' \
		--openings data/evals.npz --out games/tournament_distill_v2

Every pair plays each opening twice (colours swapped). Many games run at once: every search step
gathers a few leaves from each game's tree and evaluates them in one batch per model, so the GPU sees
large batches even though each tree only gets `--sims` simulations per move.

Writes `winrate_matrix.npz` in the project's format (winrate_matrix[a, b] = a's score with White
against b), all games as PGN, and `progress.json` (live; read by play/server.py --tournament-progress).
"""
import argparse
import glob
import json
import os
import sys
import time

import chess
import chess.pgn
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "cpp", "mcts"))

import nash_mcts  # noqa: E402
from core.nash import compute_nash_equilibrium, logit_transform, symmetrize_winrate_matrix  # noqa: E402


def load_models(paths):
	import tensorflow as tf
	for gpu in tf.config.list_physical_devices("GPU"):
		tf.config.experimental.set_memory_growth(gpu, True)
	nets = []
	for path in paths:
		model = tf.keras.models.load_model(path, compile=False)

		@tf.function(input_signature=[tf.TensorSpec([None, 8, 8, 36], tf.float32)])
		def call(x, model=model):
			out = model(x, training=False)
			return tf.cast(out["policy"], tf.float32), tf.cast(out["value"], tf.float32)
		call(tf.zeros((1, 8, 8, 36)))
		nets.append(lambda x, call=call: [t.numpy() for t in call(tf.constant(x))])
	return nets


def pick_openings(path, n, seed=0):
	"""Balanced opening positions (30+ pieces, Stockfish expected score 0.42-0.58) from the eval data."""
	data = np.load(path)
	fens, wdl = data["fens"], data["wdl"]
	rng = np.random.default_rng(seed)
	out = []
	for i in rng.permutation(len(fens)):
		fen = fens[i].decode()
		score = wdl[i, 0] + 0.5 * wdl[i, 1]
		if sum(c.isalpha() for c in fen.split()[0]) >= 30 and 0.42 <= score <= 0.58 and fen not in out:
			out.append(fen)
			if len(out) == n:
				return out
	return out


class Game:
	def __init__(self, gid, white, black, opening, args):
		self.id, self.players, self.opening = gid, (white, black), opening
		self.board = chess.Board(opening)
		self.trees = []
		for _ in range(2):
			t = nash_mcts.MCTS(c_puct=args.c_puct, dirichlet_epsilon=0.0)
			t.reset(opening, [], False)
			self.trees.append(t)
		self.args = args
		self.result = None
		self.values = []  # root value (side to move) per ply
		self._start_move()

	def side(self):  # 0 = White's player, 1 = Black's
		return 0 if self.board.turn == chess.WHITE else 1

	def tree(self):
		return self.trees[self.side()]

	def player(self):
		return self.players[self.side()]

	def _start_move(self):
		self.idle = 0
		self.target = self.tree().root_visits() + self.args.sims

	def search_done(self):
		t = self.tree()
		return t.root_visits() >= self.target or t.root_proven() != "unknown"

	def play(self):
		t = self.tree()
		value = t.root_value()
		uci = t.select_move(0.0)
		self.values.append(value)
		self.board.push_uci(uci)
		for tree in self.trees:
			tree.advance(uci)
		b = self.board
		if b.is_checkmate():
			self.result = "1-0" if b.turn == chess.BLACK else "0-1"
		elif b.is_game_over(claim_draw=True):
			self.result = "1/2-1/2"
		elif len(b.move_stack) >= self.args.max_plies:
			self.result = "1/2-1/2"
		elif len(self.values) >= 4 and all(v < self.args.resign for v in self.values[-3::2]) \
				and all(v > 1 - self.args.resign for v in self.values[-4::2]):
			# Both sides agree for two moves each: the side that just moved is lost.
			self.result = "0-1" if b.turn == chess.BLACK else "1-0"
		if self.result is None:
			self._start_move()

	def score_white(self):
		return {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}[self.result]


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--models", required=True, help="glob of .keras checkpoints")
	parser.add_argument("--openings", required=True, help="evals.npz from distill_lichess_evals.py prep")
	parser.add_argument("--n-openings", type=int, default=12)
	parser.add_argument("--sims", type=int, default=400, help="new simulations per move")
	parser.add_argument("--leaves", type=int, default=8, help="leaves gathered per tree per step")
	parser.add_argument("--parallel", type=int, default=160, help="games in flight")
	parser.add_argument("--c-puct", type=float, default=2.0)
	parser.add_argument("--max-plies", type=int, default=300)
	parser.add_argument("--resign", type=float, default=0.03)
	parser.add_argument("--out", required=True)
	args = parser.parse_args()

	paths = sorted(glob.glob(args.models))
	names = [os.path.splitext(os.path.basename(p))[0] for p in paths]
	n = len(paths)
	os.makedirs(args.out, exist_ok=True)
	nets = load_models(paths)
	openings = pick_openings(args.openings, args.n_openings)
	schedule = [(a, b, o) for a in range(n) for b in range(n) if a != b for o in openings]
	print(f"{n} models, {len(openings)} openings, {len(schedule)} games, {args.sims} sims/move", flush=True)

	score = np.zeros((n, n))   # score[a, b]: a's points with White against b
	games = np.zeros((n, n))
	wdl_counts = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
	pgn_path = os.path.join(args.out, "games.pgn")
	open(pgn_path, "w").close()
	active, next_game, finished, plies, evals = [], 0, 0, 0, 0
	start = last_write = time.time()

	def write_progress(final=False):
		played = games + games.T
		with np.errstate(invalid="ignore", divide="ignore"):
			w = np.where(games > 0, score / np.maximum(games, 1), np.nan)
		state = {"models": names, "games_total": len(schedule), "games_done": finished, "elapsed": time.time() - start,
		         "plies": plies, "evals": evals, "sims": args.sims, "openings": len(openings), "results": wdl_counts,
		         "winrate_white": [[None if x != x else float(x) for x in row] for row in w],
		         "pair_games": played.tolist(), "final": final}
		# Symmetric score of row vs column over both colours (games so far).
		sym = np.full((n, n), np.nan)
		for a in range(n):
			for b in range(n):
				if a != b and played[a, b] > 0:
					sym[a, b] = (score[a, b] + games[b, a] - score[b, a]) / played[a, b]
		state["score"] = [[None if x != x else float(x) for x in row] for row in sym]
		if final or finished == len(schedule):
			W = np.where(games > 0, score / np.maximum(games, 1), 0.5)
			np.fill_diagonal(W, 0.5)
			S = symmetrize_winrate_matrix(W)
			nash = compute_nash_equilibrium(logit_transform(S))
			state["nash"] = nash.tolist()
			state["elo"] = bradley_terry(score, games).tolist()
			np.savez_compressed(os.path.join(args.out, "winrate_matrix.npz"), winrate_matrix=W.astype(np.float32),
			                    models=np.array(names))
		else:
			state["elo"] = bradley_terry(score, games).tolist() if finished > n else None
		tmp = os.path.join(args.out, "progress.json.tmp")
		with open(tmp, "w") as f:
			json.dump(state, f)
		os.replace(tmp, os.path.join(args.out, "progress.json"))

	while next_game < len(schedule) or active:
		while len(active) < args.parallel and next_game < len(schedule):
			a, b, o = schedule[next_game]
			active.append(Game(next_game, a, b, o, args))
			next_game += 1
		# One search step for every game: gather, evaluate per model, back up.
		batches = {}
		for g in active:
			x = g.tree().gather(args.leaves)
			if len(x):
				batches.setdefault(g.player(), []).append((g, x))
		for model, items in batches.items():
			x = np.concatenate([it[1] for it in items])
			policy, wdl = nets[model](x)
			evals += len(x)
			i = 0
			for g, xi in items:
				k = len(xi)
				g.tree().backup(policy[i:i + k], wdl[i:i + k])
				i += k
		gathered = {id(g) for items in batches.values() for g, _ in items}
		for g in active:
			# A tree that found nothing to evaluate three steps running (everything terminal or proven) moves now.
			g.idle = 0 if id(g) in gathered else g.idle + 1
			if g.search_done() or g.idle >= 3:
				g.play()
				plies += 1
		still = []
		for g in active:
			if g.result is None:
				still.append(g)
				continue
			a, b = g.players
			score[a, b] += g.score_white()
			games[a, b] += 1
			wdl_counts[g.result] += 1
			finished += 1
			pgn = chess.pgn.Game.from_board(g.board)
			pgn.headers.update({"White": names[a], "Black": names[b], "Result": g.result, "FEN": g.opening, "SetUp": "1",
			                    "Round": str(g.id)})
			with open(pgn_path, "a") as f:
				print(pgn, file=f, end="\n\n")
		active = still
		if time.time() - last_write > 2:
			last_write = time.time()
			write_progress()
			print(f"{finished}/{len(schedule)} games, {plies} plies, {evals / (time.time() - start):,.0f} evals/s", flush=True)
	write_progress(final=True)
	print(json.dumps({"nash": dict(zip(names, np.round(json.load(open(os.path.join(args.out, "progress.json")))["nash"], 4).tolist()))}), flush=True)


def bradley_terry(score, games, iters=2000):
	"""Elo-scale ratings from all games (colour ignored), anchored so the first model is 0."""
	n = len(score)
	played = games + games.T
	# a's points vs b over both colours, plus one virtual draw per pair that met so a 100% score stays finite.
	won = score + (games.T - score.T) + 0.5 * (played > 0)
	played = played + (played > 0)
	r = np.zeros(n)
	for _ in range(iters):
		p = 1 / (1 + 10 ** ((r[None, :] - r[:, None]) / 400))
		grad = (won - played * p).sum(1)
		hess = (played * p * (1 - p)).sum(1) * np.log(10) / 400 + 1e-9
		r += np.clip(grad / hess, -50, 50) * 0.5
		r -= r[0]
	return r


if __name__ == "__main__":
	main()
