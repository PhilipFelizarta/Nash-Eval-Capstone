"""
Round robin between policy/value checkpoints, each playing with the C++ MCTS (nash_mcts), then the
MaxEnt Nash equilibrium of the meta-game (core/nash.py), as in the original tournament pipeline.

	python self_play_agent/checkpoint_round_robin.py --models 'out/epochs/epoch_*.keras' \
		--openings data/evals.npz --out games/tournament_distill_v2

Every pair plays each opening twice (colours swapped), each side searching `--sims` simulations per move.

Layout: `--workers` processes own the games and their C++ search trees; the main process owns the GPU.
Each worker splits its games into two slots and double-buffers them: while one slot's leaves are on the GPU,
it backs up, moves and gathers for the other. A slot's leaves (sorted by the model to move) go through shared
memory; the main process merges every pending slot into one XLA call that runs all checkpoints at once, writes
the policy/value back into each slot's shared buffers and wakes the workers.

Writes `winrate_matrix.npz` in the project's format (winrate_matrix[a, b] = a's score with White
against b), all games as PGN, and two live files read by play/server.py: `progress.json` (--tournament-progress)
and `games.json` (every game's moves and evals, next to it; the /games viewer).
"""
import argparse
import glob
import json
import multiprocessing as mp
import os
import sys
import time
from collections import deque
from multiprocessing import connection, shared_memory

import chess
import chess.pgn
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "cpp", "mcts"))

import nash_mcts  # noqa: E402

PLANES = (8, 8, 36)
BUCKETS = (64, 128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096)


def load_models(paths):
	"""Returns (call, policy_shape, value_shape). call(x[n_models, bucket, 8, 8, 36]) -> (policy, value) numpy.

	All models run in one XLA-compiled call: per-call dispatch costs ~10-20 ms, so one call per step instead of
	one per model is what keeps the GPU busy. Batches are padded to BUCKETS so XLA compiles a handful of shapes.
	"""
	import tensorflow as tf
	for gpu in tf.config.list_physical_devices("GPU"):
		tf.config.experimental.set_memory_growth(gpu, True)
	models = [tf.keras.models.load_model(path, compile=False) for path in paths]

	# Planes are 0/1, so they travel as uint8; the nets compute in float16 anyway, so the policy comes back as
	# float16. That halves-to-quarters the host<->GPU traffic, which is what bounds this call.
	@tf.function(jit_compile=True)
	def graph(x):
		x = tf.cast(x, tf.float32)
		outs = [model(x[i], training=False) for i, model in enumerate(models)]
		return (tf.stack([tf.cast(o["policy"], tf.float16) for o in outs]),
		        tf.stack([tf.cast(o["value"], tf.float32) for o in outs]))

	def call(x):
		policy, value = graph(tf.constant(x))
		return policy.numpy(), value.numpy()

	policy, value = call(np.zeros((len(models), BUCKETS[0], *PLANES), np.uint8))
	return call, policy.shape[2:], value.shape[2:]


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


class Slot:
	"""A worker's half of its games plus the shared buffers their leaves and evaluations go through."""

	def __init__(self, names, rows, n_models, policy_shape, value_shape):
		self.shm = [shared_memory.SharedMemory(name=name) for name in names]
		self.x = np.ndarray((rows, *PLANES), np.uint8, self.shm[0].buf)
		self.policy = np.ndarray((rows, *policy_shape), np.float16, self.shm[1].buf)
		self.value = np.ndarray((rows, *value_shape), np.float32, self.shm[2].buf)
		self.n_models = n_models
		self.active, self.items = [], []

	def gather(self, leaves):
		"""Gathers leaves from every active game into the input buffer, grouped by model; returns per-model counts."""
		by_model = [[] for _ in range(self.n_models)]
		for g in self.active:
			x = g.tree().gather(leaves)
			if len(x):
				by_model[g.player()].append((g, x))
		row, counts = 0, []
		for items in by_model:
			start = row
			for _, x in items:
				self.x[row:row + len(x)] = x
				row += len(x)
			counts.append(row - start)
		self.items = [it for items in by_model for it in items]
		return counts

	def backup_and_move(self):
		"""Backs up the evaluations, plays every finished search.

		Returns (moves played as (game id, uci, White's expected score from the mover's search), finished games).
		"""
		row, gathered = 0, set()
		for g, x in self.items:
			k = len(x)
			g.tree().backup(self.policy[row:row + k], self.value[row:row + k])
			row += k
			gathered.add(id(g))
		moved, done, still = [], [], []
		for g in self.active:
			# A tree that found nothing to evaluate three steps running (everything terminal or proven) moves now.
			g.idle = 0 if id(g) in gathered else g.idle + 1
			if g.search_done() or g.idle >= 3:
				white_moved = g.board.turn == chess.WHITE
				g.play()
				v = float(g.values[-1])
				moved.append((g.id, g.board.move_stack[-1].uci(), v if white_moved else 1 - v))
			if g.result is None:
				still.append(g)
			else:
				done.append((g.id, g.players, g.opening, [m.uci() for m in g.board.move_stack], g.result))
		self.active = still
		return moved, done


def worker(conn, specs, args, slot_names, rows, n_models, policy_shape, value_shape):
	queue = deque(specs)
	per_slot = rows // args.leaves
	slots = [Slot(names, rows, n_models, policy_shape, value_shape) for names in slot_names]

	def refill(slot):
		while len(slot.active) < per_slot and queue:
			slot.active.append(Game(*queue.popleft(), args))

	for i, slot in enumerate(slots):
		refill(slot)
		conn.send(("eval", i, slot.gather(args.leaves), [], []))
	live = len(slots)
	while live:
		_, i = conn.recv()
		slot = slots[i]
		moved, done = slot.backup_and_move()
		refill(slot)
		if slot.active:
			conn.send(("eval", i, slot.gather(args.leaves), moved, done))
		else:
			conn.send(("bye", i, None, moved, done))
			live -= 1
	for slot in slots:
		for shm in slot.shm:
			shm.close()


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--models", required=True, help="glob of .keras checkpoints")
	parser.add_argument("--openings", required=True, help="evals.npz from distill_lichess_evals.py prep")
	parser.add_argument("--n-openings", type=int, default=12)
	parser.add_argument("--sims", type=int, default=800, help="new simulations per move")
	parser.add_argument("--leaves", type=int, default=16, help="leaves gathered per tree per step")
	parser.add_argument("--parallel", type=int, default=660, help="games in flight (over all workers)")
	parser.add_argument("--workers", type=int, default=5, help="tree-search processes")
	parser.add_argument("--c-puct", type=float, default=2.0)
	parser.add_argument("--max-plies", type=int, default=300)
	parser.add_argument("--resign", type=float, default=0.03)
	parser.add_argument("--out", required=True)
	args = parser.parse_args()

	from core.nash import compute_nash_equilibrium, logit_transform, symmetrize_winrate_matrix

	paths = sorted(glob.glob(args.models))
	names = [os.path.splitext(os.path.basename(p))[0] for p in paths]
	n = len(paths)
	os.makedirs(args.out, exist_ok=True)
	call, policy_shape, value_shape = load_models(paths)
	openings = pick_openings(args.openings, args.n_openings)
	schedule = [(a, b, o) for a in range(n) for b in range(n) if a != b for o in openings]
	print(f"{n} models, {len(openings)} openings, {len(schedule)} games, {args.sims} sims/move, "
	      f"{args.workers} workers x 2 slots, {args.parallel} games in flight", flush=True)

	# Shared buffers: one (input, policy, value) triple per worker slot.
	rows = max(1, args.parallel // (2 * args.workers)) * args.leaves
	sizes = [rows * int(np.prod(PLANES)), rows * int(np.prod(policy_shape)) * 2, rows * int(np.prod(value_shape)) * 4]
	ctx = mp.get_context("spawn")
	shms, slots, conns, procs = [], {}, [], []
	for w in range(args.workers):
		slot_names = []
		for i in range(2):
			triple = [shared_memory.SharedMemory(create=True, size=s) for s in sizes]
			shms += triple
			slot_names.append([s.name for s in triple])
			slots[w, i] = (np.ndarray((rows, *PLANES), np.uint8, triple[0].buf),
			               np.ndarray((rows, *policy_shape), np.float16, triple[1].buf),
			               np.ndarray((rows, *value_shape), np.float32, triple[2].buf))
		parent, child = ctx.Pipe()
		specs = [(gid, a, b, o) for gid, (a, b, o) in enumerate(schedule) if gid % args.workers == w]
		p = ctx.Process(target=worker, args=(child, specs, args, slot_names, rows, n, policy_shape, value_shape),
		                daemon=True)
		p.start()
		child.close()  # so a dead worker shows up as EOF instead of a hang
		conns.append(parent)
		procs.append(p)
	owner = {id(c): w for w, c in enumerate(conns)}

	score = np.zeros((n, n))   # score[a, b]: a's points with White against b
	games = np.zeros((n, n))
	wdl_counts = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
	pgn_path = os.path.join(args.out, "games.pgn")
	open(pgn_path, "w").close()
	finished, plies, evals, steps, gpu_time = 0, 0, 0, 0, 0.0
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

	# Every game that has made a move, for the game viewer (play/server.py /games): moves and, per ply, White's
	# expected score according to the side that just searched.
	trace, results = {}, {}

	def write_games():
		out = [{"id": gid, "white": names[schedule[gid][0]], "black": names[schedule[gid][1]], "opening": schedule[gid][2],
		        "moves": moves, "scores": scores, "result": results.get(gid)}
		       for gid, (moves, scores) in sorted(trace.items())]
		tmp = os.path.join(args.out, "games.json.tmp")
		with open(tmp, "w") as f:
			json.dump(out, f, separators=(",", ":"))
		os.replace(tmp, os.path.join(args.out, "games.json"))

	def record(done):
		nonlocal finished
		with open(pgn_path, "a") as f:
			for gid, (a, b), opening, moves, result in done:
				results[gid] = result
				score[a, b] += {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}[result]
				games[a, b] += 1
				wdl_counts[result] += 1
				finished += 1
				board = chess.Board(opening)
				for m in moves:
					board.push_uci(m)
				pgn = chess.pgn.Game.from_board(board)
				pgn.headers.update({"White": names[a], "Black": names[b], "Result": result, "FEN": opening,
				                    "SetUp": "1", "Round": str(gid)})
				print(pgn, file=f, end="\n\n")

	def evaluate(pending):
		"""One GPU call for every pending slot; splits in two if a model's batch outgrows the largest bucket."""
		nonlocal evals, gpu_time
		totals = np.sum([counts for _, _, counts in pending], axis=0)
		if totals.max() > BUCKETS[-1] and len(pending) > 1:
			evaluate(pending[:len(pending) // 2])
			evaluate(pending[len(pending) // 2:])
			return
		size = next(b for b in BUCKETS if b >= max(1, totals.max()))
		x = np.zeros((n, size, *PLANES), np.uint8)
		fill = [0] * n
		for w, i, counts in pending:
			src, row = slots[w, i][0], 0
			for m, k in enumerate(counts):
				x[m, fill[m]:fill[m] + k] = src[row:row + k]
				fill[m] += k
				row += k
		t0 = time.time()
		policy, value = call(x)
		gpu_time += time.time() - t0
		evals += int(totals.sum())
		fill = [0] * n
		for w, i, counts in pending:
			_, dst_p, dst_v = slots[w, i]
			row = 0
			for m, k in enumerate(counts):
				dst_p[row:row + k] = policy[m, fill[m]:fill[m] + k]
				dst_v[row:row + k] = value[m, fill[m]:fill[m] + k]
				fill[m] += k
				row += k
			conns[w].send(("ok", i))

	live = 2 * args.workers
	last = (0, 0, 0.0)
	open_conns, slots_left = list(conns), [2] * args.workers
	pending, first = [], None
	try:
		while live:
			# Batch: wait until at least half the live slots are waiting, or 10 ms after the first one arrived.
			timeout = None if not pending else max(0.0, first + 0.01 - time.time())
			for c in connection.wait(open_conns, timeout):
				while c in open_conns and c.poll():
					kind, i, counts, moved, done = c.recv()
					plies += len(moved)
					for gid, uci, white_score in moved:
						trace.setdefault(gid, ([], []))
						trace[gid][0].append(uci)
						trace[gid][1].append(round(white_score, 4))
					record(done)
					if kind == "eval":
						pending.append((owner[id(c)], i, counts))
						first = first or time.time()
					else:
						live -= 1
						slots_left[owner[id(c)]] -= 1
						if not slots_left[owner[id(c)]]:
							open_conns.remove(c)
			if pending and (len(pending) >= max(1, (live + 1) // 2) or time.time() - first >= 0.01):
				evaluate(pending)
				steps += 1
				pending, first = [], None
			if time.time() - last_write > 2:
				now = time.time()
				write_progress()
				write_games()
				d_evals, d_steps, d_gpu = evals - last[0], steps - last[1], gpu_time - last[2]
				print(f"{finished}/{len(schedule)} games, {plies} plies, {d_evals / (now - last_write):,.0f} evals/s, "
				      f"gpu {d_gpu / (now - last_write):.0%}, {d_evals / max(d_steps, 1):,.0f} evals/call", flush=True)
				last_write, last = now, (evals, steps, gpu_time)
		write_progress(final=True)
		write_games()
	finally:
		for p in procs:
			p.join(timeout=5)
		for s in shms:
			s.close()
			s.unlink()
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
