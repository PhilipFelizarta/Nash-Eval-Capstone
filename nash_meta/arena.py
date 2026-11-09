"""
Arena: the GPU half of the Nash meta-game training loop (see nash_meta/README.md).

It plays two kinds of games, all with the C++ MCTS at `--sims` simulations per move:

* actor games, the exploration: both players are drawn independently from the current Nash mixture over the
  model pool (with replacement), from a large pool of preset balanced openings, with root Dirichlet noise and
  temperature sampling for the first `--temp-plies` plies. Finished games go to RUN/games/*.jsonl for the labeler.
* league games, the evaluation: when the trainer drops a checkpoint into RUN/pool/, it plays every model in the
  pool with both colours on a fixed set of evaluation openings (no noise), then the MaxEnt Nash of the pool is
  recomputed (core/nash.py) and becomes the actor's new sampling distribution. League games take priority.

The pool is capped at `--pool-size` models (oldest evicted). Actor games pause while the labeler's backlog is
above `--max-backlog` positions, so every generated position gets its Stockfish subgame.

	python nash_meta/arena.py --run RUN --init-models 'out/v2/epochs/epoch_*.keras' \
		--init-tournament out/v2/tournament --openings data/evals.npz

State is RUN/league.json (pool, Nash, pairwise results, throughput; rewritten every few seconds). Restarting
with the same --run resumes: finished league games are kept and an unfinished round only plays what is missing.
"""
import argparse
import glob
import json
import multiprocessing as mp
import os
import random
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import connection, shared_memory

import chess
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "cpp", "mcts"))

import nash_mcts  # noqa: E402
from self_play_agent.checkpoint_round_robin import BUCKETS, PLANES, bradley_terry, pick_openings  # noqa: E402


class Game:
	def __init__(self, spec, args):
		self.id, self.kind, self.opening = spec["id"], spec["kind"], spec["opening"]
		self.players = (spec["white"], spec["black"])
		self.explore = self.kind == "actor"
		self.args = args
		self.board = chess.Board(self.opening)
		self.trees = []
		for _ in range(2):
			t = nash_mcts.MCTS(c_puct=args.c_puct, dirichlet_alpha=args.dirichlet_alpha,
			                   dirichlet_epsilon=args.dirichlet_epsilon if self.explore else 0.0,
			                   seed=random.getrandbits(31))
			t.reset(self.opening, [], self.explore)
			self.trees.append(t)
		self.result = None
		self.values = []   # root value (side to move) per ply
		self.scores = []   # White's expected score per ply, from the side that searched
		self._start_move()

	def tree(self):
		return self.trees[0 if self.board.turn == chess.WHITE else 1]

	def player(self):
		return self.players[0 if self.board.turn == chess.WHITE else 1]

	def _start_move(self):
		self.idle = 0
		self.target = self.tree().root_visits() + self.args.sims

	def search_done(self):
		t = self.tree()
		return t.root_visits() >= self.target or t.root_proven() != "unknown"

	def play(self):
		t = self.tree()
		value = t.root_value()
		explore = self.explore and len(self.board.move_stack) < self.args.temp_plies
		uci = t.select_move(self.args.temperature if explore else 0.0)
		self.values.append(value)
		self.scores.append(round(value if self.board.turn == chess.WHITE else 1 - value, 4))
		self.board.push_uci(uci)
		for tree in self.trees:
			tree.advance(uci)
		b = self.board
		if b.is_checkmate():
			self.result = "1-0" if b.turn == chess.BLACK else "0-1"
		elif b.is_game_over(claim_draw=True) or len(b.move_stack) >= self.args.max_plies:
			self.result = "1/2-1/2"
		elif len(self.values) >= 4 and all(v < self.args.resign for v in self.values[-3::2]) \
				and all(v > 1 - self.args.resign for v in self.values[-4::2]):
			self.result = "0-1" if b.turn == chess.BLACK else "1-0"
		if self.result is None:
			self._start_move()

	def record(self):
		return {"id": self.id, "kind": self.kind, "white": self.players[0], "black": self.players[1],
		        "opening": self.opening, "moves": [m.uci() for m in self.board.move_stack], "scores": self.scores,
		        "result": self.result}


class Slot:
	"""Half of a worker's games plus the shared buffers their leaves and evaluations go through."""

	def __init__(self, names, rows):
		self.shm = [shared_memory.SharedMemory(name=name) for name in names]
		self.x = np.ndarray((rows, *PLANES), np.uint8, self.shm[0].buf)
		self.policy = np.ndarray((rows, nash_mcts.POLICY_SIZE), np.float16, self.shm[1].buf)
		self.value = np.ndarray((rows, 3), np.float32, self.shm[2].buf)
		self.active, self.items = [], []

	def gather(self, leaves):
		"""Gathers leaves into the input buffer grouped by model; returns [(model id, rows)] in buffer order."""
		by_model = {}
		for g in self.active:
			x = g.tree().gather(leaves)
			if len(x):
				by_model.setdefault(g.player(), []).append((g, x))
		row, counts, self.items = 0, [], []
		for model in sorted(by_model):
			start = row
			for g, x in by_model[model]:
				self.x[row:row + len(x)] = x
				row += len(x)
				self.items.append((g, x))
			counts.append((model, row - start))
		return counts

	def backup_and_move(self):
		row, gathered = 0, set()
		for g, x in self.items:
			k = len(x)
			g.tree().backup(self.policy[row:row + k], self.value[row:row + k])
			row += k
			gathered.add(id(g))
		moved, done, still = 0, [], []
		for g in self.active:
			g.idle = 0 if id(g) in gathered else g.idle + 1
			if g.search_done() or g.idle >= 3:
				g.play()
				moved += 1
			if g.result is None:
				still.append(g)
			else:
				done.append(g.record())
		self.active = still
		return moved, done


def worker(conn, args, slot_names, rows):
	"""Owns games and their trees; the main process feeds it game specs and evaluations."""
	queue = deque()
	per_slot = rows // args.leaves
	slots = [Slot(names, rows) for names in slot_names]
	busy = [False] * len(slots)

	def refill(slot):
		while len(slot.active) < per_slot and queue:
			slot.active.append(Game(queue.popleft(), args))

	while True:
		msg = conn.recv()
		if msg[0] == "stop":
			break
		if msg[0] == "specs":
			queue.extend(msg[1])
			for i, slot in enumerate(slots):
				if not busy[i]:
					refill(slot)
					if slot.active:
						busy[i] = True
						conn.send(("eval", i, slot.gather(args.leaves), 0, []))
			continue
		i = msg[1]
		slot = slots[i]
		moved, done = slot.backup_and_move()
		refill(slot)
		if slot.active:
			conn.send(("eval", i, slot.gather(args.leaves), moved, done))
		else:
			busy[i] = False
			conn.send(("idle", i, None, moved, done))
	for slot in slots:
		for shm in slot.shm:
			shm.close()


class Models:
	"""Checkpoints on the GPU, one XLA-compiled call per model; inputs as uint8 planes, policy back as float16."""

	def __init__(self):
		import tensorflow as tf
		for gpu in tf.config.list_physical_devices("GPU"):
			tf.config.experimental.set_memory_growth(gpu, True)
		self.tf = tf
		self.calls = {}

	def load(self, model_id, path):
		tf = self.tf
		model = tf.keras.models.load_model(path, compile=False)

		@tf.function(jit_compile=True)
		def call(x):
			out = model(tf.cast(x, tf.float32), training=False)
			return tf.cast(out["policy"], tf.float16), tf.cast(out["value"], tf.float32)
		self.calls[model_id] = call

	def evaluate(self, model_id, x):
		size = next((b for b in BUCKETS if b >= len(x)), None)
		if size is None:
			parts = [self.evaluate(model_id, x[i:i + BUCKETS[-1]]) for i in range(0, len(x), BUCKETS[-1])]
			return np.concatenate([p for p, _ in parts]), np.concatenate([v for _, v in parts])
		buf = np.zeros((size, *PLANES), np.uint8)
		buf[:len(x)] = x
		policy, value = self.calls[model_id](self.tf.constant(buf))
		return policy.numpy()[:len(x)], value.numpy()[:len(x)]


def solve_nash(W):
	"""MaxEnt Nash of the pool given White's score matrix (core/nash.py; takes minutes for a dozen models)."""
	import contextlib
	from core.nash import compute_nash_equilibrium, logit_transform, symmetrize_winrate_matrix
	with open(os.devnull, "w") as null, contextlib.redirect_stderr(null):
		nash = compute_nash_equilibrium(logit_transform(symmetrize_winrate_matrix(W)))
	nash = np.maximum(nash, 0)
	return nash / nash.sum()


class League:
	"""Pool membership, pairwise results (White's points per ordered pair) and the Nash mixture."""

	def __init__(self, path):
		self.path = path
		self.models = []       # [{"name", "path"}], index = model id
		self.pool = []         # model ids, oldest first
		self.score, self.games = {}, {}   # "a,b" -> White's points / games with a as White against b
		self.sigma_ids = []    # the pool when the Nash was last computed (new checkpoints join once rated)
		self.sigma = []        # Nash weight (plus --nash-mix uniform) per sigma_ids member
		self.history = []      # one entry per Nash recomputation
		self.rounds = []       # model ids whose evaluation round is not complete yet
		self.solving = None    # (rated ids, score, games, future) while a Nash solve runs
		self.stale = False     # results changed during that solve: solve again when it finishes

	def key(self, a, b):
		return f"{a},{b}"

	def add(self, name, path):
		self.models.append({"name": name, "path": path})
		return len(self.models) - 1

	def result(self, a, b, white_points):
		k = self.key(a, b)
		self.score[k] = self.score.get(k, 0.0) + white_points
		self.games[k] = self.games.get(k, 0) + 1

	def matrices(self, ids=None):
		ids = self.pool if ids is None else ids
		n = len(ids)
		score, games = np.zeros((n, n)), np.zeros((n, n))
		for i, a in enumerate(ids):
			for j, b in enumerate(ids):
				if i != j:
					score[i, j] = self.score.get(self.key(a, b), 0.0)
					games[i, j] = self.games.get(self.key(a, b), 0)
		return score, games

	def recompute(self, solver):
		"""Starts a MaxEnt Nash solve over the rated pool (members whose evaluation round is complete) in `solver`,
		a separate process, so games keep running; poll() applies it."""
		rated = [m for m in self.pool if m not in self.rounds]
		if not rated:  # nothing rated yet (a new run without --init-tournament): sample the pool uniformly
			self.sigma_ids, self.sigma = list(self.pool), [1 / len(self.pool)] * len(self.pool)
			return
		if self.solving is not None:
			self.stale = True
			return
		score, games = self.matrices(rated)
		W = np.where(games > 0, score / np.maximum(games, 1), 0.5)
		np.fill_diagonal(W, 0.5)
		self.solving, self.stale = (rated, score, games, solver.submit(solve_nash, W)), False

	def poll(self, mix, solver):
		"""Applies a finished Nash solve (to the members still in the pool); returns whether the mixture changed."""
		if self.solving is None or not self.solving[3].done():
			return False
		rated, score, games, future = self.solving
		self.solving = None
		self.set_nash(rated, future.result(), bradley_terry(score, games), mix)
		if self.stale:
			self.recompute(solver)
		return True

	def set_nash(self, rated, nash, elo, mix):
		nash = np.asarray(nash, float)
		self.history.append({"time": time.time(), "pool": [self.models[m]["name"] for m in rated],
		                     "nash": nash.tolist(), "elo": (np.asarray(elo) - np.max(elo)).tolist()})
		keep = [i for i, m in enumerate(rated) if m in self.pool]
		weights = nash[keep] if nash[keep].sum() > 0 else np.ones(len(keep))
		weights = weights / weights.sum()
		self.sigma_ids = [rated[i] for i in keep]
		self.sigma = ((1 - mix) * weights + mix / len(keep)).tolist()

	def save(self, extra):
		score, games = self.matrices()
		state = {"models": self.models, "pool": self.pool, "sigma_ids": self.sigma_ids, "sigma": self.sigma, "history": self.history,
		         "rounds": self.rounds, "score": self.score, "games": self.games,
		         "pool_names": [self.models[m]["name"] for m in self.pool],
		         "pool_score": [[None if games[i, j] == 0 else score[i, j] / games[i, j] for j in range(len(self.pool))]
		                        for i in range(len(self.pool))], **extra}
		tmp = self.path + ".tmp"
		with open(tmp, "w") as f:
			json.dump(state, f)
		os.replace(tmp, self.path)

	@classmethod
	def load(cls, path):
		league = cls(path)
		with open(path) as f:
			state = json.load(f)
		for k in ("models", "pool", "sigma_ids", "sigma", "history", "rounds", "score", "games"):
			setattr(league, k, state[k])
		return league


def main():
	p = argparse.ArgumentParser()
	p.add_argument("--run", required=True, help="run directory shared with labeler.py and trainer.py")
	p.add_argument("--init-models", help="glob of the pretraining checkpoints that seed the pool (new runs)")
	p.add_argument("--init-tournament", help="checkpoint_round_robin.py output dir: reuse its results for the seed pool")
	p.add_argument("--openings", required=True, help="evals.npz: preset openings come from here")
	p.add_argument("--actor-openings", type=int, default=2000)
	p.add_argument("--league-openings", type=int, default=12, help="per colour, per pair (same set as the round robin)")
	p.add_argument("--pool-size", type=int, default=12)
	p.add_argument("--nash-mix", type=float, default=0.0, help="mix this much uniform into the actor's Nash sampling")
	p.add_argument("--sims", type=int, default=800)
	p.add_argument("--leaves", type=int, default=16)
	p.add_argument("--parallel", type=int, default=480, help="games in flight")
	p.add_argument("--workers", type=int, default=5)
	p.add_argument("--c-puct", type=float, default=2.0)
	p.add_argument("--dirichlet-alpha", type=float, default=0.3)
	p.add_argument("--dirichlet-epsilon", type=float, default=0.25, help="actor games only")
	p.add_argument("--temperature", type=float, default=1.0, help="actor games, first --temp-plies plies")
	p.add_argument("--temp-plies", type=int, default=12)
	p.add_argument("--max-plies", type=int, default=300)
	p.add_argument("--resign", type=float, default=0.03)
	p.add_argument("--max-backlog", type=int, default=20_000, help="pause actor games above this many unlabeled positions")
	p.add_argument("--shard-games", type=int, default=50, help="actor games per RUN/games file")
	args = p.parse_args()

	for d in ("pool", "games"):
		os.makedirs(os.path.join(args.run, d), exist_ok=True)
	league_path = os.path.join(args.run, "league.json")
	models = Models()
	ctx = mp.get_context("spawn")
	solver = ProcessPoolExecutor(1, mp_context=ctx)
	league_openings = pick_openings(args.openings, args.league_openings)
	actor_openings = [o for o in pick_openings(args.openings, args.actor_openings + args.league_openings, seed=1)
	                  if o not in league_openings][:args.actor_openings]

	if os.path.exists(league_path):
		league = League.load(league_path)
		print(f"resuming: {len(league.models)} models, pool {len(league.pool)}", flush=True)
	else:
		league = League(league_path)
		paths = sorted(glob.glob(args.init_models))
		for path in paths:
			league.pool.append(league.add(os.path.splitext(os.path.basename(path))[0], os.path.abspath(path)))
		if args.init_tournament:
			# Seed results from the round robin (same evaluation openings, so nothing is replayed).
			with open(os.path.join(args.init_tournament, "progress.json")) as f:
				rr = json.load(f)
			ids = {league.models[m]["name"]: m for m in league.pool}
			for i, a in enumerate(rr["models"]):
				for j, b in enumerate(rr["models"]):
					w, played = rr["winrate_white"][i][j], rr["pair_games"][i][j]
					if i != j and w is not None and a in ids and b in ids:
						k = league.key(ids[a], ids[b])
						league.games[k] = played // 2
						league.score[k] = w * (played // 2)
		else:
			league.pool = league.pool[-args.pool_size:]
			league.rounds = list(league.pool)  # play the seed pool's round robin first
		league.pool = league.pool[-args.pool_size:]
		names = [league.models[m]["name"] for m in league.pool]
		if args.init_tournament and rr.get("nash") and names == rr["models"]:
			# The round robin already solved this exact pool.
			score, games = league.matrices()
			league.set_nash(list(league.pool), rr["nash"], bradley_terry(score, games), args.nash_mix)
		else:
			league.recompute(solver)
	if league.solving is None and not league.sigma:
		league.recompute(solver)
	for m, info in enumerate(league.models):
		models.load(m, info["path"])
	print("nash:", ", ".join(f"{league.models[m]['name']} {s:.3f}" for m, s in zip(league.sigma_ids, league.sigma)), flush=True)

	# Workers and their shared buffers (two slots each, double-buffered as in checkpoint_round_robin.py).
	rows = max(1, args.parallel // (2 * args.workers)) * args.leaves
	sizes = [rows * int(np.prod(PLANES)), rows * nash_mcts.POLICY_SIZE * 2, rows * 3 * 4]
	shms, slots, conns, procs = [], {}, [], []
	for w in range(args.workers):
		names = []
		for i in range(2):
			triple = [shared_memory.SharedMemory(create=True, size=s) for s in sizes]
			shms += triple
			names.append([s.name for s in triple])
			slots[w, i] = (np.ndarray((rows, *PLANES), np.uint8, triple[0].buf),
			               np.ndarray((rows, nash_mcts.POLICY_SIZE), np.float16, triple[1].buf),
			               np.ndarray((rows, 3), np.float32, triple[2].buf))
		parent, child = ctx.Pipe()
		proc = ctx.Process(target=worker, args=(child, args, names, rows), daemon=True)
		proc.start()
		child.close()
		conns.append(parent)
		procs.append(proc)
	owner = {id(c): w for w, c in enumerate(conns)}
	capacity = 3 * (rows // args.leaves)  # specs a worker holds: two full slots plus a queue

	league_queue, outstanding, spec_worker = deque(), [0] * args.workers, {}
	round_left = {}  # model id -> league games of its round still to finish
	round_of = {}    # league game id -> the model whose round it belongs to
	queued = {}      # "white,black" -> league games scheduled but not finished
	next_id = int(time.time() * 1000) % 10**9 * 1000
	stats = {"actor_games": 0, "actor_positions": 0, "league_games": 0, "plies": 0, "evals": 0, "throttled": False}
	shard, shard_n, shard_positions = None, 0, 0
	start = time.time()

	def schedule_round(new):
		"""Every game `new` still owes the pool: each opponent, both colours, every evaluation opening."""
		nonlocal next_id
		count = 0
		for opp in league.pool:
			if opp == new:
				continue
			for white, black in ((new, opp), (opp, new)):
				k = league.key(white, black)
				for o in league_openings[league.games.get(k, 0) + queued.get(k, 0):]:
					next_id += 1
					league_queue.append({"id": next_id, "kind": "league", "white": white, "black": black, "opening": o})
					round_of[next_id] = new
					queued[k] = queued.get(k, 0) + 1
					count += 1
		round_left[new] = round_left.get(new, 0) + count
		print(f"league round for {league.models[new]['name']}: {count} games", flush=True)
		if count == 0:
			finish_round(new)

	def finish_round(new):
		round_left.pop(new, None)
		if new in league.rounds:
			league.rounds.remove(new)
		if new in league.pool:
			league.recompute(solver)

	def add_checkpoint(path):
		name = os.path.splitext(os.path.basename(path))[0]
		m = league.add(name, os.path.abspath(path))
		models.load(m, path)
		league.pool.append(m)
		league.rounds.append(m)
		while len(league.pool) > args.pool_size:
			evict(league.pool[0])
		schedule_round(m)

	def evict(old):
		"""Drops the oldest pool member and the league games it still had queued."""
		league.pool.remove(old)
		round_left.pop(old, None)
		if old in league.rounds:
			league.rounds.remove(old)
		keep, dropped = deque(), []
		for spec in league_queue:
			(dropped if old in (spec["white"], spec["black"]) else keep).append(spec)
		league_queue.clear()
		league_queue.extend(keep)
		for spec in dropped:
			queued[league.key(spec["white"], spec["black"])] -= 1
			round_done(round_of.pop(spec["id"]))

	def round_done(m):
		if m in round_left:
			round_left[m] -= 1
			if round_left[m] == 0:
				finish_round(m)

	for m in list(league.rounds):
		schedule_round(m)

	def backlog():
		"""Positions waiting for a Stockfish subgame: the labeler's queue plus the shard still being written."""
		try:
			with open(os.path.join(args.run, "labeler.json")) as f:
				queued = json.load(f)["backlog"]
		except (OSError, ValueError, KeyError):
			queued = 0
		return queued + shard_positions

	def actor_spec():
		nonlocal next_id
		next_id += 1
		if league.sigma:
			white, black = random.choices(league.sigma_ids, weights=league.sigma, k=2)
		else:  # first Nash still being solved
			white, black = random.choices(league.pool, k=2)
		return {"id": next_id, "kind": "actor", "white": white, "black": black, "opening": random.choice(actor_openings)}

	def dispatch():
		throttled = backlog() > args.max_backlog
		stats["throttled"] = throttled
		for w in range(args.workers):
			batch = []
			while outstanding[w] + len(batch) < capacity:
				if league_queue:
					batch.append(league_queue.popleft())
				elif not throttled:
					batch.append(actor_spec())
				else:
					break
			if batch:
				for s in batch:
					spec_worker[s["id"]] = w
				outstanding[w] += len(batch)
				conns[w].send(("specs", batch))

	def write_actor(rec):
		nonlocal shard, shard_n, shard_positions
		if shard is None:
			shard = os.path.join(args.run, "games", f"actor_{int(time.time() * 1000)}.jsonl")
			shard_n = 0
		shard_positions += len(rec["moves"])
		rec = {**rec, "white": league.models[rec["white"]]["name"], "black": league.models[rec["black"]]["name"]}
		with open(shard + ".part", "a") as f:
			f.write(json.dumps(rec, separators=(",", ":")) + "\n")
		shard_n += 1
		if shard_n >= args.shard_games:
			flush_shard()

	def flush_shard():
		nonlocal shard, shard_positions
		if shard is not None:
			os.replace(shard + ".part", shard)
			shard, shard_positions = None, 0

	def finished(rec):
		w = spec_worker.pop(rec["id"])
		outstanding[w] -= 1
		if rec["kind"] == "actor":
			stats["actor_games"] += 1
			stats["actor_positions"] += len(rec["moves"])
			write_actor(rec)
			return
		stats["league_games"] += 1
		a, b = rec["white"], rec["black"]
		league.result(a, b, {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}[rec["result"]])
		queued[league.key(a, b)] -= 1
		round_done(round_of.pop(rec["id"]))

	known = {info["path"] for info in league.models}
	pending, first = [], None
	last_scan = last_write = last_flush = 0.0
	last = (0, 0.0)
	gpu_time = 0.0
	dispatch()
	try:
		while True:
			now = time.time()
			if now - last_scan > 15:
				last_scan = now
				for path in sorted(glob.glob(os.path.join(args.run, "pool", "*.keras"))):
					if os.path.abspath(path) not in known:
						known.add(os.path.abspath(path))
						add_checkpoint(path)
			if league.poll(args.nash_mix, solver):
				print("nash:", ", ".join(f"{league.models[m]['name']} {s:.3f}" for m, s in zip(league.sigma_ids, league.sigma)),
				      flush=True)
			timeout = 1.0 if not pending else max(0.0, first + 0.01 - time.time())
			for c in connection.wait(conns, timeout):
				while c.poll():
					kind, i, counts, moved, done = c.recv()
					stats["plies"] += moved
					for rec in done:
						finished(rec)
					if kind == "eval":
						pending.append((owner[id(c)], i, counts))
						first = first or time.time()
			busy = sum(1 for w in range(args.workers) if outstanding[w] > 0) * 2
			if pending and (len(pending) >= max(1, busy // 2) or time.time() - first >= 0.01):
				# One call per model with rows pending, over every waiting slot.
				by_model = {}
				for w, i, counts in pending:
					row = 0
					for model, k in counts:
						by_model.setdefault(model, []).append((w, i, row, k))
						row += k
				t0 = time.time()
				for model, parts in by_model.items():
					x = np.concatenate([slots[w, i][0][r:r + k] for w, i, r, k in parts])
					policy, value = models.evaluate(model, x)
					stats["evals"] += len(x)
					o = 0
					for w, i, r, k in parts:
						slots[w, i][1][r:r + k] = policy[o:o + k]
						slots[w, i][2][r:r + k] = value[o:o + k]
						o += k
				gpu_time += time.time() - t0
				for w, i, _ in pending:
					conns[w].send(("ok", i))
				pending, first = [], None
			dispatch()
			if time.time() - last_flush > 120:
				last_flush = time.time()
				flush_shard()
			if time.time() - last_write > 3:
				el = time.time() - last_write
				rate = (stats["evals"] - last[0]) / el
				gpu = (gpu_time - last[1]) / el
				last_write, last = time.time(), (stats["evals"], gpu_time)
				league.save({"stats": {**stats, "evals_per_s": rate, "gpu_busy": gpu, "elapsed": time.time() - start,
				                       "league_queue": len(league_queue), "in_flight": sum(outstanding),
				                       "backlog": backlog(), "round_left": {league.models[m]["name"]: n for m, n in round_left.items()}},
				             "updated": time.time()})
				print(f"actor {stats['actor_games']} games / {stats['actor_positions']} positions, league "
				      f"{stats['league_games']} (queue {len(league_queue)}), {rate:,.0f} evals/s, gpu {gpu:.0%}"
				      f"{', THROTTLED' if stats['throttled'] else ''}", flush=True)
	finally:
		flush_shard()
		solver.shutdown(wait=False, cancel_futures=True)
		for c in conns:
			try:
				c.send(("stop",))
			except OSError:
				pass
		for proc in procs:
			proc.join(timeout=5)
		for s in shms:
			s.close()
			s.unlink()


if __name__ == "__main__":
	main()
