"""GPU nodes/second: old Python searches vs Python MCTS vs C++ MCTS, with real (untrained) ResNets.

nodes = positions evaluated by the network (MCTS: one per simulation; depth-1/2: every leaf).
Run from the repo root after building nash_mcts: python cpp/mcts/bench_gpu.py
Needs TensorFlow with a GPU. Under WSL, put /usr/lib/wsl/lib and the pip nvidia/*/lib dirs on LD_LIBRARY_PATH.
"""
import math, os, sys, time
import numpy as np
import chess

sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "cpp", "mcts"))
import tensorflow as tf
for gpu in tf.config.list_physical_devices("GPU"):
	tf.config.experimental.set_memory_growth(gpu, True)
from core import model_framework, self_play, move_encoding  # sets mixed_float16
from core.chess_environment import fast_fen_to_example
import nash_mcts

print("TF", tf.__version__, tf.config.list_physical_devices("GPU"), flush=True)
FEN = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"  # middlegame, 48 legal moves
MIN_TIME = float(os.environ.get("MIN_TIME", "4"))


def timed(fn, min_time=MIN_TIME):
	fn()  # warm-up (tracing, cuDNN autotune)
	runs, start = 0, time.perf_counter()
	while True:
		fn(); runs += 1
		elapsed = time.perf_counter() - start
		if elapsed >= min_time:
			return runs, elapsed


def fast_call(model, outputs):
	@tf.function(input_signature=[tf.TensorSpec([None, 8, 8, 36], tf.float32)])
	def f(x):
		out = model(x, training=False)
		return tuple(tf.cast(out[k], tf.float32) for k in outputs) if outputs else tf.cast(out, tf.float32)
	return f


class Counting:
	"""Keras-like wrapper for the old searches: .predict(x) through either model.predict or a tf.function."""
	def __init__(self, model, fast):
		self.model, self.positions = model, 0
		self.f = fast_call(model, None) if fast else None
	def predict(self, x, batch_size=None, verbose=0):
		self.positions += len(x)
		if self.f is None:
			return self.model.predict(x, batch_size=batch_size, verbose=0)
		return self.f(tf.constant(x, tf.float32)).numpy()


def bench_old(select, value_model, fast):
	m = Counting(value_model, fast)
	runs, elapsed = timed(lambda: select(chess.Board(FEN), m))
	return m.positions / elapsed


class Node:
	__slots__ = ("move", "parent", "children", "prior", "n", "w", "terminal")
	def __init__(self, move, parent, prior):
		self.move, self.parent, self.prior = move, parent, prior
		self.children, self.n, self.w, self.terminal = None, 0, 0.0, None


def py_mcts(evaluate, sims, c_puct=2.0, fpu_reduction=0.25):
	"""Pure-Python PUCT, same rules as nash_mcts, one network call per simulation."""
	board = chess.Board(FEN)
	root = Node(None, None, 1.0)
	for _ in range(sims):
		node, pushed = root, 0
		while node.children:
			parent_q = 1.0 - node.w / node.n if node.n else 0.5
			sqrt_n = math.sqrt(max(node.n, 1))
			best, best_score = None, -1e18
			for c in node.children:
				q = c.w / c.n if c.n else parent_q - fpu_reduction
				score = q + c_puct * c.prior * sqrt_n / (1 + c.n)
				if score > best_score:
					best, best_score = c, score
			board.push(best.move); pushed += 1
			node = best
		if node.terminal is not None:
			v = node.terminal
		elif board.is_checkmate():
			node.terminal = v = 0.0
		elif self_play.is_drawn(board):
			node.terminal = v = 0.5
		else:
			logits, wdl = evaluate(fast_fen_to_example(board.fen())[None])
			moves, idx = move_encoding.legal_move_indices(board)
			priors = move_encoding.masked_policy(logits[0], idx)
			node.children = [Node(m, node, p) for m, p in zip(moves, priors)]
			v = float(wdl[0, 0] + 0.5 * wdl[0, 1])
		val = 1.0 - v
		while node is not None:
			node.n += 1; node.w += val; val = 1.0 - val
			node = node.parent
		for _ in range(pushed):
			board.pop()


def bench_cpp(evaluate, batch, trees=1, sims=800):
	def run():
		ts = []
		for i in range(trees):
			t = nash_mcts.MCTS(seed=i); t.reset(FEN, [], False); ts.append(t)
		while True:
			active = [t for t in ts if t.root_visits() < sims]
			if not active:
				return
			xs = [t.gather(min(batch, sims - t.root_visits())) for t in active]
			sizes = [len(x) for x in xs]
			if sum(sizes):
				policy, wdl = evaluate(np.concatenate([x for x in xs if len(x)]))
			else:
				policy, wdl = np.zeros((0, 4672), np.float32), np.zeros((0, 3), np.float32)
			o = 0
			for t, k in zip(active, sizes):
				t.backup(policy[o:o + k], wdl[o:o + k]); o += k
	runs, elapsed = timed(run)
	return runs * trees * sims / elapsed


def raw_throughput(f, batch):
	x = tf.constant(np.random.rand(batch, 8, 8, 36).astype(np.float32))
	def run():
		out = f(x)
		_ = out[0].numpy()
	runs, elapsed = timed(run, 2.0)
	return runs * batch / elapsed


results = {}
for blocks, filters in [(4, 512), (6, 64)]:
	name = f"{blocks}x{filters}"
	value_model = model_framework.exploratory_model(FILTERS=filters, BLOCKS=blocks)
	pv_model = model_framework.exploratory_model(FILTERS=filters, BLOCKS=blocks, policy_head=True)
	pv = fast_call(pv_model, ["policy", "value"])
	evaluate = lambda x: tuple(t.numpy() for t in pv(tf.constant(x, tf.float32)))

	r = {}
	r["raw"] = {b: raw_throughput(pv, b) for b in (1, 64, 1024)}
	print(name, "raw", {b: round(v) for b, v in r["raw"].items()}, flush=True)
	r["py_d1_predict"] = bench_old(self_play.select_stochastic_move, value_model, fast=False)
	r["py_d2_predict"] = bench_old(self_play.select_depth2_move, value_model, fast=False)
	r["py_d1_fast"] = bench_old(self_play.select_stochastic_move, value_model, fast=True)
	r["py_d2_fast"] = bench_old(self_play.select_depth2_move, value_model, fast=True)
	runs, elapsed = timed(lambda: py_mcts(evaluate, 300))
	r["py_mcts"] = runs * 300 / elapsed
	r["cpp_b1"] = bench_cpp(evaluate, 1, sims=300)
	r["cpp_b64"] = bench_cpp(evaluate, 64)
	r["cpp_16x64"] = bench_cpp(evaluate, 64, trees=16)
	results[name] = r
	print(name, {k: (round(v) if not isinstance(v, dict) else v) for k, v in r.items() if k != "raw"}, flush=True)

print("\nRESULTS (nodes/s)")
rows = [
	("Network alone, batch 1", lambda r: r["raw"][1]),
	("Network alone, batch 64", lambda r: r["raw"][64]),
	("Network alone, batch 1024", lambda r: r["raw"][1024]),
	("Old Python depth-1 (model.predict)", lambda r: r["py_d1_predict"]),
	("Old Python depth-2 (model.predict)", lambda r: r["py_d2_predict"]),
	("Old Python depth-1 (tf.function)", lambda r: r["py_d1_fast"]),
	("Old Python depth-2 (tf.function)", lambda r: r["py_d2_fast"]),
	("Python MCTS, batch 1", lambda r: r["py_mcts"]),
	("C++ MCTS, batch 1", lambda r: r["cpp_b1"]),
	("C++ MCTS, batch 64", lambda r: r["cpp_b64"]),
	("C++ MCTS, 16 trees x 64", lambda r: r["cpp_16x64"]),
]
print(f"{'':<38}" + "".join(f"{n:>12}" for n in results))
for label, get in rows:
	print(f"{label:<38}" + "".join(f"{get(r):>12,.0f}" for r in results.values()))
