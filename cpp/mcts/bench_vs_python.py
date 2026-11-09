"""python cpp/mcts/bench_vs_python.py

Nodes/second, Python vs C++, with a zero-cost stub network so only search overhead is measured.

nodes = positions encoded and handed to the network (MCTS: one per simulation; depth-1/2: every leaf).
"""
import math, os, sys, time
import numpy as np
import chess

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "cpp", "mcts"))
import nash_mcts
from core import self_play, move_encoding
from core.chess_environment import fast_fen_to_example

POSITIONS = {
	"start": chess.STARTING_FEN,
	"middlegame": "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
	"endgame": "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
}
UNIFORM_POLICY = np.zeros(move_encoding.POLICY_SIZE, np.float32)
DRAW = np.float32([0.0, 1.0, 0.0])


class Stub:
	"""Keras-like stub for the old search: constant WDL, no cost."""
	def __init__(self): self.positions = 0
	def predict(self, x, batch_size=None, verbose=0):
		self.positions += len(x)
		return np.tile(DRAW, (len(x), 1))


def timed(fn, min_time=3.0):
	runs, start = 0, time.perf_counter()
	while True:
		fn(); runs += 1
		elapsed = time.perf_counter() - start
		if elapsed >= min_time: return runs, elapsed


def bench_old(select, fen):
	stub = Stub()
	runs, elapsed = timed(lambda: select(chess.Board(fen), stub))
	return stub.positions / elapsed


# Pure-Python PUCT with the same rules as nash_mcts: python-chess move generation, fast_fen_to_example
# encoding, move_encoding priors, FPU, value = W + D/2 flipped each ply. Batch size 1.
class Node:
	__slots__ = ("move", "parent", "children", "prior", "n", "w", "terminal")
	def __init__(self, move, parent, prior):
		self.move, self.parent, self.prior = move, parent, prior
		self.children, self.n, self.w, self.terminal = None, 0, 0.0, None


def py_mcts(fen, sims, c_puct=2.0, fpu_reduction=0.25):
	board = chess.Board(fen)
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
				if score > best_score: best, best_score = c, score
			board.push(best.move); pushed += 1
			node = best
		if node.terminal is not None:
			v = node.terminal
		elif board.is_checkmate():
			node.terminal = v = 0.0
		elif self_play.is_drawn(board):
			node.terminal = v = 0.5
		else:
			x = fast_fen_to_example(board.fen())[None]  # network input
			logits, wdl = UNIFORM_POLICY, DRAW           # stub network
			moves, idx = move_encoding.legal_move_indices(board)
			priors = move_encoding.masked_policy(logits, idx)
			node.children = [Node(m, node, p) for m, p in zip(moves, priors)]
			v = float(wdl[0] + 0.5 * wdl[1])
		val = 1.0 - v
		while node is not None:
			node.n += 1; node.w += val; val = 1.0 - val
			node = node.parent
		for _ in range(pushed): board.pop()
	return root


def bench_py_mcts(fen, sims=1500):
	runs, elapsed = timed(lambda: py_mcts(fen, sims))
	return runs * sims / elapsed


def cpp_stub(x):
	return np.zeros((len(x), nash_mcts.POLICY_SIZE), np.float32), np.tile(DRAW, (len(x), 1))


def bench_cpp(fen, batch, sims=20000):
	def run():
		tree = nash_mcts.MCTS(seed=0); tree.reset(fen, [], False)
		while tree.root_visits() < sims:
			x = tree.gather(min(batch, sims - tree.root_visits()))
			tree.backup(*cpp_stub(x))
	runs, elapsed = timed(run)
	return runs * sims / elapsed


rows = []
for name, fen in POSITIONS.items():
	r = {
		"py_d1": bench_old(self_play.select_stochastic_move, fen),
		"py_d2": bench_old(self_play.select_depth2_move, fen),
		"py_mcts": bench_py_mcts(fen),
		"cpp_1": bench_cpp(fen, 1),
		"cpp_64": bench_cpp(fen, 64),
	}
	rows.append((name, r))
	print(name, {k: round(v) for k, v in r.items()}, flush=True)

print()
print(f"{'position':<11}{'Py depth-1':>11}{'Py depth-2':>11}{'Py MCTS':>9}{'C++ MCTS b1':>13}{'C++ MCTS b64':>14}{'C++/Py MCTS':>13}")
for name, r in rows:
	print(f"{name:<11}{r['py_d1']:>11,.0f}{r['py_d2']:>11,.0f}{r['py_mcts']:>9,.0f}{r['cpp_1']:>13,.0f}{r['cpp_64']:>14,.0f}{r['cpp_64'] / r['py_mcts']:>12,.0f}x")
