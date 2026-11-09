"""
Timed MCTS engine for the play UI: the C++ tree (nash_mcts) plus a TF network on the GPU.

Two network kinds:
	- "policy": a policy/value model (exploratory_model(..., policy_head=True)) on the current encoding.
	- "legacy": an old value-only checkpoint (pre-PR #9 encoding). The tree gets its priors from a
	  one-move lookahead: every leaf's children are evaluated with the value net, the children's scores
	  become the policy (softmax(score / prior_temperature)), and the best child's score is the leaf value.
"""
import os
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor

import chess
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "cpp", "mcts"))

import nash_mcts
from core import move_encoding
from core.self_play import is_drawn


def _legacy_encode(fens):
	from play.legacy.chess_environment import fast_fen_to_example
	return np.stack([fast_fen_to_example(f) for f in fens]).astype(np.float32)


class Network:
	def __init__(self, path=None, kind="policy", blocks=6, filters=64):
		import tensorflow as tf
		for gpu in tf.config.list_physical_devices("GPU"):
			tf.config.experimental.set_memory_growth(gpu, True)
		from core import model_framework

		self.kind = kind
		if path:
			model = tf.keras.models.load_model(path, compile=False)
			self.name = os.path.basename(path)
		else:
			model = model_framework.exploratory_model(FILTERS=filters, BLOCKS=blocks, policy_head=(kind == "policy"))
			self.name = f"untrained {blocks}x{filters} ({kind}, placeholder)"
		self.model = model

		@tf.function(input_signature=[tf.TensorSpec([None, 8, 8, 36], tf.float32)])
		def call(x):
			out = model(x, training=False)
			if isinstance(out, dict):
				return tf.cast(out["policy"], tf.float32), tf.cast(out["value"], tf.float32)
			return tf.zeros((tf.shape(x)[0], move_encoding.POLICY_SIZE)), tf.cast(out, tf.float32)

		self._call = call
		self.positions = 0
		self(np.zeros((1, 8, 8, 36), np.float32))  # trace + cuDNN warm-up
		self(np.zeros((256, 8, 8, 36), np.float32))
		self.positions = 0

	def __call__(self, x):
		import tensorflow as tf
		self.positions += len(x)
		policy, value = self._call(tf.constant(x))
		return policy.numpy(), value.numpy()


class Engine:
	def __init__(self, network, batch_size=64, prior_temperature=0.1, workers=8, **mcts_kwargs):
		self.net = network
		self.batch_size = batch_size
		self.prior_temperature = prior_temperature
		self.tree = nash_mcts.MCTS(dirichlet_epsilon=0.0, **mcts_kwargs)
		self.pool = ProcessPoolExecutor(workers) if network.kind == "legacy" else None
		self.lock = threading.Lock()
		self.snapshot = {}

	def new_game(self, fen=chess.STARTING_FEN):
		self.tree.reset(fen, [], False)

	def advance(self, uci):
		self.tree.advance(uci)

	# --- evaluation -------------------------------------------------------------------------------
	def _evaluate(self, x, board):
		if self.net.kind == "policy":
			return self.net(x)
		return self._evaluate_legacy(board)

	def _evaluate_legacy(self, root_board):
		"""One-move lookahead with the value net for every pending leaf (see module docstring)."""
		leaves = []
		for path in self.tree.pending_paths():
			b = root_board.copy(stack=False)
			for uci in path:
				b.push_uci(uci)
			leaves.append(b)

		fens, plan = [], []  # plan: per leaf, list of (policy index, fixed score or row in fens)
		for b in leaves:
			white = b.turn == chess.WHITE
			entries = []
			for move in b.legal_moves:
				b.push(move)
				if b.is_checkmate():
					entries.append((move_encoding.move_to_index(move, white), 1.0))
				elif is_drawn(b):
					entries.append((move_encoding.move_to_index(move, white), 0.5))
				else:
					entries.append((move_encoding.move_to_index(move, white), len(fens)))
					fens.append(b.fen())
				b.pop()
			plan.append(entries)

		wdl_children = np.zeros((0, 3), np.float32)
		if fens:
			chunks = [fens[i:i + 64] for i in range(0, len(fens), 64)]
			x = np.concatenate(list(self.pool.map(_legacy_encode, chunks)))
			_, wdl_children = self.net(x)

		logits = np.full((len(leaves), move_encoding.POLICY_SIZE), -1e9, np.float32)
		wdl = np.zeros((len(leaves), 3), np.float32)
		for i, entries in enumerate(plan):
			scores = []
			for index, ref in entries:
				if isinstance(ref, float):
					s = ref
				else:  # child WDL is from the opponent's view: our score = their loss + half their draw
					c = wdl_children[ref]
					s = float(c[2] + 0.5 * c[1])
				logits[i, index] = s / self.prior_temperature
				scores.append(s)
			best = max(scores)
			wdl[i] = (best, 0.0, 1.0 - best)  # expected score as a W/L split, so W + D/2 = best
		return logits, wdl

	# --- search -----------------------------------------------------------------------------------
	def think(self, board, seconds, on_update=None, update_every=0.15):
		"""Searches for `seconds`, publishing snapshots; returns the chosen move (UCI)."""
		start = time.perf_counter()
		positions0 = self.net.positions
		visits0 = self.tree.root_visits()
		last = 0.0
		while True:
			elapsed = time.perf_counter() - start
			if elapsed >= seconds:
				break
			x = self.tree.gather(self.batch_size)
			if len(x) or self.tree.num_pending():
				policy, wdl = self._evaluate(x, board)
			else:
				policy, wdl = np.zeros((0, move_encoding.POLICY_SIZE), np.float32), np.zeros((0, 3), np.float32)
			self.tree.backup(policy, wdl)
			if self.tree.root_proven() != "unknown" and elapsed > 0.5:
				break  # forced result found
			if elapsed - last >= update_every:
				last = elapsed
				self._publish(board, start, positions0, visits0, on_update)
		snap = self._publish(board, start, positions0, visits0, on_update, final=True)
		return self.tree.select_move(0.0) if snap["moves"] else None

	def _publish(self, board, start, positions0, visits0, on_update, final=False):
		elapsed = time.perf_counter() - start
		stats = self.tree.root_stats()
		visits_total = sum(s[2] for s in stats) or 1
		moves = []
		for uci, index, visits, q, prior in sorted(stats, key=lambda s: (-s[2], -s[4])):
			move = chess.Move.from_uci(uci)
			moves.append({
				"uci": uci, "san": board.san(move), "visits": visits, "share": visits / visits_total,
				"q": None if q != q else q, "prior": float(prior),
			})
		pv_san, b = [], board.copy(stack=False)
		for uci in self.tree.principal_variation(12):
			move = chess.Move.from_uci(uci)
			pv_san.append(b.san(move))
			b.push(move)
		new_visits = self.tree.root_visits() - visits0
		snap = {
			"elapsed": elapsed,
			"root_visits": self.tree.root_visits(),
			"new_visits": new_visits,
			"nodes_per_second": new_visits / elapsed if elapsed > 0 else 0.0,
			"net_positions_per_second": (self.net.positions - positions0) / elapsed if elapsed > 0 else 0.0,
			"tree_nodes": self.tree.num_nodes(),
			"max_depth": self.tree.max_depth(),
			"root_value": self.tree.root_value(),  # expected score for the engine (side to move)
			"proven": self.tree.root_proven(),
			"moves": moves[:20],
			"pv": self.tree.principal_variation(12),
			"pv_san": pv_san,
			"best": moves[0]["uci"] if moves else None,
			"final": final,
		}
		with self.lock:
			self.snapshot = snap
		if on_update:
			on_update(snap)
		return snap
