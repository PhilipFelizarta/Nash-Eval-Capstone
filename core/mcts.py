"""
Python driver for the C++ PUCT tree search in cpp/mcts (extension module `nash_mcts`).

C++ owns move generation, the (8, 8, 36) input encoding, the tree, selection and backup.
Python only runs the network:

	tree = nash_mcts.MCTS()
	tree.reset(fen, history_uci)
	while tree.root_visits() < n:
		x = tree.gather(batch)                       # (k, 8, 8, 36) leaves that need the network
		policy_logits, wdl = evaluate(x)             # (k, 4672), (k, 3)
		tree.backup(policy_logits, wdl)

Value convention: WDL index 0 = the side to move at the leaf wins. The tree backs up
W + D/2 for that side and flips it (1 - v) at every ply.

Build the extension first: python cpp/mcts/build.py
"""
import os
import sys

import chess
import chess.pgn
import numpy as np

try:
	import nash_mcts
except ImportError:  # Built in place by cpp/mcts/build.py
	sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cpp", "mcts"))
	import nash_mcts

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def make_evaluator(model):
	"""
	Wraps a model into evaluate(batch) -> (policy_logits (k, 4672), wdl (k, 3)), both float32.

	Accepts a Keras model with dict outputs {"policy", "value"} (called as model(x, training=False)),
	or any callable returning a (policy_logits, wdl) tuple or a {"policy", "value"} dict.
	"""
	keras_like = hasattr(model, "predict") and hasattr(model, "layers")

	def evaluate(batch):
		out = model(batch, training=False) if keras_like else model(batch)
		if isinstance(out, dict):
			policy, value = out["policy"], out["value"]
		else:
			policy, value = out
		return np.asarray(policy, dtype=np.float32), np.asarray(value, dtype=np.float32)

	return evaluate


def _root_and_history(board_or_fen, history_uci):
	if isinstance(board_or_fen, chess.Board):
		if history_uci is None:
			history_uci = [move.uci() for move in board_or_fen.move_stack]
			return board_or_fen.root().fen(), history_uci
		return board_or_fen.fen(), list(history_uci)
	return board_or_fen, list(history_uci or [])


def search(tree, evaluate, num_simulations, batch_size=64):
	"""Runs gather -> evaluate -> backup on an already-rooted tree until the root has num_simulations visits."""
	while tree.root_visits() < num_simulations:
		x = tree.gather(min(batch_size, num_simulations - tree.root_visits()))
		if len(x):
			try:
				policy, wdl = evaluate(x)
			except BaseException:
				tree.cancel()
				raise
		else:  # Only terminal leaves this round
			policy, wdl = np.zeros((0, nash_mcts.POLICY_SIZE), np.float32), np.zeros((0, 3), np.float32)
		tree.backup(policy, wdl)
	return tree


def run_mcts(board_or_fen, model, num_simulations=800, batch_size=64, history_uci=None, add_noise=True, tree=None, **mcts_kwargs):
	"""
	Searches a position and returns the nash_mcts.MCTS tree (use root_stats() / select_move()).

	Args:
		board_or_fen: chess.Board (its move_stack becomes the repetition history) or a FEN string.
		model: Keras model with {"policy", "value"} outputs, or a callable evaluate(batch) -> (policy_logits, wdl).
		history_uci: moves already played from the FEN / board position. If None and a board is given,
			the board's root position and move_stack are used.
		add_noise: Dirichlet noise at the root (turn off for evaluation games).
		tree: an existing tree to keep searching (e.g. after advance()); it is not reset.
		mcts_kwargs: c_puct, fpu_reduction, dirichlet_alpha, dirichlet_epsilon, virtual_loss, seed.
	"""
	if tree is None:
		tree = nash_mcts.MCTS(**mcts_kwargs)
		fen, history = _root_and_history(board_or_fen, history_uci)
		tree.reset(fen, history, add_noise)
	return search(tree, make_evaluator(model), num_simulations, batch_size)


def select_mcts_move(board, model, num_simulations=800, temperature=0.0, batch_size=64, add_noise=False, **mcts_kwargs):
	"""
	Returns (chess.Move, root_stats), root_stats = [(uci, policy_index, visits, Q, prior), ...].
	(None, []) if the side to move has no legal moves.
	"""
	if not any(board.legal_moves):
		return None, []
	tree = run_mcts(board, model, num_simulations, batch_size, add_noise=add_noise, **mcts_kwargs)
	return chess.Move.from_uci(tree.select_move(temperature)), tree.root_stats()


class MultiMCTS:
	"""
	Drives several trees at once: their leaves are concatenated into one network batch, which is
	how self-play keeps an accelerator busy. Trees must already be rooted (reset/advance).
	"""

	def __init__(self, trees):
		self.trees = list(trees)

	def search(self, model, num_simulations, batch_size=64):
		evaluate = make_evaluator(model)
		while True:
			active = [t for t in self.trees if t.root_visits() < num_simulations]
			if not active:
				return self.trees
			batches = [t.gather(min(batch_size, num_simulations - t.root_visits())) for t in active]
			sizes = [len(x) for x in batches]
			if sum(sizes):
				try:
					policy, wdl = evaluate(np.concatenate([x for x in batches if len(x)]))
				except BaseException:
					for t in active:
						t.cancel()
					raise
			else:
				policy, wdl = np.zeros((0, nash_mcts.POLICY_SIZE), np.float32), np.zeros((0, 3), np.float32)
			offset = 0
			for tree, k in zip(active, sizes):
				tree.backup(policy[offset:offset + k], wdl[offset:offset + k])
				offset += k


def visit_distribution(tree, drop_proven_losses=True):
	"""
	(moves, policy_indices, visits) at the root: the policy training target. Visits spent on moves
	that the search later proved lose are zeroed (unless every move loses).
	"""
	stats = tree.root_stats()
	visits = np.array([s[2] for s in stats], dtype=np.int64)
	if drop_proven_losses:
		losing = np.array([p == "loss" for p in tree.root_move_proven()], dtype=bool)
		if losing.any() and not losing.all():
			visits[losing] = 0
	return [s[0] for s in stats], np.array([s[1] for s in stats], dtype=np.int64), visits


def _game_result(board):
	"""Result, scoring threefold repetition and the fifty-move rule as draws (as the search does)."""
	if board.is_game_over():
		return board.result()
	if board.is_repetition(3) or board.is_fifty_moves():
		return "1/2-1/2"
	return "*"


def play_game_mcts(white_model, black_model, pgn_save_path=None, num_simulations=200, temperature_moves=15, max_moves=100,
				   starting_fen=STARTING_FEN, batch_size=64, add_noise=True, temperature=1.0, white_name="white",
				   black_name="black", seed=0, **mcts_kwargs):
	"""
	Plays one MCTS game between two (already loaded) models. Each side keeps its own tree and both
	trees are advanced after every move (tree reuse).

	Returns:
		(result, records): result is "1-0", "0-1", "1/2-1/2" or "*" (cut off at max_moves); records has
		one dict per ply with the root FEN, the move played and the root visit distribution.
	"""
	evaluators = {chess.WHITE: make_evaluator(white_model), chess.BLACK: make_evaluator(black_model)}
	trees = {chess.WHITE: nash_mcts.MCTS(seed=seed, **mcts_kwargs), chess.BLACK: nash_mcts.MCTS(seed=seed + 1, **mcts_kwargs)}
	for tree in trees.values():
		tree.reset(starting_fen, [], add_noise)

	board = chess.Board(starting_fen)
	game = chess.pgn.Game()
	game.setup(board)
	game.headers["FEN"] = starting_fen
	game.headers["White"] = white_name
	game.headers["Black"] = black_name

	records = []
	turn_count = 0
	node = game
	while _game_result(board) == "*":
		tree = trees[board.turn]
		search(tree, evaluators[board.turn], num_simulations, batch_size)

		temp = temperature if turn_count < temperature_moves * 2 else 0.0  # temperature_moves is in full moves
		uci = tree.select_move(temp)
		moves, indices, visits = visit_distribution(tree)
		chosen = moves.index(uci)
		records.append({
			"fen": board.fen(),
			"move": uci,
			"moves": moves,
			"policy_indices": indices,
			"visits": visits,
			"root_value": tree.root_value(),  # expected score for the side to move
		})

		move = chess.Move.from_uci(uci)
		board.push(move)
		node = node.add_variation(move)
		node.comment = f"N: {visits[chosen]}/{tree.root_visits()} Q: {tree.root_stats()[chosen][3]:.2f}"
		for t in trees.values():
			t.advance(uci)

		turn_count += 1
		if turn_count > max_moves * 2:
			break

	result = _game_result(board)
	if pgn_save_path is not None:
		from core.self_play import write_game
		write_game(game, result, pgn_save_path)
	return result, records


def self_play_game_mcts(model_a_path, model_b_path, pgn_save_path, num_simulations=200, temperature_moves=15, max_moves=100,
						starting_fen=STARTING_FEN, **kwargs):
	"""
	MCTS self-play between two saved policy+value models (model A plays White). Mirrors
	core.self_play.self_play_game_stochastic: each model is loaded once and the PGN is written with
	its Result header ("*" when cut off at max_moves).

	Returns:
		(result, records): see play_game_mcts; records[i]["visits"] are the policy training targets.
	"""
	from tensorflow.keras.models import load_model

	model_a = load_model(model_a_path, compile=False)
	model_b = load_model(model_b_path, compile=False)
	return play_game_mcts(model_a, model_b, pgn_save_path, num_simulations=num_simulations, temperature_moves=temperature_moves,
						  max_moves=max_moves, starting_fen=starting_fen, white_name=model_a_path, black_name=model_b_path, **kwargs)
