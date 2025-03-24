import chess
import chess.pgn
import numpy as np
import tensorflow as tf
from core.chess_environment import fast_fen_to_example
from tensorflow.keras.models import load_model


def softmax(x):
	"""Compute softmax values for each set of scores in x."""
	exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
	return exp_x / exp_x.sum()


def select_stochastic_move(board, model, temp=1.0):
	"""
	Selects a move probabilistically using softmax over (win-loss)/temp.

	Args:
		board (chess.Board): The current game position.
		model (tf.keras.Model): The model to evaluate positions.
		temp (float): Temperature for softmax exploration (higher = more randomness).

	Returns:
		tuple: (chess.Move, np.array) The selected move and its WDL probabilities.
	"""
	legal_moves = list(board.legal_moves)
	if not legal_moves:
		return None, None  # No legal moves, game over

	best_checkmate_move = None
	position_tensors = []
	move_map = {}

	for move in legal_moves:
		board.push(move)

		# **Immediate Checkmate Handling**
		if board.is_checkmate():
			board.pop()  # Restore board state
			return move, np.array([0.0, 0.0, 1.0])  # Guaranteed win

		# **Move Encoding**
		position_tensors.append(fast_fen_to_example(board.fen()))
		move_map[len(position_tensors) - 1] = move
		board.pop()

	# Convert to batch and get model predictions
	position_tensors = np.array(position_tensors)
	predictions = model.predict(position_tensors, batch_size=len(position_tensors), verbose=0)

	# **Override Model Predictions for Draws**
	for i, move in move_map.items():
		board.push(move)
		if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
			predictions[i] = np.array([0.0, 1.0, 0.0])  # Force draw to be neutral (WDL = 0)
		board.pop()

	# **Compute Softmax Over (Win) / Temperature**
	move_values = (predictions[:, 2]) / temp
	move_probabilities = softmax(move_values)

	# **Select Move Stochastically**
	selected_index = np.random.choice(len(legal_moves), p=move_probabilities)
	return move_map[selected_index], predictions[selected_index]


def self_play_game_stochastic(model_a_path, model_b_path, pgn_save_path, temp=1.0, max_moves=100):
	"""
	Runs a stochastic self-play game between two models and saves the PGN with WDL evaluations.

	Args:
		model_a_path (str): Path to the first model (plays White).
		model_b_path (str): Path to the second model (plays Black).
		pgn_save_path (str): Path to save the PGN file.
		temp (float): Temperature for move randomness.

	Returns:
		str: The final result of the game ("1-0", "0-1", "1/2-1/2").
	"""
	# Load models
	model_a = load_model(model_a_path, compile=False)
	model_b = load_model(model_b_path, compile=False)

	board = chess.Board()
	game = chess.pgn.Game()
	game.headers["White"] = model_a_path
	game.headers["Black"] = model_b_path

	turn_count = 0
	node = game

	while not board.is_game_over():
		current_model = model_a if board.turn == chess.WHITE else model_b
		best_move, wdl = select_stochastic_move(board, current_model, temp=temp)

		if best_move is None:
			break  # No valid move, should not happen in normal play

		board.push(best_move)
		node = node.add_variation(best_move)  # Store move in PGN

		# Format WDL probabilities with 2 decimal places
		l, d, w = map(lambda x: f"{x:.2f}", wdl)
		node.comment = f"WDL: {w}/{d}/{l}"  # Save WDL evaluation

		turn_count += 1

		if turn_count > max_moves * 2:
			break

	# Save the PGN file with WDL evaluations
	with open(pgn_save_path, "w", encoding="utf-8") as pgn_file:
		pgn_file.write(str(game))

	print(f"Game saved to {pgn_save_path}")

	# Return the game result
	return board.result()