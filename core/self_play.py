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
	if temp != 0.0:
		move_values = (predictions[:, 2] - predictions[:, 0]) / temp
		move_probabilities = softmax(move_values)

		# **Select Move Stochastically**
		selected_index = np.random.choice(len(legal_moves), p=move_probabilities)
	else:
		selected_index = np.argmax(predictions[:, 2] - predictions[:, 0])

	return move_map[selected_index], predictions[selected_index]


def self_play_game_stochastic(model_a_path, model_b_path, pgn_save_path, temp=1.0, max_moves=100, starting_fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"):
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

	board = chess.Board(starting_fen)
	game = chess.pgn.Game()
	game.setup(board)
	game.headers["FEN"] = starting_fen
	game.headers["White"] = model_a_path
	game.headers["Black"] = model_b_path

	turn_count = 0
	node = game

	resign_color = None
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

		if wdl[0] > 0.95: # 95% chance of loss resign.
			resign_color = board.turn # True.. black resigns
			break

	# Save the PGN file with WDL evaluations
	with open(pgn_save_path, "w", encoding="utf-8") as pgn_file:
		pgn_file.write(str(game))

	print(f"Game saved to {pgn_save_path}")

	if resign_color is not None: # resignation
		map_resignation = {True: "1-0", False: "0-1"}
		print("Game Resigned: ", map_resignation[resign_color])
		return map_resignation[resign_color]

	# Return the game result
	return board.result()

def select_depth2_move(board, model):
	"""
	Selects the best move using a depth-2 minimax-style search with batched evaluation.

	Args:
		board (chess.Board): Current game state.
		model (tf.keras.Model): Model returning [loss, draw, win] from current player's POV.

	Returns:
		tuple: (chess.Move, np.array) Selected move and its estimated WDL.
	"""
	def expected_score(p): return 0.5 * p[1] + 1.0 * p[2]

	legal_moves = list(board.legal_moves)
	if not legal_moves:
		return None, None

	move_map = {}  # Maps move index to (parent move, child move)
	position_tensors = []  # Batched evaluation inputs

	# Build the full depth-2 tree and gather all positions to evaluate
	for i, move in enumerate(legal_moves):
		board.push(move)

		# Check for immediate win
		if board.is_checkmate():
			board.pop()
			return move, np.array([0.0, 0.0, 1.0])  # Immediate win

		opponent_moves = list(board.legal_moves)

		# If opponent has no move, it's a terminal state
		if not opponent_moves:
			if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
				position_tensors.append(None)  # Handle separately
				move_map[len(position_tensors) - 1] = (move, None)
			else:
				# Opponent is checkmated, this move is winning
				board.pop()
				return move, np.array([0.0, 0.0, 1.0])

		for opp_move in opponent_moves:
			board.push(opp_move)
			fen = board.fen()
			position_tensors.append(fast_fen_to_example(fen))
			move_map[len(position_tensors) - 1] = (move, opp_move)
			board.pop()

		board.pop()

	# Run predictions in batch (excluding special cases)
	valid_indices = [i for i, x in enumerate(position_tensors) if x is not None]
	batched_input = np.array([position_tensors[i] for i in valid_indices])
	predictions = model.predict(batched_input, batch_size=len(batched_input), verbose=0)

	# Insert predictions back into the full list
	full_predictions = []
	j = 0
	for i in range(len(position_tensors)):
		if position_tensors[i] is None:
			full_predictions.append(np.array([0.0, 1.0, 0.0]))  # Draw by rule
		else:
			# Handle draw-like states (stalemate, insufficient material, etc.)
			move = move_map[i][1]
			board.push(move)
			if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
				full_predictions.append(np.array([0.0, 1.0, 0.0]))
			else:
				full_predictions.append(predictions[j])
			board.pop()
			j += 1

	# Aggregate: for each player move, find worst-case opponent response
	from collections import defaultdict

	move_to_evals = defaultdict(list)
	for i, (parent_move, _) in move_map.items():
		move_to_evals[parent_move].append(full_predictions[i])

	# Minimize our win probability (opponent plays best), then select the move with max worst-case win
	best_move = None
	best_eval = None

	for move, evals in move_to_evals.items():
		worst = min(evals, key=lambda p: expected_score(p))  # Opponent minimizes our win
		if best_eval is None or expected_score(worst) > expected_score(best_eval):
			best_move = move
			best_eval = worst

	return best_move, best_eval

def self_play_game_depth2(model_a_path, model_b_path, pgn_save_path, max_moves=100, starting_fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"):
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

	board = chess.Board(starting_fen)
	game = chess.pgn.Game()
	game.setup(board)
	game.headers["FEN"] = starting_fen
	game.headers["White"] = model_a_path
	game.headers["Black"] = model_b_path

	turn_count = 0
	node = game

	resign_color = None
	while not board.is_game_over():
		current_model = model_a if board.turn == chess.WHITE else model_b
		best_move, ldw = select_depth2_move(board, current_model)

		if best_move is None:
			break  # No valid move, should not happen in normal play

		board.push(best_move)
		node = node.add_variation(best_move)  # Store move in PGN

		# Format WDL probabilities with 2 decimal places
		l, d, w = map(lambda x: f"{x:.2f}", ldw)
		node.comment = f"WDL: {w}/{d}/{l}"  # Save WDL evaluation

		turn_count += 1

		if turn_count > max_moves * 2:
			break

		if ldw[0] > 0.95: # 95% chance of loss resign.
			resign_color = board.turn # True.. black resigns
			break

	# Save the PGN file with WDL evaluations
	with open(pgn_save_path, "w", encoding="utf-8") as pgn_file:
		pgn_file.write(str(game))

	print(f"Game saved to {pgn_save_path}")

	if resign_color is not None: # resignation
		map_resignation = {True: "1-0", False: "0-1"}
		print("Game Resigned: ", map_resignation[resign_color])
		return map_resignation[resign_color]

	# Return the game result
	return board.result()