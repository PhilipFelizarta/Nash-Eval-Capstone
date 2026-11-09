import chess
import chess.pgn
import numpy as np
from core.chess_environment import fast_fen_to_example

# Value convention used everywhere in this file:
# the model maps a position to [P(side to move wins), P(draw), P(side to move loses)].
# This matches the training labels (label 0 = side to move went on to win, see chess_database.py).
# Depth-1 search scores positions where the opponent is to move, so for the mover the
# same vector reads [loss, draw, win]. Depth-2 scores positions where the mover is to
# move again, so it reads [win, draw, loss].

DRAW = np.array([0.0, 1.0, 0.0])


def is_drawn(board):
	"""True if the position is a draw by rule (or one the side to move can claim right now)."""
	return board.is_stalemate() or board.is_insufficient_material() or board.is_fifty_moves() or board.is_repetition(3)


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
		tuple: (chess.Move, np.array) The selected move and the model's output for the
		position after it, i.e. [loss, draw, win] from the mover's point of view.
	"""
	legal_moves = list(board.legal_moves)
	if not legal_moves:
		return None, None  # No legal moves, game over

	position_tensors = []
	move_map = {}

	for move in legal_moves:
		board.push(move)

		# **Immediate Checkmate Handling**
		if board.is_checkmate():
			board.pop()  # Restore board state
			return move, np.array([0.0, 0.0, 1.0])  # Side to move (the opponent) has lost

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
		if is_drawn(board):
			predictions[i] = DRAW  # Force draw to be neutral (WDL = 0)
		board.pop()

	# **Compute Softmax Over (Win - Loss) / Temperature, from the mover's point of view**
	move_values = predictions[:, 2] - predictions[:, 0]
	if temp != 0.0:
		move_probabilities = softmax(move_values / temp)

		# **Select Move Stochastically**
		selected_index = np.random.choice(len(legal_moves), p=move_probabilities)
	else:
		selected_index = np.argmax(move_values)

	return move_map[selected_index], predictions[selected_index]


def write_game(game, result, pgn_save_path):
	game.headers["Result"] = result
	with open(pgn_save_path, "w", encoding="utf-8") as pgn_file:
		pgn_file.write(str(game))

	print(f"Game saved to {pgn_save_path}")


def self_play_game_stochastic(model_a_path, model_b_path, pgn_save_path, temp=1.0, max_moves=100, starting_fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"):
	"""
	Runs a stochastic self-play game between two models and saves the PGN with WDL evaluations.

	Args:
		model_a_path (str): Path to the first model (plays White).
		model_b_path (str): Path to the second model (plays Black).
		pgn_save_path (str): Path to save the PGN file.
		temp (float): Temperature for move randomness.

	Returns:
		str: The final result of the game ("1-0", "0-1", "1/2-1/2", or "*" if max_moves was hit).
	"""
	from tensorflow.keras.models import load_model

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
		node.comment = f"WDL: {w}/{d}/{l}"  # Save WDL evaluation (mover's point of view)

		turn_count += 1

		if turn_count > max_moves * 2:
			break

		if wdl[0] > 0.95: # 95% chance the mover loses: the mover resigns.
			resign_color = board.turn # Side to move now is the winner. True.. black resigns
			break

	if resign_color is not None: # resignation
		map_resignation = {True: "1-0", False: "0-1"}
		print("Game Resigned: ", map_resignation[resign_color])
		result = map_resignation[resign_color]
	else:
		result = board.result()

	# Save the PGN file with WDL evaluations
	write_game(game, result, pgn_save_path)
	return result


def select_depth2_move(board, model):
	"""
	Selects the best move using a depth-2 minimax-style search with batched evaluation.

	Args:
		board (chess.Board): Current game state.
		model (tf.keras.Model): Model returning [win, draw, loss] for the side to move.

	Returns:
		tuple: (chess.Move, np.array) Selected move and its worst-case leaf, which is
		[win, draw, loss] from the mover's point of view.
	"""
	def expected_score(p): return 1.0 * p[0] + 0.5 * p[1]  # The mover is to move at every leaf

	legal_moves = list(board.legal_moves)
	if not legal_moves:
		return None, None

	leaf_parents = []  # Our move that leads to each leaf
	leaf_values = []  # Exact value for terminal leaves, None if the model should score it
	position_tensors = []  # Batched evaluation inputs, one per None in leaf_values

	# Build the full depth-2 tree and gather all positions to evaluate
	for move in legal_moves:
		board.push(move)

		# Immediate win
		if board.is_checkmate():
			board.pop()
			return move, np.array([1.0, 0.0, 0.0])

		# Our move ends the game in a draw (stalemate, repetition, ...): it's a leaf
		if is_drawn(board):
			leaf_parents.append(move)
			leaf_values.append(DRAW)
			board.pop()
			continue

		for opp_move in board.legal_moves:
			board.push(opp_move)
			leaf_parents.append(move)

			if board.is_checkmate():
				leaf_values.append(np.array([0.0, 0.0, 1.0]))  # The opponent mates us
			elif is_drawn(board):
				leaf_values.append(DRAW)
			else:
				leaf_values.append(None)
				position_tensors.append(fast_fen_to_example(board.fen()))

			board.pop()

		board.pop()

	# Run predictions in batch (terminal leaves are already scored)
	if position_tensors:
		predictions = iter(model.predict(np.array(position_tensors), batch_size=len(position_tensors), verbose=0))
		leaf_values = [value if value is not None else next(predictions) for value in leaf_values]

	# For each of our moves, assume the opponent picks the reply that is worst for us
	worst_by_move = {}
	for parent_move, value in zip(leaf_parents, leaf_values):
		if parent_move not in worst_by_move or expected_score(value) < expected_score(worst_by_move[parent_move]):
			worst_by_move[parent_move] = value

	# Then pick the move with the best worst case
	best_move = max(worst_by_move, key=lambda move: expected_score(worst_by_move[move]))
	return best_move, worst_by_move[best_move]


def self_play_game_depth2(model_a_path, model_b_path, pgn_save_path, max_moves=100, starting_fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"):
	"""
	Runs a depth-2 self-play game between two models and saves the PGN with WDL evaluations.

	Args:
		model_a_path (str): Path to the first model (plays White).
		model_b_path (str): Path to the second model (plays Black).
		pgn_save_path (str): Path to save the PGN file.

	Returns:
		str: The final result of the game ("1-0", "0-1", "1/2-1/2", or "*" if max_moves was hit).
	"""
	from tensorflow.keras.models import load_model

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
		best_move, wdl = select_depth2_move(board, current_model)

		if best_move is None:
			break  # No valid move, should not happen in normal play

		board.push(best_move)
		node = node.add_variation(best_move)  # Store move in PGN

		# Format WDL probabilities with 2 decimal places
		w, d, l = map(lambda x: f"{x:.2f}", wdl)
		node.comment = f"WDL: {w}/{d}/{l}"  # Save WDL evaluation (mover's point of view)

		turn_count += 1

		if turn_count > max_moves * 2:
			break

		if wdl[-1] > 0.95: # 95% chance the mover loses: the mover resigns.
			resign_color = board.turn # Side to move now is the winner. True.. black resigns
			break

	if resign_color is not None: # resignation
		map_resignation = {True: "1-0", False: "0-1"}
		print("Game Resigned: ", map_resignation[resign_color])
		result = map_resignation[resign_color]
	else:
		result = board.result()

	# Save the PGN file with WDL evaluations
	write_game(game, result, pgn_save_path)
	return result