import numpy as np
import chess
import os
import json
import time
import zstandard as zstd


def fast_fen_to_example(fen):
	"""
	Converts a FEN string into a (8, 8, 17) tensor where:
		- First 6 channels: Current player's pieces
		- Channels 7 & 8: Current player's kingside & queenside castling rights
		- Next 6 channels: Opponent's pieces
		- Channels 15 & 16: Opponent's kingside & queenside castling rights
		- Channel 17 encodes a bit for the en passant square
		- Rotates 180 deg when Black to move.

	Args:
		fen (str): Chess position in FEN notation.

	Returns:
		np.ndarray: (8, 8, 17) tensor representation of the board.
	"""
	tensor = np.zeros((8, 8, 17), dtype=np.float32)

	# Define piece mappings (relative to current player)
	piece_map = {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5}
	opponent_offset = 8  # Opponent pieces stored at index 8+

	# Split FEN into components
	parts = fen.split()
	board_part, turn, castling_rights, en_passant = parts[0], parts[1], parts[2], parts[3]

	# Identify whose turn it is
	is_white_turn = turn == 'w'

	# Parse board
	rows = board_part.split("/")
	for row_idx, row in enumerate(rows):
		col_idx = 0
		for char in row:
			if char.isdigit():
				col_idx += int(char)  # Empty squares
			else:
				piece_index = piece_map.get(char.upper(), None)
				if piece_index is not None:
					if (char.isupper() and is_white_turn) or (char.islower() and not is_white_turn):
						tensor[row_idx, col_idx, piece_index] = 1  # Current player
					else:
						tensor[row_idx, col_idx, piece_index + opponent_offset] = 1  # Opponent
				col_idx += 1

	# Encode castling rights
	tensor[:, :, 6] = 'K' in castling_rights if is_white_turn else 'k' in castling_rights  # Kingside castling
	tensor[:, :, 7] = 'Q' in castling_rights if is_white_turn else 'q' in castling_rights  # Queenside castling
	tensor[:, :, 6 + opponent_offset] = 'k' in castling_rights if is_white_turn else 'K' in castling_rights  # Opponent kingside
	tensor[:, :, 7 + opponent_offset] = 'q' in castling_rights if is_white_turn else 'Q' in castling_rights  # Opponent queenside

	# Encode en passant target square (Channel 14)
	if en_passant != "-":
		ep_square = chess.SQUARE_NAMES.index(en_passant)
		row, col = divmod(ep_square, 8)
		tensor[row, col, 16] = 1  # Mark en passant square

	if not is_white_turn:
		tensor = np.rot90(tensor, k=2, axes=(0, 1))  # Rotates all (8,8) slices

	return tensor

def game_to_batch(game_json):
	"""
	Converts a list of game move data into (input tensor, label) pairs for training.

	Args:
		game_json (list): List of move dictionaries from a game.

	Returns:
		list: List of (tensor, label) tuples for training.
	"""
	examples = []
	labels = []

	for move_data in game_json:
		fen = move_data["fen"]
		label = move_data["label_sparse"]  # Sparse categorical cross-entropy label

		# Convert FEN to tensor
		tensor = fast_fen_to_example(fen)

		# Store the (tensor, label) pair
		examples.append(tensor)
		labels.append(label)

	return examples, labels

def benchmark_game_to_batch(json_path):
	"""
	Measures the execution time of game_to_batch() function.
	"""
	if not os.path.exists(json_path):
		print(f"Error: File {json_path} not found.")
		return

	with open(json_path, "r", encoding="utf-8") as f:
		game_data = json.load(f)

	start_time = time.perf_counter()  
	dataset = game_to_batch(game_data)  
	end_time = time.perf_counter()  

	elapsed_time = end_time - start_time
	print(f"\nProcessing time: {elapsed_time:.6f} seconds")
	print(f"Total moves processed: {len(dataset[0])}")
	print(f"Example tensor shape: {dataset[0][0].shape}")

if __name__ == "__main__":
	json_path = "fen_data/game_0.json"
	benchmark_game_to_batch(json_path)