import numpy as np
import chess
import os
import json
import time
import zstandard as zstd
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.colors import ListedColormap

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

def tensor_to_fen(tensor, is_white_turn=True):
	"""
	Reconstructs a FEN string from a (8,8,17) tensor.
	Args:
		tensor (np.ndarray): (8, 8, 17) tensor representation of the board.
		is_white_turn (bool): Whether the current player to move is White.

	Returns:
		str: Reconstructed FEN string.
	"""
	# Reverse rotation if it's black's turn (since it was rotated in `fast_fen_to_example`)
	if not is_white_turn:
		tensor = np.rot90(tensor, k=2, axes=(0, 1))  

	# Define piece mapping (reverse of `fast_fen_to_example`)
	piece_map = {0: "P", 1: "N", 2: "B", 3: "R", 4: "Q", 5: "K"}
	opponent_offset = 8

	# Convert tensor to FEN board representation
	fen_rows = []
	for row in range(8):
		fen_row = ""
		empty_count = 0
		for col in range(8):
			# Check which piece is present
			piece_found = False
			for i in range(6):  # Check player's pieces
				if tensor[row, col, i] == 1:
					if empty_count > 0:
						fen_row += str(empty_count)  # Add accumulated empty squares
						empty_count = 0
					fen_row += piece_map[i] if is_white_turn else piece_map[i].lower()
					piece_found = True
					break
			for i in range(8, 14):  # Check opponent's pieces
				if tensor[row, col, i] == 1:
					if empty_count > 0:
						fen_row += str(empty_count)
						empty_count = 0
					fen_row += piece_map[i - opponent_offset].lower() if is_white_turn else piece_map[i - opponent_offset]
					piece_found = True
					break
			if not piece_found:
				empty_count += 1
		if empty_count > 0:
			fen_row += str(empty_count)  # Append last empty squares count
		fen_rows.append(fen_row)

	# Castling rights
	castling = ""
	if tensor[0, :, 6].any():  # White kingside
		castling += "K"
	if tensor[0, :, 7].any():  # White queenside
		castling += "Q"
	if tensor[0, :, 14].any():  # Black kingside
		castling += "k"
	if tensor[0, :, 15].any():  # Black queenside
		castling += "q"
	castling = castling if castling else "-"

	# En passant target square
	ep_square = "-"
	for row in range(8):
		for col in range(8):
			if tensor[row, col, 16] == 1:
				ep_square = chess.square_name((7 - row) * 8 + col)  # Adjust for black-to-move rotation

	# Construct final FEN string
	fen = "/".join(fen_rows) + f" {'w' if is_white_turn else 'b'} {castling} {ep_square} 0 1"
	return fen

# Unicode dictionary for chess pieces
UNICODE_PIECES = {
	'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚', 'p': '♟',
	'R': '♜', 'N': '♞', 'B': '♝', 'Q': '♛', 'K': '♚', 'P': '♟'
}

def draw_chessboard(ax):
	"""Draws an 8x8 chessboard using Matplotlib with egg-colored light squares and brown dark squares."""
	
	# Define custom colors for squares
	light_square_color = "#FAEBD7"  # Egg-like color (AntiqueWhite)
	dark_square_color = "#8B4513"   # Brown (SaddleBrown)

	# Create custom colormap
	custom_cmap = ListedColormap([dark_square_color, light_square_color])

	# Generate board pattern
	board_colors = np.array([[1 if (i + j) % 2 == 0 else 0 for j in range(8)] for i in range(8)])
	
	# Draw Chessboard
	ax.imshow(board_colors, cmap=custom_cmap, extent=[0, 8, 0, 8])
	
	# Draw grid lines
	for i in range(9):
		ax.plot([i, i], [0, 8], color='black', linewidth=1)
		ax.plot([0, 8], [i, i], color='black', linewidth=1)

	ax.set_xticks([])
	ax.set_yticks([])

def draw_pieces(ax, board):
	"""Overlays Unicode chess pieces on the board with proper contrast."""
	for square in chess.SQUARES:
		piece = board.piece_at(square)
		if piece:
			col, row = chess.square_file(square), 7 - chess.square_rank(square)
			piece_symbol = UNICODE_PIECES[piece.symbol()]

			# Determine if square is light or dark
			is_light_square = (row + col) % 2 == 0
			is_white_piece = piece.color == chess.WHITE

			# Set text color and outline
			text_color = 'black' if is_white_piece else 'white'
			outline_color = 'white' if is_white_piece else 'black'  # White outline for black pieces, black for white

			# Apply stroke effect for visibility
			path_effects_style = [
				path_effects.Stroke(linewidth=1, foreground=outline_color),  # Thicker outline
				path_effects.Normal()
			]

			# Draw the piece
			ax.text(
				col + 0.5, row + 0.5, piece_symbol,
				fontsize=32, ha='center', va='center', color=text_color,
				path_effects=path_effects_style
			)

def visualize_tensor_as_board(tensor, is_white_turn=True):
	"""Converts tensor into a chessboard and visualizes it."""
	fen = tensor_to_fen(tensor, is_white_turn)  # Convert tensor back to FEN
	board = chess.Board(fen)

	fig, ax = plt.subplots(figsize=(6, 6))
	draw_chessboard(ax)  # Draw board
	draw_pieces(ax, board)  # Draw pieces

	plt.show()
	
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