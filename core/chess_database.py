import numpy as np
import chess.pgn
import zstandard as zstd
import io
import tensorflow as tf
import core.chess_environment as chess_env

GAMMA = 0.99 # Discount factor for importance sampling.

def extract_game_result(headers):
	"""Extracts game result from PGN headers and avoids incomplete games."""
	result = headers.get("Result", "*")
	if result == "1-0":
		return 1  # White wins
	elif result == "0-1":
		return -1  # Black wins
	elif result == "1/2-1/2":
		return 0  # Draw
	return None  # Ignore incomplete games

def pgn_zst_generator(zst_file, batch_size=32):
	"""
	Streams PGN data from a .zst file using a single-threaded generator.
	"""
	dctx = zstd.ZstdDecompressor()
	with open(zst_file, "rb") as compressed:
		with dctx.stream_reader(compressed) as reader:
			text_stream = io.TextIOWrapper(reader, encoding="utf-8")

			game_data = []
			labels = []
			importance_weights = []

			while True:
				game = chess.pgn.read_game(text_stream)
				if game is None:
					break  # End of file

				board = game.board()
				node = game
				game_result = extract_game_result(game.headers)

				if game_result is None:  # Skip incomplete games
					continue  

				# Compute total number of plies in the game
				ply_count = 0
				temp_node = game
				while temp_node.variations:
					temp_node = temp_node.variation(0)
					ply_count += 1

				current_ply = 0

				while node.variations:
					next_node = node.variation(0)
					move = next_node.move
					fen = board.fen()
					turn = board.turn
					relative_modifier = -1 if turn else 1

					# Convert FEN to tensor
					tensor = chess_env.fast_fen_to_example(fen)

					# Importance weight based on distance to final move
					importance = GAMMA ** (ply_count - current_ply)

					# Store in batch
					game_data.append(tensor)
					labels.append(int((game_result * relative_modifier) + 1))  
					importance_weights.append(importance)

					# Move forward
					board.push(move)
					node = next_node
					current_ply += 1

					# Yield batch when ready
					if len(game_data) >= batch_size:
						yield np.array(game_data), np.array(labels), np.array(importance_weights)
						game_data, labels, importance_weights = [], [], []

			# Yield remaining data
			if game_data:
				yield np.array(game_data), np.array(labels), np.array(importance_weights)


def get_tf_dataset(zst_file, batch_size=32, num_parallel_calls=tf.data.experimental.AUTOTUNE, buffer_size=128):
	"""
	Returns a TensorFlow Dataset streaming from a multi-threaded PGN generator.
	"""
	output_signature = (
		tf.TensorSpec(shape=(None, 8, 8, 17), dtype=tf.float32),  # Input tensor batch
		tf.TensorSpec(shape=(None,), dtype=tf.int32),  # Sparse categorical labels
		tf.TensorSpec(shape=(None,), dtype=tf.float32)  # Importance weights
	)

	dataset = tf.data.Dataset.from_generator(
		lambda: pgn_zst_generator(zst_file, batch_size),
		output_signature=output_signature
	)

	# Parallel loading and prefetching for speedup
	dataset = dataset.shuffle(buffer_size) \
					 .map(lambda x, y, w: (tf.py_function(lambda a: a, [x], tf.float32), y, w),
						  num_parallel_calls=num_parallel_calls) \
					 .prefetch(tf.data.experimental.AUTOTUNE)

	return dataset