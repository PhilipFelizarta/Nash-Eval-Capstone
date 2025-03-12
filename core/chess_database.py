import numpy as np
import chess.pgn
import zstandard as zstd
import io
import sys
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
		with dctx.stream_reader(compressed, read_across_frames=True) as reader:
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
				ply_count = sum(1 for _ in game.mainline_moves())  # Efficient ply counter
				current_ply = 0

				# Iterate through only mainline moves
				for move in game.mainline_moves():
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
		tf.TensorSpec(shape=(None, 8, 8, 35), dtype=tf.float32),  # Input tensor batch
		tf.TensorSpec(shape=(None,), dtype=tf.int32),  # Sparse categorical labels
		tf.TensorSpec(shape=(None,), dtype=tf.float32)  # Importance weights
	)

	dataset = tf.data.Dataset.from_generator(
		lambda: pgn_zst_generator(zst_file, batch_size),
		output_signature=output_signature
	)

	# Parallel loading and prefetching for speedup
	dataset = dataset.shuffle(buffer_size) \
					 .map(lambda x, y, w: (tf.cast(x, tf.float32), y, w),
						  num_parallel_calls=num_parallel_calls) \
					 .prefetch(tf.data.experimental.AUTOTUNE)

	return dataset

def count_ply_and_games(zst_file, sample_count=1):
	"""
	Counts the total number of games and ply count in a .zst PGN file.
	Returns the total number of games, total ply count, and a sample PGN.
	"""
	dctx = zstd.ZstdDecompressor()
	with open(zst_file, "rb") as compressed:
		with dctx.stream_reader(compressed) as reader:
			text_stream = io.TextIOWrapper(reader, encoding="utf-8")
			
			total_games = 0
			total_positions = 0
			sample_games = []
			
			while True:
				game = chess.pgn.read_game(text_stream)
				if game is None:
					break  # End of file
				
				total_games += 1
				ply_count = sum(1 for _ in game.mainline_moves())
				total_positions += ply_count
				
				if len(sample_games) < sample_count:
					sample_games.append(str(game))
					print(str(game))
				
				# Live progress update
				sys.stdout.write(f"\rGames Processed: {total_games}, Positions Processed: {total_positions}")
				sys.stdout.flush()

			return total_games, total_positions, sample_games