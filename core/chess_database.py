import random
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

def game_positions(game):
	"""
	Yields (fen, label, importance) for every position before a move in a finished game.

	label is from the side to move's point of view: 0 = went on to win, 1 = draw, 2 = lost.
	importance weights positions near the end of the game more heavily.
	"""
	game_result = extract_game_result(game.headers)
	if game_result is None:  # Skip incomplete games
		return

	moves = list(game.mainline_moves())
	ply_count = len(moves)
	board = game.board()

	for current_ply, move in enumerate(moves):
		relative_modifier = -1 if board.turn else 1
		yield board.fen(), int((game_result * relative_modifier) + 1), GAMMA ** (ply_count - current_ply)
		board.push(move)

def encode_batch(samples):
	"""Encodes a list of (fen, label, importance) samples into model inputs."""
	fens, labels, importance_weights = zip(*samples)
	tensors = np.array([chess_env.fast_fen_to_example(fen) for fen in fens])
	return tensors, np.array(labels), np.array(importance_weights, dtype=np.float32)

def pgn_zst_generator(zst_file, batch_size=32, shuffle_buffer=200_000, skip_games=0, seed=None, log_every=100_000):
	"""
	Streams PGN data from a .zst file using a single-threaded generator.

	Positions (not batches) go through a shuffle buffer before batching, so a batch mixes
	positions from thousands of games instead of being consecutive plies of one game.
	The buffer holds FENs and is only encoded at batch time, so 200k positions is ~50 MB.

	skip_games skips that many games from the start of the file, so a resumed run can
	continue on games the previous run has not seen.
	"""
	rng = random.Random(seed)
	buffer = []
	batch = []

	dctx = zstd.ZstdDecompressor()
	with open(zst_file, "rb") as compressed:
		with dctx.stream_reader(compressed, read_across_frames=True) as reader:
			text_stream = io.TextIOWrapper(reader, encoding="utf-8")

			for _ in range(skip_games):
				if not chess.pgn.skip_game(text_stream):
					break  # End of file

			games_read = skip_games
			while True:
				game = chess.pgn.read_game(text_stream)
				if game is None:
					break  # End of file

				games_read += 1
				if games_read % log_every == 0:
					print(f"\n[data] games consumed (use as SKIP_GAMES to resume past them): {games_read}")

				for sample in game_positions(game):
					if len(buffer) < shuffle_buffer:
						buffer.append(sample)
						continue

					# Swap a random buffered position out for the new one
					i = rng.randrange(shuffle_buffer)
					batch.append(buffer[i])
					buffer[i] = sample

					# Yield batch when ready
					if len(batch) >= batch_size:
						yield encode_batch(batch)
						batch = []

	# Yield remaining data
	rng.shuffle(buffer)
	batch.extend(buffer)
	for start in range(0, len(batch), batch_size):
		yield encode_batch(batch[start:start + batch_size])


def get_tf_dataset(zst_file, batch_size=32, shuffle_buffer=200_000, skip_games=0, num_parallel_calls=tf.data.experimental.AUTOTUNE):
	"""
	Returns a TensorFlow Dataset streaming shuffled batches from a PGN generator.
	"""
	output_signature = (
		tf.TensorSpec(shape=(None, 8, 8, 36), dtype=tf.float32),  # Input tensor batch
		tf.TensorSpec(shape=(None,), dtype=tf.int32),  # Sparse categorical labels
		tf.TensorSpec(shape=(None,), dtype=tf.float32)  # Importance weights
	)

	dataset = tf.data.Dataset.from_generator(
		lambda: pgn_zst_generator(zst_file, batch_size, shuffle_buffer=shuffle_buffer, skip_games=skip_games),
		output_signature=output_signature
	)

	# Shuffling happens per position inside the generator; here we only cast and prefetch
	dataset = dataset.map(lambda x, y, w: (tf.cast(x, tf.float32), y, w),
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
		
	
def load_games_reversed(pgn_path):
	"""Loads and returns a list of PGN game strings in reverse order."""
	with open(pgn_path, "r", encoding="utf-8") as f:
		text = f.read()

	# Robust split: look for games starting with a header line `[Event`
	raw_games = text.strip().split("\n[Event ")
	games = [raw_games[0]] + [f"[Event {g}" for g in raw_games[1:]] if len(raw_games) > 1 else raw_games
	games = [g.strip() for g in games if g.strip()]

	print(f"🔎 Found {len(games)} games in PGN.")
	return games[::-1]  # Reverse for bottom-up streaming

def reversed_pgn_generator(pgn_file, batch_size=32, max_batches=100000):	
	games = load_games_reversed(pgn_file)

	game_data, labels, importance_weights = [], [], []
	batch_count = 0

	for g in games:
		game = chess.pgn.read_game(io.StringIO(g))
		if game is None:
			continue

		for fen, label, importance in game_positions(game):
			game_data.append(chess_env.fast_fen_to_example(fen))
			labels.append(label)
			importance_weights.append(importance)

			if len(game_data) >= batch_size:
				yield np.array(game_data), np.array(labels), np.array(importance_weights)
				game_data, labels, importance_weights = [], [], []
				batch_count += 1
				if batch_count >= max_batches:
					return

	# Yield any remaining data
	if game_data and batch_count < max_batches:
		yield np.array(game_data), np.array(labels), np.array(importance_weights)

def get_tf_testset_from_end(pgn_file, batch_size=32, max_batches=100000):
	output_signature = (
		tf.TensorSpec(shape=(None, 8, 8, 36), dtype=tf.float32),
		tf.TensorSpec(shape=(None,), dtype=tf.int32),
		tf.TensorSpec(shape=(None,), dtype=tf.float32)
	)

	dataset = tf.data.Dataset.from_generator(
		lambda: reversed_pgn_generator(pgn_file, batch_size, max_batches),
		output_signature=output_signature
	)

	return dataset.prefetch(tf.data.AUTOTUNE)