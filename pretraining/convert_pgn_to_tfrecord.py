import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import tensorflow as tf
import chess.pgn
import zstandard as zstd
import io
import core.chess_environment as chess_env

GAMMA = 0.99  # Discount factor for importance sampling.

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

def serialize_example(tensor, label, weight):
	"""
	Creates a serialized TensorFlow Example for TFRecord.
	"""

	tensor = tensor.astype(np.uint8)  # Ensure it's stored efficiently
	feature = {
		'tensor': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tensor.tobytes()])),
		'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
		'weight': tf.train.Feature(float_list=tf.train.FloatList(value=[weight]))
	}
	example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
	return example_proto.SerializeToString()

def convert_pgn_to_tfrecord(zst_file, tfrecord_file):
	"""
	Reads a .pgn.zst file, extracts features, and writes to a TFRecord file.
	"""
	dctx = zstd.ZstdDecompressor()
	with open(zst_file, "rb") as compressed:
		with dctx.stream_reader(compressed) as reader, tf.io.TFRecordWriter(tfrecord_file) as writer:
			text_stream = io.TextIOWrapper(reader, encoding="utf-8")

			total_games = 0

			while True:
				game = chess.pgn.read_game(text_stream)
				if game is None:
					break  # End of file

				board = game.board()
				game_result = extract_game_result(game.headers)
				if game_result is None:
					continue  # Skip incomplete games

				ply_count = sum(1 for _ in game.mainline_moves())  # Total plies in the game
				current_ply = 0

				for move in game.mainline_moves():
					fen = board.fen()
					turn = board.turn
					relative_modifier = -1 if turn else 1

					tensor = chess_env.fast_fen_to_example(fen)  # Convert FEN to tensor
					importance = GAMMA ** (ply_count - current_ply)

					label = int((game_result * relative_modifier) + 1)
					
					# Serialize and write example
					writer.write(serialize_example(tensor, label, importance))
					

					board.push(move)
					current_ply += 1
				
				# Live progress update
				total_games += 1
				sys.stdout.write(f"\rGames Processed: {total_games}")
				sys.stdout.flush()

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description="Convert PGN .zst to TFRecord")
	parser.add_argument("input_pgn_zst", type=str, help="Path to .pgn.zst file")
	parser.add_argument("output_tfrecord", type=str, help="Output TFRecord file")
	
	args = parser.parse_args()
	convert_pgn_to_tfrecord(args.input_pgn_zst, args.output_tfrecord)