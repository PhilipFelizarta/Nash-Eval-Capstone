import chess
import chess.pgn
import re
import json
import zstandard as zstd
import os
from tqdm import tqdm
import io

def extract_game_result(headers):
	result =  headers.get("Result", "*")

	if result == "1-0": # White win
		return 1
	elif result == "0-1": # Black win
		return -1
	elif result =="1/2-1/2": # Draw
		return 0
	
	return 2 # Incomplete game

def process_pgn_stream(file_obj, output_folder, max_games):
	"""Processes a PGN stream from a file-like object and saves up to `max_games` as JSON."""
	os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists

	text_stream = io.TextIOWrapper(file_obj, encoding="utf-8")

	for game_index in tqdm(range(max_games), desc="Processing Games"):
		game = chess.pgn.read_game(text_stream)
		if game is None:
			print("End of PGN file reached.")
			break  # No more games available

		board = game.board()
		move_number = 1
		node = game
		game_result = extract_game_result(game.headers)

		game_data = []  # Store positions for this game

		while node.variations:
			next_node = node.variation(0)
			move = next_node.move

			# Extract metadata
			fen = board.fen()
			move_uci = move.uci()
			# Save position details
			example = {
				"move_number": move_number,
				"fen": fen,
				"move": move_uci,
				"game_result": game_result
			}
			game_data.append(example)

			# Make the move on the board
			board.push(move)
			move_number += 1

			# Advance to the next node
			node = next_node

		# Save each game to a separate JSON file
		json_filename = os.path.join(output_folder, f"game_{game_index}.json")
		with open(json_filename, "w", encoding="utf-8") as f:
			json.dump(game_data, f, indent=4)

	print(f"Saved {game_index + 1} games to {output_folder}")

def pgn_zst_to_json(zst_file, output_folder, max_games=1000):
	"""Reads a .pgn.zst file, extracts PGNs, and saves each game as a separate JSON file."""
	dctx = zstd.ZstdDecompressor()

	with open(zst_file, "rb") as compressed:
		with dctx.stream_reader(compressed) as reader:
			process_pgn_stream(reader, output_folder, max_games)

# Example usage:
pgn_zst_to_json("data/lichess_db_standard_rated_2013-01.pgn.zst", "fen_data/", max_games=1)