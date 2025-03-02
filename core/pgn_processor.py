import chess
import chess.pgn
import re
import json

def extract_evaluation(comment):
	"""Extracts evaluation score from a PGN comment."""
	eval_match = re.search(r"\[%eval (-?\d+(\.\d+)?)\]", comment)
	if eval_match:
		return float(eval_match.group(1))
	return None

def extract_clock_time(comment):
	"""Extracts clock time from a PGN comment."""
	clock_match = re.search(r"\[%clk (\d+):(\d+):(\d+)\]", comment)
	if clock_match:
		hours, minutes, seconds = map(int, clock_match.groups())
		return hours * 3600 + minutes * 60 + seconds
	clock_match = re.search(r"\[%clk (\d+):(\d+)\]", comment)  # Format mm:ss
	if clock_match:
		minutes, seconds = map(int, clock_match.groups())
		return minutes * 60 + seconds
	return None

def process_pgn(pgn_file):
	"""Processes a PGN file and extracts training examples from each game."""
	examples = []
	with open(pgn_file, 'r', encoding='utf-8') as file:
		game = chess.pgn.read_game(file)

		while game:
			board = game.board()
			move_number = 1
			node = game

			while node.variations:
				next_node = node.variation(0)
				move = next_node.move

				# Extract metadata
				fen = board.fen()
				move_uci = move.uci()
				eval_score = extract_evaluation(next_node.comment)
				clock_time = extract_clock_time(next_node.comment)

				# Create an example
				example = {
					"move_number": move_number,
					"fen": fen,
					"move": move_uci,
					"evaluation": eval_score,
					"clock_time": clock_time
				}
				examples.append(example)

				# Make the move on the board
				board.push(move)
				move_number += 1

				# Advance to the next node
				node = next_node

			# Read the next game in the PGN file
			game = chess.pgn.read_game(file)

	return examples

def pgn_to_json(pgn_file, json_name):
	examples = process_pgn(pgn_file)
	
	# Save as JSON for easy loading
	with open(f"{json_name}.json", "w", encoding="utf-8") as f:
		json.dump(examples, f, indent=4)

	print(f"Processed {len(examples)} positions and saved to chess_data.json")