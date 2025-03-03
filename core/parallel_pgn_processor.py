import chess.pgn
import json
import zstandard as zstd
import os
import multiprocessing as mp
import io
import logging
from time import time
from tqdm import tqdm

# Set up logging
logging.basicConfig(
	format="%(asctime)s [%(levelname)s] %(message)s",
	level=logging.INFO,
	handlers=[
		logging.FileHandler("parallel_pgn_processor.log"),
		logging.StreamHandler()
	]
)

def extract_game_result(headers):
	"""Extracts game result from PGN headers."""
	result = headers.get("Result", "*")
	if result == "1-0": return 1  # White wins
	elif result == "0-1": return -1  # Black wins
	elif result == "1/2-1/2": return 0  # Draw
	return 2  # Incomplete game

def find_next_game_start(text_stream):
	"""Streams through lines until the next '[Event' tag is found."""
	for line in text_stream:
		if line.startswith("[Event "):
			return True  # Found a valid game start
	return False  # Reached end of file without finding a game

def process_chunk(zst_file, start_pos, end_pos, worker_id, output_folder):
	"""Worker process that reads a portion of the .pgn.zst file, ensuring correct game boundaries."""
	dctx = zstd.ZstdDecompressor()
	logging.info(f"Worker {worker_id}: Seeking valid start position from approximate byte {start_pos}.")

	with open(zst_file, "rb") as compressed:
		with dctx.stream_reader(compressed) as reader:
			text_stream = io.TextIOWrapper(reader, encoding="utf-8")

			# Read and discard data until we get near `start_pos`
			bytes_read = 0
			while bytes_read < start_pos:
				chunk = reader.read(min(1024, start_pos - bytes_read))
				if not chunk:
					break  # End of file reached
				bytes_read += len(chunk)

			# Ensure we're starting at a valid game
			if not find_next_game_start(text_stream):
				logging.warning(f"Worker {worker_id}: Could not find a valid game start, skipping.")
				return

			game_index = 0
			game_data_list = []
			batch_size = 2000

			while bytes_read < end_pos:
				game = chess.pgn.read_game(text_stream)
				if game is None:
					break  # End of file or range reached

				try:
					board = game.board()
					node = game
					game_result = extract_game_result(game.headers)

					if game_result == 2:  # Invalid game
						continue

					game_data = []
					move_number = 1

					while node.variations:
						next_node = node.variation(0)
						move = next_node.move
						fen = board.fen()
						turn = board.turn
						relative_modifier = -1 if turn else 1

						example = {
							"fen": fen,
							"move": move.uci(),
							"game_result": game_result,
							"label_sparse": int((game_result * relative_modifier) + 1),
						}
						game_data.append(example)

						try:
							board.push(move)
						except Exception as e:
							logging.error(f"Worker {worker_id}: Illegal move '{move.uci()}' at game {game_index}: {e}")
							break  # Skip this game

						move_number += 1
						node = next_node

					game_data_list.append(game_data)
					game_index += 1

				except Exception as e:
					logging.error(f"Worker {worker_id}: Error parsing game {game_index}: {e}")
					continue  # Skip corrupt game

				# Save batch every `batch_size` games
				if len(game_data_list) >= batch_size:
					json_filename = os.path.join(output_folder, f"worker_{worker_id}_batch_{game_index//batch_size}.json")
					with open(json_filename, "w", encoding="utf-8") as f:
						json.dump(game_data_list, f, indent=4)
					game_data_list = []  # Reset batch

			# Save remaining data
			if game_data_list:
				json_filename = os.path.join(output_folder, f"worker_{worker_id}_batch_{game_index//batch_size}.json")
				with open(json_filename, "w", encoding="utf-8") as f:
					json.dump(game_data_list, f, indent=4)

	logging.info(f"Worker {worker_id}: Finished processing from byte {start_pos} to {end_pos}.")

def pgn_zst_parallel(zst_file, output_folder, num_workers=8):
	"""Splits the .pgn.zst file into byte chunks and assigns them to different workers."""
	os.makedirs(output_folder, exist_ok=True)

	# Get total file size
	file_size = os.path.getsize(zst_file)
	chunk_size = file_size // num_workers

	processes = []
	start_time = time()

	for worker_id in tqdm(range(num_workers)):
		start_pos = worker_id * chunk_size
		end_pos = file_size if worker_id == num_workers - 1 else (worker_id + 1) * chunk_size

		process = mp.Process(target=process_chunk, args=(zst_file, start_pos, end_pos, worker_id, output_folder))
		process.start()
		processes.append(process)

	for process in processes:
		process.join()

	total_time = time() - start_time
	logging.info(f"Processing completed in {total_time:.2f} seconds.")

if __name__ == "__main__":
	target_date = "2025-02"
	pgn_zst_parallel(f"data/lichess_db_standard_rated_{target_date}.pgn.zst", f"fen_data/{target_date}/", num_workers=32)