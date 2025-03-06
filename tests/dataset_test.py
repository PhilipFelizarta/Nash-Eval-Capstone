import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import core.chess_database as chess_database

if __name__ == "__main__":
	# Load dataset
	zst_file = "data/LumbrasGigaBase 2024.pgn.zst"
	total_games, total_positions, sample_pgns = chess_database.count_ply_and_games(zst_file, sample_count=1)

	print(f"Total games: {total_games}")
	print(f"Total positions: {total_positions}")
	print("Sample PGN:")
	print(sample_pgns[0] if sample_pgns else "No samples available")
