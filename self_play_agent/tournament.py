import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import json
import numpy as np
from itertools import product
from core.self_play import self_play_game_stochastic


def find_models(model_folder, N_players=None):
	"""
	Scans the model folder and selects a subset of models for the tournament.
	Ensures the first and last model are always included, and evenly selects
	the remaining models.

	Args:
		model_folder (str): Path to the folder containing models.
		N_players (int, optional): Max number of models in the tournament.

	Returns:
		list: List of selected model file paths.
	"""
	all_models = sorted([
		os.path.join(model_folder, f) for f in os.listdir(model_folder)
		if f.endswith(".h5")
	])

	if len(all_models) < 2:
		raise ValueError("At least two models are required for a tournament.")

	if N_players is None or N_players >= len(all_models):
		return all_models  # Use all models if not capping

	# Always include the first and last model
	selected_models = [all_models[0], all_models[-1]]

	if N_players > 2:
		remaining_models = all_models[1:-1]  # Exclude first and last
		indices = np.linspace(0, len(remaining_models) - 1, N_players - 2, dtype=int)
		selected_models += [remaining_models[i] for i in indices]

	return selected_models


def run_tournament(model_folder, N_players=8, M=5, temp=0.8):
	"""
	Runs a round-robin tournament with a limited number of models.

	Args:
		model_folder (str): Path to the folder containing models.
		N_players (int): Max number of models in the tournament.
		M (int): Number of games per matchup (per color).
		temp (float): Temperature parameter for stochastic move selection.
	"""
	models = find_models(model_folder, N_players)
	N = len(models)

	print("Creating Tournament...\n Models: ", models)
	timestamp = time.strftime("%Y%m%d_%H%M%S")
	tournament_folder = f"games/tournament_{timestamp}"
	os.makedirs(tournament_folder, exist_ok=True)

	# Initialize winrate matrix
	winrate_matrix = np.zeros((N, N), dtype=np.float32)
	match_counts = np.zeros((N, N), dtype=np.int32)

	# Store tournament metadata
	tournament_info = {
		"timestamp": timestamp,
		"num_models": N,
		"num_games_per_matchup": M,
		"temperature": temp,
		"models": {i: os.path.basename(models[i]) for i in range(N)},
		"results": []
	}

	# Create all possible matchups (including self-play)
	for idx_a, idx_b in product(range(N), repeat=2):
		model_a = models[idx_a]
		model_b = models[idx_b]
		matchup_folder = os.path.join(tournament_folder, f"{os.path.basename(model_a)}_{os.path.basename(model_b)}")
		os.makedirs(matchup_folder, exist_ok=True)

		# Each pair plays M times as White/Black
		for i in range(M):
			game_path = os.path.join(matchup_folder, f"game_{i}.pgn")
			result = self_play_game_stochastic(model_a, model_b, game_path, temp=temp)

			# Update winrate matrix from White's perspective
			if result == "1-0":
				winrate_matrix[idx_a, idx_b] += 1  # White won
			elif result == "1/2-1/2":
				winrate_matrix[idx_a, idx_b] += 0.5  # Draw
			elif result == "0-1":
				pass  # Black won; no need to update White's winrate
			else:
				winrate_matrix[idx_a, idx_b] += 0.5  # Any other case, count as a draw

			match_counts[idx_a, idx_b] += 1  # Increment game count
			
	# Normalize winrate matrix
	with np.errstate(divide="ignore", invalid="ignore"):
		winrate_matrix = np.where(match_counts > 0, winrate_matrix / match_counts, 0)

	# Save winrate matrix
	npz_path = os.path.join(tournament_folder, "winrate_matrix.npz")
	np.savez_compressed(npz_path, winrate_matrix=winrate_matrix, models=models)

	# Save tournament metadata as JSON
	json_path = os.path.join(tournament_folder, "tournament_info.json")
	with open(json_path, "w", encoding="utf-8") as json_file:
		json.dump(tournament_info, json_file, indent=4)

	print(f"Tournament completed! Results saved to {tournament_folder}/")


if __name__ == "__main__":
	model_folder = "models/eda_cnn"  # Path to models
	N_players = 2  # Max number of models in the tournament
	M = 1  # Number of games per matchup (per color)
	temp = 1 / 100.0  # Exploration parameter

	run_tournament(model_folder, N_players, M, temp)