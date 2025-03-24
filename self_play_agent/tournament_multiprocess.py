import sys
import os
import time
import json
import numpy as np
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add core path for importing self_play_game_stochastic
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.self_play import self_play_game_stochastic


def find_models(model_folder, N_players=None):
	"""
	Scans the model folder and selects a subset of models for the tournament.
	Ensures the first and last model are always included, and evenly selects
	the remaining models.
	"""
	all_models = sorted([
		os.path.join(model_folder, f) for f in os.listdir(model_folder)
		if f.endswith(".h5")
	])

	if len(all_models) < 2:
		raise ValueError("At least two models are required for a tournament.")

	if N_players is None or N_players >= len(all_models):
		return all_models  # Use all models if not capping

	if N_players > 2:
		indices = np.linspace(0, len(all_models) - 1, N_players, dtype=int)
		selected_models = [all_models[i] for i in indices]
		return selected_models

	return all_models


def play_matchup(idx_a, idx_b, model_a, model_b, matchup_folder, M, temp):
	"""
	Plays M games between two models (from model_a vs model_b perspective).
	Returns win and match statistics for updating the winrate matrix.
	"""
	results = {
		"idx_a": idx_a,
		"idx_b": idx_b,
		"wins": 0.0,
		"total": 0,
	}

	if idx_a == idx_b:
		results["wins"] = M / 2
		results["total"] = M
		return results

	os.makedirs(matchup_folder, exist_ok=True)

	for i in range(M):
		game_path = os.path.join(matchup_folder, f"game_{i}.pgn")
		result = self_play_game_stochastic(model_a, model_b, game_path, temp=temp, max_moves=50)

		if result == "1-0":
			results["wins"] += 1
		elif result == "1/2-1/2":
			results["wins"] += 0.5
		results["total"] += 1

	return results


def run_tournament(model_folder, N_players=8, M=5, temp=0.8, workers=4):
	"""
	Runs a round-robin tournament using multiprocessing.
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

	# Tournament metadata
	tournament_info = {
		"timestamp": timestamp,
		"num_models": N,
		"num_games_per_matchup": M,
		"temperature": temp,
		"models": {i: os.path.basename(models[i]) for i in range(N)},
		"results": []
	}

	# Run matchups in parallel
	with ProcessPoolExecutor(max_workers=workers) as executor:
		futures = []
		for idx_a, idx_b in product(range(N), repeat=2):
			model_a = models[idx_a]
			model_b = models[idx_b]
			matchup_folder = os.path.join(tournament_folder, f"{os.path.basename(model_a)}_vs_{os.path.basename(model_b)}")

			futures.append(
				executor.submit(
					play_matchup,
					idx_a, idx_b, model_a, model_b, matchup_folder, M, temp
				)
			)

		# Collect results
		for future in as_completed(futures):
			result = future.result()
			a, b = result["idx_a"], result["idx_b"]
			winrate_matrix[a, b] += result["wins"]
			match_counts[a, b] += result["total"]

	# Normalize winrate matrix
	with np.errstate(divide="ignore", invalid="ignore"):
		winrate_matrix = np.where(match_counts > 0, winrate_matrix / match_counts, 0)

	# Save winrate matrix
	npz_path = os.path.join(tournament_folder, "winrate_matrix.npz")
	np.savez_compressed(npz_path, winrate_matrix=winrate_matrix, models=models)

	# Save tournament metadata
	json_path = os.path.join(tournament_folder, "tournament_info.json")
	with open(json_path, "w", encoding="utf-8") as json_file:
		json.dump(tournament_info, json_file, indent=4)

	print(f"Tournament completed! Results saved to {tournament_folder}/")


if __name__ == "__main__":
	model_folder = "models/RESNET_36p_4x512"
	N_players = int(os.getenv("N_PLAYERS"))
	M = int(os.getenv("N_GAMES"))
	temp = float(os.getenv("TEMP"))
	workers = int(os.getenv("WORKERS"))
	
	print("Number of workers used: ", workers)  # Number of workers being used by default

	run_tournament(model_folder, N_players, M, temp, workers)