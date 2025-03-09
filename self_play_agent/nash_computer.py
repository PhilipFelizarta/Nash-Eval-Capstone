import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import scipy.optimize
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from core.nash import logit_transform, compute_nash_equilibrium, symmetrize_winrate_matrix

def load_winrate_matrix(tournament_folder):
	npz_path = os.path.join(tournament_folder, "winrate_matrix.npz")

	if not os.path.exists(npz_path):
		raise FileNotFoundError(f"Winrate matrix not found in {tournament_folder}")

	data = np.load(npz_path, allow_pickle=True)
	winrate_matrix = data["winrate_matrix"]
	model_names = data["models"].tolist()  # Convert to list for pandas
	tournament_name = os.path.basename(tournament_folder)

	return winrate_matrix, model_names, tournament_name

def save_nash_results(tournament_folder):
	winrate_matrix, model_names, tournament_name = load_winrate_matrix(tournament_folder)

	symm_winrate_matrix = symmetrize_winrate_matrix(winrate_matrix)

	logit_matrix = logit_transform(symm_winrate_matrix)
	nash_distribution = compute_nash_equilibrium(logit_matrix)

	df = pd.DataFrame({"Model": model_names, "Nash_Probability": nash_distribution})
	os.makedirs(f"nash/{tournament_name}", exist_ok=True)
	df.to_csv(f"nash/{tournament_name}/nash.csv", index=False)

	print(f"Nash results saved to nash/{tournament_name}/nash.csv")


def load_nash_results(tournament_folder):
	"""
	Loads the winrate matrix and Nash equilibrium results.

	Args:
		tournament_folder (str): Path to the tournament directory.

	Returns:
		np.ndarray: Winrate matrix.
		pd.DataFrame: Nash equilibrium results.
		list: Model names.
	"""
	# Load winrate matrix
	npz_path = os.path.join(tournament_folder, "winrate_matrix.npz")
	if not os.path.exists(npz_path):
		raise FileNotFoundError(f"Winrate matrix not found in {tournament_folder}")
	
	data = np.load(npz_path, allow_pickle=True)
	winrate_matrix = data["winrate_matrix"]
	winrate_matrix = symmetrize_winrate_matrix(winrate_matrix)
	model_names = data["models"].tolist()

	# Load Nash results
	tournament_name = os.path.basename(tournament_folder)
	csv_path = os.path.join("nash", f"{tournament_name}/nash.csv")
	if not os.path.exists(csv_path):
		raise FileNotFoundError(f"Nash equilibrium file not found at {csv_path}")

	df_nash = pd.read_csv(csv_path)

	return winrate_matrix, df_nash, model_names

def plot_winrate_matrix(winrate_matrix, model_names, tournament_name):
	"""
	Plots a heatmap of the winrate matrix.

	Args:
		winrate_matrix (np.ndarray): The winrate matrix.
		model_names (list): List of model names.
		tournament_name (str): Name of the tournament.
	"""
	plt.figure(figsize=(10, 8))
	sns.heatmap(winrate_matrix, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=model_names, yticklabels=model_names)
	plt.xlabel("Model")
	plt.ylabel("Model")
	plt.title(f"Winrate Matrix - {tournament_name}")
	plt.xticks(rotation=45, ha="right")
	plt.yticks(rotation=0)
	plt.tight_layout()
	plt.savefig(f"nash/{tournament_name}/winrate_matrix.png", dpi=300, bbox_inches="tight")
	plt.close()

def plot_nash_equilibria(df_nash, tournament_name):
	"""
	Plots the Nash equilibrium distribution as a bar chart.

	Args:
		df_nash (pd.DataFrame): DataFrame containing Nash equilibrium results.
		tournament_name (str): Name of the tournament.
	"""
	# Sort models by Nash probability for better visualization
	df_nash = df_nash.sort_values(by="Nash_Probability", ascending=False)

	models = df_nash["Model"]
	nash_probs = df_nash["Nash_Probability"]

	# Create a single bar chart
	plt.figure(figsize=(12, 6))
	plt.bar(models, nash_probs, color="blue", alpha=0.7)
	plt.ylabel("Nash Probability")
	plt.xlabel("Model")
	plt.title(f"MaxEnt Nash Equilibrium ({tournament_name})")
	plt.ylim(0, 1)

	# Rotate model names for readability
	plt.xticks(rotation=45, ha="right")

	# Save the plot
	output_path = f"nash/{tournament_name}/nash_image.png"
	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	plt.savefig(output_path, dpi=300, bbox_inches="tight")
	plt.close()

	print(f"Nash equilibrium plot saved to {output_path}")

if __name__ == "__main__":
	tournament_folder = "games/tournament_eda_model_10N"  # Example path
	save_nash_results(tournament_folder)
	winrate_matrix, df_nash, model_names = load_nash_results(tournament_folder)

	# Plot winrate matrix heatmap
	plot_winrate_matrix(winrate_matrix, model_names, os.path.basename(tournament_folder))

	# Plot Nash equilibrium results
	plot_nash_equilibria(df_nash, os.path.basename(tournament_folder))