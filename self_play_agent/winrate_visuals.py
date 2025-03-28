import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from core.nash import symmetrize_winrate_matrix

def load_winrate_matrix(tournament_folder, start_num):
	npz_path = os.path.join(tournament_folder, "winrate_matrix.npz")

	if not os.path.exists(npz_path):
		raise FileNotFoundError(f"Winrate matrix not found in {tournament_folder}")

	data = np.load(npz_path, allow_pickle=True)
	winrate_matrix = data["winrate_matrix"]
	model_names = data["models"].tolist()  # Convert to list for pandas
	fixed_names = [start_num + int(name.split("_")[-1].split(".")[0]) for name in model_names]
	tournament_name = os.path.basename(tournament_folder)
	print(fixed_names)

	return symmetrize_winrate_matrix(winrate_matrix), fixed_names, tournament_name

def plot_winrate_matrix(winrate_matrix, model_names, tournament_name):
	plt.figure(figsize=(10, 8))
	sns.heatmap(winrate_matrix, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=model_names, yticklabels=model_names)
	plt.xlabel("Epoch")
	plt.ylabel("Epoch")
	plt.title(f"Winrate Matrix - {tournament_name}")
	plt.xticks(rotation=45, ha="right")
	plt.yticks(rotation=0)
	plt.tight_layout()
	plt.savefig(f"nash/{tournament_name}/winrate_matrix.png", dpi=300, bbox_inches="tight")
	plt.close()

if __name__ == "__main__":
	tournament_folder = "games/tournament_RESNET_P2_Depth1_WL"  # Example path
	winrate_matrix, model_names, tournament_name, = load_winrate_matrix(tournament_folder, 22)

	# Plot winrate matrix heatmap
	plot_winrate_matrix(winrate_matrix, model_names, os.path.basename(tournament_folder))