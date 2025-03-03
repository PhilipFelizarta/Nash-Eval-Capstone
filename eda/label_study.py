import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import collections
import seaborn as sns
import pandas as pd
import core.chess_database as chess_database

# Standard chess material values
PIECE_VALUES = {
	"Pawn": 1, "Knight": 3, "Bishop": 3, "Rook": 5, "Queen": 9
}

RESULT_CATEGORIES = {0: "Hero Win", 1: "Draw", 2: "Villain Win"}
PIECE_NAMES = ["Pawn", "Knight", "LS Bishop", "DS Bishop", "Rook", "Queen", "King"]
COLORS = {0: "#5393ED", 1: "#999999", 2: "#e3242b"}

def analyze_chess_data(zst_file, sample_size=100000):
	"""Analyzes dataset in one pass, computing class distribution, material counts, and material imbalance."""

	# Initialize tracking structures
	label_counts = collections.Counter()
	total_importance = collections.defaultdict(float)
	material_stats = {0: np.zeros(7), 1: np.zeros(7), 2: np.zeros(7)}
	total_weights = {0: 0, 1: 0, 2: 0}
	imbalance_data = []

	# Process data in a single pass
	gen = chess_database.pgn_zst_generator(zst_file, batch_size=256)
	count = 0

	for game_data, labels, importance in gen:
		for i in range(len(labels)):
			label = labels[i]
			weight = importance[i]

			# Class distribution
			label_counts[label] += 1
			total_importance[label] += weight

			# Material count per game result
			material_stats[label] += weight * (np.sum(game_data[i, :, :, :7], axis=(0, 1)))
			total_weights[label] += weight

			# Material imbalance tracking
			hero_material = sum(PIECE_VALUES[p] * np.sum(game_data[i, :, :, j]) for j, p in enumerate(PIECE_VALUES))
			villain_material = sum(PIECE_VALUES[p] * np.sum(game_data[i, :, :, j+9]) for j, p in enumerate(PIECE_VALUES))
			imbalance_data.append({"Result": RESULT_CATEGORIES[label], "Imbalance": hero_material - villain_material, "Weight": weight})

			count += 1

		if count >= sample_size:
			break

	# Convert imbalance data to Pandas DataFrame for seaborn
	imbalance_df = pd.DataFrame(imbalance_data)

	# --- Plot Class Distribution ---
	total_weight = sum(total_importance.values())
	plt.figure(figsize=(8, 6))
	plt.bar(
		[RESULT_CATEGORIES[k] for k in total_importance.keys()],
		total_importance.values(),
		color=[COLORS[k] for k in total_importance.keys()]
	)
	plt.ylabel("Weighted Position Count (by Importance)")
	plt.title("Weighted Game Result Distribution")
	plt.xticks(rotation=30)
	plt.grid(axis='y', linestyle='--', alpha=0.7)

	for label, weight in total_importance.items():
		print(f"{RESULT_CATEGORIES[label]}: {weight:.2f} ({(weight / total_weight) * 100:.2f}%)")

	plt.savefig("figures/standard_eda/weighted_class_distribution.png", dpi=300, bbox_inches="tight")
	plt.close()

	# --- Plot Material Count ---
	avg_material = {k: material_stats[k] / max(total_weights[k], 1) for k in material_stats}
	
	fig, ax = plt.subplots(figsize=(12, 6))
	bar_width = 0.3
	x = np.arange(7)

	for idx, label in enumerate(RESULT_CATEGORIES.keys()):
		ax.bar(x + idx * bar_width, avg_material[label], width=bar_width, color=COLORS[label], label=RESULT_CATEGORIES[label])

	ax.set_xticks(x + bar_width)
	ax.set_xticklabels(PIECE_NAMES)
	ax.set_ylabel("Weighted Average Hero Piece Count")
	ax.set_title("Weighted Average Hero Piece Count per Game Result")
	ax.legend()
	plt.grid(axis="y", linestyle="--", alpha=0.7)
	plt.savefig("figures/standard_eda/weighted_material_count.png", dpi=300, bbox_inches="tight")
	plt.close()

	# --- Plot Material Imbalance Histogram ---
	fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True, sharey=True)
	fig.suptitle("Material Imbalance Distribution Across Game Results", fontsize=16)

	for idx, (label, result_name) in enumerate(RESULT_CATEGORIES.items()):
		ax = axes[idx]
		subset = imbalance_df[imbalance_df["Result"] == result_name]
		
		# Plot discrete histogram with integer bins
		sns.histplot(
			subset, x="Imbalance", weights="Weight", binwidth=1, discrete=True,
			color=COLORS[label], stat="density", element="step", ax=ax, kde=True
		)
		# Highlight critical values: 0.0
		for v in [0]:
			ax.axvline(v, color="black", linestyle="dashed", alpha=0.7, linewidth=1.2)

		ax.set_title(f"{result_name}", fontsize=14)
		ax.set_xlabel("Material Imbalance (Hero - Villain)")
		ax.set_ylabel("Density")
		ax.grid(axis="y", linestyle="--", alpha=0.7)
		ax.set_xticks(range(int(subset["Imbalance"].min()), int(subset["Imbalance"].max()) + 1, 1))  # Force integer ticks

	plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit title
	plt.savefig("figures/standard_eda/weighted_material_imbalance_histogram.png", dpi=300, bbox_inches="tight")
	plt.close()

if __name__ == "__main__":
	zst_file = "data/LumbrasGigaBase 2024.pgn.zst"
	analyze_chess_data(zst_file, sample_size=int(1e6))
