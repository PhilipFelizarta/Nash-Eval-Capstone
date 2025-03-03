import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Model
import chess
import imageio.v2 as imageio
from PIL import Image

import core.chess_database as chess_database
import core.chess_environment as chess_env
import collections

PIECE_NAMES = ["Pawn", "Knight", "LS Bishop", "DS Bishop", "Rook", "Queen", "King"]
RESULT_CATEGORIES = {0: "Hero Win", 1: "Draw", 2: "Villain Win"}

def label_distribution(zst_file, sample_size=100000):
	label_counts = collections.Counter()
	gen = chess_database.pgn_zst_generator(zst_file, batch_size=256)

	for _, labels in gen:
		label_counts.update(labels.tolist())
		if sum(label_counts.values()) >= sample_size:
			break

	total = sum(label_counts.values())
	for label, count in label_counts.items():
		print(f"Label {label}: {count} ({(count / total) * 100:.2f}%)")

def plot_piece_density_by_type(zst_file, sample_size=1000):
	gen = chess_database.pgn_zst_generator(zst_file, batch_size=32)

	hero_boards = np.zeros((8, 8, 7))  # One board per piece type for the hero
	villain_boards = np.zeros((8, 8, 7))

	count = 0
	for game_data, _, _ in gen:
		game_data = np.array(game_data)  # Ensure it's an array
		
		hero_pieces = np.sum(game_data[:, :, :, :7], axis=0)  # Sum over batch (0:Pawn,1:Knight,2:LSB,3:DSB,4:Rook,5:Queen,6:King)
		villain_pieces = np.sum(game_data[:, :, :, 9:16], axis=0)  # Adjusted for LSB/DSB

		hero_boards += hero_pieces
		villain_boards += villain_pieces
		count += game_data.shape[0]

		if count >= sample_size:
			break

	if count == 0:
		print("No positions processed.")
		return

	avg_hero = hero_boards / count
	avg_villain = villain_boards / count

	# Plot Hero Piece Density by Type
	fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # Adjusted for 7 pieces
	fig.suptitle(f"Hero Piece Density Across {count} Positions")
	for i, ax in enumerate(axes.flat[:7]):
		chess_env.draw_chessboard(ax)
		im = ax.imshow(avg_hero[:, :, i], cmap="PuBuGn", interpolation="nearest", alpha=0.9, extent=[0, 8, 0, 8])
		ax.set_title(PIECE_NAMES[i])
		fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
	
	plt.savefig("figures/standard_eda/hero_piece_distribution.png", dpi=300, bbox_inches="tight")
	plt.close()

	# Plot Villain Piece Density by Type
	fig, axes = plt.subplots(2, 4, figsize=(20, 10))
	fig.suptitle(f"Villain Piece Density Across {count} Positions")
	for i, ax in enumerate(axes.flat[:7]):
		chess_env.draw_chessboard(ax)
		im = ax.imshow(avg_villain[:, :, i], cmap="PuBuGn", interpolation="nearest", alpha=0.9, extent=[0, 8, 0, 8])
		ax.set_title(PIECE_NAMES[i])
		fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
	
	plt.savefig("figures/standard_eda/villain_piece_distribution.png", dpi=300, bbox_inches="tight")
	plt.close()

def plot_piece_density_by_label(zst_file, sample_size=100000):
	"""Plots hero and villain piece densities for different game outcomes, adjusted by importance sampling."""
	gen = chess_database.pgn_zst_generator(zst_file, batch_size=32)

	piece_densities = {
		0: {"hero": np.zeros((8, 8, 7)), "villain": np.zeros((8, 8, 7)), "weight_sum": 0, "count_samples": 0},
		1: {"hero": np.zeros((8, 8, 7)), "villain": np.zeros((8, 8, 7)), "weight_sum": 0, "count_samples": 0},
		2: {"hero": np.zeros((8, 8, 7)), "villain": np.zeros((8, 8, 7)), "weight_sum": 0, "count_samples": 0},
	}

	count = 0
	for game_data, labels, importance in gen:
		for i in range(len(labels)):
			label = labels[i]
			if label in piece_densities:
				weight = importance[i]

				piece_densities[label]["hero"] += weight * game_data[i, :, :, :7]
				piece_densities[label]["villain"] += weight * game_data[i, :, :, 9:16]
				piece_densities[label]["weight_sum"] += weight
				piece_densities[label]["count_samples"] += 1  
				
				count += 1
		
		if count >= sample_size:
			break

	for result_label, result_name in RESULT_CATEGORIES.items():
		total_weight = piece_densities[result_label]["weight_sum"]
		num_samples = piece_densities[result_label]["count_samples"]

		if total_weight == 0:
			print(f"No positions processed for {result_name}.")
			continue

		avg_hero = piece_densities[result_label]["hero"] / total_weight
		avg_villain = piece_densities[result_label]["villain"] / total_weight

		# Plot Hero Piece Density
		fig, axes = plt.subplots(3, 3, figsize=(16, 12))
		fig.suptitle(f"Hero Piece Density ({result_name})\nTotal Weight: {total_weight:.2f} | Samples: {num_samples}", fontsize=16)
		plt.subplots_adjust(hspace=0.2, wspace=0.3)

		for i, ax in enumerate(axes.flat):
			if i < 7:
				chess_env.draw_chessboard(ax)
				im = ax.imshow(avg_hero[:, :, i], cmap="PuBuGn", interpolation="nearest", alpha=0.9, extent=[0, 8, 0, 8])
				ax.set_title(f"{PIECE_NAMES[i]}", fontsize=14)
				fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
			else:
				ax.axis("off")

		plt.savefig(f"figures/standard_eda/hero_piece_distribution_{result_label}.png", dpi=300, bbox_inches="tight")
		plt.close()

		# Plot Villain Piece Density
		fig, axes = plt.subplots(3, 3, figsize=(16, 12))
		fig.suptitle(f"Villain Piece Density ({result_name})\nTotal Weight: {total_weight:.2f} | Samples: {num_samples}", fontsize=16)
		plt.subplots_adjust(hspace=0.2, wspace=0.3)

		for i, ax in enumerate(axes.flat):
			if i < 7:
				chess_env.draw_chessboard(ax)
				im = ax.imshow(avg_villain[:, :, i], cmap="PuBuGn", interpolation="nearest", alpha=0.9, extent=[0, 8, 0, 8])
				ax.set_title(f"{PIECE_NAMES[i]}", fontsize=14)
				fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
			else:
				ax.axis("off")

		plt.savefig(f"figures/standard_eda/villain_piece_distribution_{result_label}.png", dpi=300, bbox_inches="tight")
		plt.close()

if __name__ == "__main__":
	zst_file = "data\LumbrasGigaBase 2024.pgn.zst"
	plot_piece_density_by_label(zst_file, sample_size=100000)