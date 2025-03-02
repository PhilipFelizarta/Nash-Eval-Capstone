import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Model
import chess

import core.chess_database as chess_database
import core.chess_environment as chess_env
import collections

PIECE_NAMES = ["Pawn", "Knight", "Bishop", "Rook", "Queen", "King"]

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

	hero_boards = np.zeros((8, 8, 6))  # One board per piece type for the hero
	villain_boards = np.zeros((8, 8, 6))

	count = 0
	for game_data, _ in gen:
		game_data = np.array(game_data)  # Ensure it's an array
		
		hero_pieces = np.sum(game_data[:, :, :, :6], axis=0)  # Sum over batch
		villain_pieces = np.sum(game_data[:, :, :, 8:14], axis=0)

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
	fig, axes = plt.subplots(2, 3, figsize=(15, 10))
	fig.suptitle(f"Hero Piece Density Across {count} Positions")
	for i, ax in enumerate(axes.flat):
		chess_env.draw_chessboard(ax)
		im = ax.imshow(avg_hero[:, :, i], cmap="PuBuGn", interpolation="nearest", alpha=0.9, extent=[0, 8, 0, 8])
		ax.set_title(PIECE_NAMES[i])
		fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
	
	plt.savefig("figures/standard_eda/hero_piece_distribution.png", dpi=300, bbox_inches="tight")
	plt.close()

	# Plot Villain Piece Density by Type
	fig, axes = plt.subplots(2, 3, figsize=(15, 10))
	fig.suptitle(f"Villain Piece Density Across {count} Positions")
	for i, ax in enumerate(axes.flat):
		chess_env.draw_chessboard(ax)
		im = ax.imshow(avg_villain[:, :, i], cmap="PuBuGn", interpolation="nearest", alpha=0.9, extent=[0, 8, 0, 8])
		ax.set_title(PIECE_NAMES[i])
		fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
	plt.savefig("figures/standard_eda/villain_piece_distribution.png", dpi=300, bbox_inches="tight")
	plt.close()

if __name__ == "__main__":
	zst_file = "data\LumbrasGigaBase 2024.pgn.zst"
	plot_piece_density_by_type(zst_file, sample_size=100)