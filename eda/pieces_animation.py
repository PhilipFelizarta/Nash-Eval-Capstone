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

def plot_piece_density_by_piece_count(zst_file, sample_size=50000):
	"""Creates animated gifs showing how piece densities evolve from 32 to 2 pieces."""
	gen = chess_database.pgn_zst_generator(zst_file, batch_size=32)

	# Storage for positions categorized by piece count
	all_game_data = {count: [] for count in range(32, 1, -1)}

	count = 0
	for game_data, _, _ in gen:
		piece_counts = np.sum(game_data[:, :, :, :7], axis=(1, 2, 3))  # Count all pieces on board
		piece_counts += np.sum(game_data[:, :, :, 9:16], axis=(1, 2, 3))

		for i in range(game_data.shape[0]):
			piece_count = int(piece_counts[i])
			if 2 <= piece_count <= 32:
				all_game_data[piece_count].append(game_data[i])

		count += game_data.shape[0]
		if count >= sample_size:
			break

	# Generate frames for each piece
	hero_frames = {i: [] for i in range(7)}
	villain_frames = {i: [] for i in range(7)}
	valid_piece_counts = [pc for pc in all_game_data.keys() if len(all_game_data[pc]) > 0]

	for piece in range(7):
		for piece_count in valid_piece_counts:
			positions = np.array(all_game_data[piece_count])  # Convert to NumPy array

			if positions.shape[0] == 0:
				continue  # Skip if no data for this piece count

			# Compute average density
			hero_snapshot = np.mean(positions[:, :, :, piece], axis=0)
			villain_snapshot = np.mean(positions[:, :, :, piece + 9], axis=0)

			# Store frames
			hero_frames[piece].append(hero_snapshot)
			villain_frames[piece].append(villain_snapshot)

	# Generate and save GIFs
	save_animation_pieces(hero_frames, villain_frames, valid_piece_counts)

def save_animation_pieces(hero_frames, villain_frames, piece_counts):
	"""Saves animations for each piece's density evolution."""
	os.makedirs("figures/animations/", exist_ok=True)
	cmap = "Reds"

	for piece in range(7):
		hero_images = []

		for frame in range(len(hero_frames[piece])):
			fig, ax = plt.subplots(1, 2, figsize=(10, 5))

			# Super title with current piece count
			current_piece_count = piece_counts[frame]
			fig.suptitle(f"Piece Density Evolution | {current_piece_count} Pieces", fontsize=14)

			# Hero Density Frame
			chess_env.draw_chessboard(ax[0])
			im = ax[0].imshow(hero_frames[piece][frame], cmap=cmap, alpha=0.9, extent=[0, 8, 0, 8], 
							  vmin=0.0, vmax=1.0)
			ax[0].set_title(f"Hero {PIECE_NAMES[piece]}")
			fig.colorbar(im, ax=ax[0])

			# Villain Density Frame
			chess_env.draw_chessboard(ax[1])
			im = ax[1].imshow(villain_frames[piece][frame], cmap=cmap, alpha=0.9, extent=[0, 8, 0, 8], 
							  vmin=0.0, vmax=1.0)
			ax[1].set_title(f"Villain {PIECE_NAMES[piece]}")
			fig.colorbar(im, ax=ax[1])

			# Save frame as image
			filename = f"figures/animations/frame_{piece}_{frame}.png"
			plt.savefig(filename, dpi=100, bbox_inches="tight", pad_inches=0.1)
			plt.close()

			img = imageio.imread(filename)
			img = np.array(Image.fromarray(img).resize((1000, 500)))  # Ensure consistent size
			hero_images.append(img)

		# Save GIFs
		imageio.mimsave(f"figures/animations/pieces_{PIECE_NAMES[piece]}.gif", hero_images, duration=0.5)

		# Clean up temp frames
		for frame in range(len(hero_frames[piece])):
			os.remove(f"figures/animations/frame_{piece}_{frame}.png")

if __name__ == "__main__":
	zst_file = "data\LumbrasGigaBase 2024.pgn.zst"
	plot_piece_density_by_piece_count(zst_file, sample_size=int(1e5))