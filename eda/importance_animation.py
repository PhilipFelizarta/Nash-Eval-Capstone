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

def plot_piece_density_by_importance(zst_file, sample_size=50000, num_frames=20):
	"""Creates animated gifs showing how piece densities evolve with importance using percentile-based thresholds."""
	gen = chess_database.pgn_zst_generator(zst_file, batch_size=32)

	# Define storage for piece densities
	all_importance = []
	all_game_data = []

	# Storage for frames
	hero_frames = {i: [] for i in range(7)}
	villain_frames = {i: [] for i in range(7)}

	count = 0
	for game_data, _, importance in gen:
		all_importance.extend(importance.tolist())  # Track all importance values
		all_game_data.append((game_data, importance))  # Store full game data

		count += game_data.shape[0]
		if count >= sample_size:
			break

	# Dynamically compute percentile thresholds across all games
	importance_thresholds = np.percentile(all_importance, np.linspace(0, 100, num_frames))

	# Generate frames for each piece
	for piece in range(7):
		for threshold in importance_thresholds:
			hero_snapshot = np.zeros((8, 8))
			villain_snapshot = np.zeros((8, 8))
			num_positions = 0  # Track valid positions

			# Iterate over ALL collected game data
			for game_data, importance in all_game_data:
				for i in range(game_data.shape[0]):
					if importance[i] >= threshold:  # Use percentile thresholding
						hero_snapshot += game_data[i, :, :, piece]
						villain_snapshot += game_data[i, :, :, piece + 9]
						num_positions += 1  # Count valid positions

			# Normalize snapshot using total valid positions
			if num_positions > 0:
				hero_snapshot /= num_positions
				villain_snapshot /= num_positions

			# Store frames
			hero_frames[piece].append(hero_snapshot.copy())
			villain_frames[piece].append(villain_snapshot.copy())

	# Generate and save GIFs
	save_animation(hero_frames, villain_frames, importance_thresholds)

def save_animation(hero_frames, villain_frames, importance_thresholds):
	"""Saves animations for each piece's density evolution."""
	os.makedirs("figures/animations", exist_ok=True)
	cmap = "magma"  # Updated colormap for better contrast

	for piece in range(7):
		hero_images = []

		for frame in range(len(hero_frames[piece])):
			fig, ax = plt.subplots(1, 2, figsize=(10, 5))

			# Super title with rounded importance value
			rounded_importance = round(importance_thresholds[frame], 2)
			fig.suptitle(f"Piece Density Evolution | Importance ≥ {rounded_importance}", fontsize=14)

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
		imageio.mimsave(f"figures/animations/importance_{PIECE_NAMES[piece]}.gif", hero_images, duration=0.5)

		# Clean up temp frames
		for frame in range(len(hero_frames[piece])):
			os.remove(f"figures/animations/frame_{piece}_{frame}.png")

if __name__ == "__main__":
	zst_file = "data\LumbrasGigaBase 2024.pgn.zst"
	plot_piece_density_by_importance(zst_file, sample_size=50000, num_frames=20)
