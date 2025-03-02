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

def generate_heatmap(model, img, class_idx):
	"""
	Generates a heatmap using GAP weights for interpretability.
	"""
	# Extract the output of the last convolutional layer
	last_conv_layer = model.layers[-2]  # GAP is directly after this layer
	last_conv_model = Model(model.input, last_conv_layer.input)
	
	# Compute feature maps from last conv layer
	last_conv_output = last_conv_model.predict(img[np.newaxis, :, :, :])  # Add batch dim
	last_conv_output = np.squeeze(last_conv_output)  # Shape: (8, 8, 64)

	# Get the final Dense layer's weights (GAP → Dense)
	final_dense_weights = model.layers[-1].weights[0].numpy()  # Shape: (64, 3)
	final_dense_weights_for_class = final_dense_weights[:, class_idx]  # Shape: (64,)

	# Compute the weighted sum of feature maps
	heat_map = np.dot(last_conv_output, final_dense_weights_for_class)

	# Normalize heatmap
	heat_map = np.maximum(heat_map, 0)  # ReLU (remove negatives)
	heat_map /= np.max(heat_map)  # Normalize to [0,1]

	return heat_map

# Function to visualize the heatmap on a chessboard
def plot_heatmap(model, sample_input, true_class, pred_label, path="example_map.png"):
	"""Visualizes the heatmap over a properly rendered chessboard."""
	
	# Generate heatmap from model
	heatmap = generate_heatmap(model, sample_input, pred_label)

	# Convert tensor back to a chess board
	fig, ax = plt.subplots(figsize=(6, 6))
	
	# Draw Chessboard
	chess_env.draw_chessboard(ax)
	
	# Convert Tensor to Board & Draw Pieces
	fen = chess_env.tensor_to_fen(sample_input)  # Convert tensor to FEN
	board = chess.Board(fen)
	chess_env.draw_pieces(ax, board)  # Draw Pieces

	# Overlay Heatmap
	heatmap_img = ax.imshow(
		heatmap, cmap='jet', alpha=0.6, extent=[0, 8, 0, 8]
	)

	# Add Colorbar
	plt.colorbar(heatmap_img, ax=ax, fraction=0.046, pad=0.04)
	plt.title(f"Label: {true_class} -- Pred:{pred_label}")
	
	# Save & Show
	os.makedirs("figures", exist_ok=True)
	plt.savefig(f"figures/{path}", dpi=300, bbox_inches="tight")
	plt.close()

if __name__ == "__main__":
	model = keras.Sequential([
		keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(8, 8, 17)),
		keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
		keras.layers.GlobalAveragePooling2D(),  # Directly after conv layers
		keras.layers.Dense(3, activation='softmax')  # Final classification layer
	])

	model.compile(
		optimizer=keras.optimizers.Adam(),
		loss=keras.losses.SparseCategoricalCrossentropy(),
		metrics=['accuracy']
	)

	zst_file = "data/lichess_db_standard_rated_2025-02.pgn.zst"
	dataset = chess_database.get_tf_dataset(zst_file, batch_size=4)

	model.fit(dataset, epochs=1, steps_per_epoch=1000)

	# Save model
	os.makedirs("models", exist_ok=True)
	model.save("models/eda_model.keras")

	# Get a sample from dataset
	sample_batch = next(iter(dataset.take(1)))  # Take one batch
	sample_input, sample_label = sample_batch

	sample_input = sample_input.numpy()[0]  # Extract first example
	true_class = int(sample_label.numpy()[0])  # Extract true class

	# Make prediction
	pred_probs = model.predict(sample_input[np.newaxis, ...])  # Add batch dim
	pred_label = np.argmax(pred_probs)  # Get the highest probability class

	# Generate and plot heatmap
	plot_heatmap(model, sample_input, true_class, pred_label, path="example.png")