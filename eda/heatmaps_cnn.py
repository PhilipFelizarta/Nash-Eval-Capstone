import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import chess
import os

import core.chess_database as chess_database
import core.chess_environment as chess_env

LABEL_MAPPING = {
	0: "Hero Win", 
	1: "Draw",
	2: "Villain Win"
}

def feature_extractor(model):
	""" Extracts the last convolutional feature map. """
	last_conv_layer = model.get_layer("last_conv")  # Extract last feature maps
	return tf.keras.Model(inputs=model.input, outputs=last_conv_layer.output)

def compute_activation_heatmap(feature_extractor, input_tensor):
	"""
	Computes a heatmap by averaging feature maps from the last convolutional layer.
	This does NOT depend on prediction labels.
	"""
	# ✅ Ensure input is a TensorFlow tensor
	input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.float32)[None, ...]  # Add batch dim

	# Extract activations from last convolutional layer
	feature_maps = feature_extractor(input_tensor)  # Shape: (1, 8, 8, C)

	# ✅ Average across channels to get a single heatmap (Shape: (8,8))
	heatmap = tf.reduce_mean(feature_maps, axis=-1)[0]  # Remove batch dim

	# Convert to NumPy and normalize
	heatmap = heatmap.numpy()
	heatmap /= np.max(np.abs(heatmap))
	heatmap = np.tanh(heatmap)  # Tanh bro

	return heatmap


def plot_heatmap(feature_extractor, model, sample_input, label, path="example_map.png"):
	""" Generates and overlays the heatmap on a chessboard with the model's prediction as a title. """
	
	# Compute activation heatmap
	heatmap = compute_activation_heatmap(feature_extractor, sample_input)

	# ✅ Get model prediction
	sample_input_tensor = tf.convert_to_tensor(sample_input, dtype=tf.float32)[None, ...]
	prediction = model(sample_input_tensor, training=False).numpy()[0]  # Extract softmax output

	# ✅ Convert softmax prediction to readable percentages
	prediction_dict = {
		"Hero Win": prediction[0] * 100,
		"Draw": prediction[1] * 100,
		"Villain Win": prediction[2] * 100,
	}

	# Format title string
	title_str = f"Hero: {prediction_dict['Hero Win']:.1f}% | Draw: {prediction_dict['Draw']:.1f}% | Villain: {prediction_dict['Villain Win']:.1f}%\nLabel: {LABEL_MAPPING[label.numpy()]}"

	# Convert tensor to chess board
	fig, ax = plt.subplots(figsize=(6, 6))

	# Draw Chessboard
	chess_env.draw_chessboard(ax)
	
	# Convert Tensor to Board & Draw Pieces
	fen = chess_env.tensor_to_fen(sample_input)
	print(fen)
	board = chess.Board(fen)
	chess_env.draw_pieces(ax, board)  # Draw pieces

	# Overlay Heatmap
	heatmap_img = ax.imshow(heatmap, cmap='seismic', alpha=1.0, extent=[0, 8, 0, 8],
						 vmin=-1, vmax=1)

	# Add Colorbar
	plt.colorbar(heatmap_img, ax=ax, fraction=0.046, pad=0.04)
	plt.title(title_str)

	# Save & Show
	os.makedirs(os.path.dirname(path), exist_ok=True)
	plt.savefig(path, dpi=300, bbox_inches="tight")
	plt.close()

if __name__ == "__main__":
	# Load trained CNN model
	model = tf.keras.models.load_model("models/eda_model_epoch_075.h5")
	
	print("---EDA Model---")
	model.summary()

	# Extract feature model
	feature_model = feature_extractor(model)

	# Load dataset
	zst_file = "data/LumbrasGigaBase 2024.pgn.zst"
	dataset = chess_database.get_tf_dataset(zst_file, batch_size=100)  # Small sample for EDA

	# Iterate over 1 batch and generate heatmaps
	for _, sample_batch in enumerate(dataset.take(1)):  
		sample_input, labels, _ = sample_batch  # Ignore labels

		for i, input in enumerate(sample_input.numpy()):
			print(f"Generating heatmap for sample {i}")

			# Generate and plot heatmap, save as example_{index}.png
			plot_heatmap(feature_model, model, input, labels[i], path=f"figures/eda/heatmaps/example_{i}.png")

			if (i + 1) % 100 == 0:
				print(f"Generated {i + 1} heatmaps...")
