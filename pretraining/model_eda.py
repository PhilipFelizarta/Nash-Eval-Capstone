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
import core.model_framework as model_framework
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

def generate_heatmap(model, img, class_idx):
	"""
	Generates a heatmap using GAP weights for interpretability.
	"""
	# Extract the output of the last convolutional layer
	last_conv_layer = model.layers[-3]  # GAP is directly after this layer
	last_conv_model = Model(model.input, last_conv_layer.input)
	
	# Compute feature maps from last conv layer
	last_conv_output = last_conv_model.predict(img[np.newaxis, :, :, :])  # Add batch dim
	last_conv_output = np.squeeze(last_conv_output)  # Shape: (8, 8, 64)

	# Get the final Dense layer's weights (GAP → Dense)
	final_dense_weights = model.layers[-1].weights[0].numpy()  # Shape: (64, 3)
	final_dense_weights_for_class = final_dense_weights[:, class_idx]  # Shape: (64,)

	# Compute the weighted sum of feature maps
	heat_map = np.dot(last_conv_output, final_dense_weights_for_class)


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
		heatmap, cmap='Reds', alpha=0.9, extent=[0, 8, 0, 8]
	)

	# Add Colorbar
	plt.colorbar(heatmap_img, ax=ax, fraction=0.046, pad=0.04)
	plt.title(f"Label: {true_class} -- Pred:{pred_label}")
	
	# Save & Show
	os.makedirs(os.path.dirname(path), exist_ok=True)
	plt.savefig(path, dpi=300, bbox_inches="tight")
	plt.close()

if __name__ == "__main__":
	model = model_framework.exploratory_model()

	zst_file = "data/LumbrasGigaBase 2024.pgn.zst"
	dataset = chess_database.get_tf_dataset(zst_file, batch_size=64).repeat()

	os.makedirs("models", exist_ok=True)
	os.makedirs("figures", exist_ok=True)  # Ensure figures directory exists

	model.summary()

	checkpoint_callback = ModelCheckpoint("models/eda_model_epoch_{epoch:03d}.h5", save_freq="epoch", save_weights_only=False)
	csv_logger = CSVLogger("logs/training_log.csv", append=True)
	plot_callback = model_framework.TrainingPlotCallback(save_interval=1, plot_path="figures/training/eda_model_training.png")

	# Train model and capture history
	history = model.fit(dataset, epochs=25, steps_per_epoch=100000, callbacks=[checkpoint_callback, csv_logger, plot_callback])

	# Save model
	model.save("models/eda_model.h5")

	# Extract loss and accuracy from history
	loss = history.history['loss']
	accuracy = history.history['accuracy']
	epochs = np.arange(1, len(loss) + 1)

	# Plot training loss and accuracy
	plt.figure(figsize=(10, 6))
	plt.plot(epochs, loss, label="Loss", color='red', marker='o')
	plt.plot(epochs, accuracy, label="Accuracy", color='blue', marker='s')

	plt.xlabel("Epochs")
	plt.ylabel("Metric Value")
	plt.title("Training Loss & Accuracy Over Epochs")
	plt.legend()
	plt.grid(True)

	# Save the training history plot
	training_plot_path = "figures/training/eda_model_training.png"
	plt.savefig(training_plot_path, dpi=300, bbox_inches="tight")
	plt.close()

	print(f"✅ Training history saved to {training_plot_path}")

	# Iterate over 1000 samples and generate heatmaps
	for i, sample_batch in enumerate(dataset.take(100)):  
		sample_input, sample_label, _ = sample_batch

		# Convert tensor to NumPy array
		sample_input = sample_input.numpy()[0]  # Extract first example
		true_class = int(sample_label.numpy()[0])  # Extract true class

		# Make prediction
		pred_probs = model.predict(sample_input[np.newaxis, ...])  # Add batch dim
		pred_label = np.argmax(pred_probs)  # Get the highest probability class

		# Generate and plot heatmap, save as example_{index}.png
		plot_heatmap(model, sample_input, true_class, pred_label, path=f"figures/eda/heatmaps/example_{i}.png")

		if (i + 1) % 100 == 0:
			print(f"Generated {i + 1} heatmaps...")