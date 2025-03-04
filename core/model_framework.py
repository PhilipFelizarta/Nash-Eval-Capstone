from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import Callback
import tensorflow as tf

import os
import matplotlib.pyplot as plt
import numpy as np

class TrainingPlotCallback(Callback):
	""" Callback to update the training plot every 10 epochs """
	def __init__(self, save_interval=10, plot_path="figures/eda_model_training.png", log_file="logs/training_log.csv"):
		super(TrainingPlotCallback, self).__init__()
		self.save_interval = save_interval
		self.plot_path = plot_path
		self.log_file = log_file
		
		# Ensure directories exist
		os.makedirs("models", exist_ok=True)
		os.makedirs("figures", exist_ok=True)
		os.makedirs("logs", exist_ok=True)

	def on_epoch_end(self, epoch, logs=None):
		if (epoch + 1) % self.save_interval == 0:  # Save every `save_interval` epochs
			# Load CSV log
			if os.path.exists(self.log_file):
				log_data = np.genfromtxt(self.log_file, delimiter=',', skip_header=1)
				epochs = log_data[:, 0] if log_data.ndim > 1 else np.array([log_data[0]])
				loss = log_data[:, 1] if log_data.ndim > 1 else np.array([log_data[1]])
				accuracy = log_data[:, 2] if log_data.ndim > 1 else np.array([log_data[2]])
			else:
				return  # No data to plot yet

			
			# Create subplots for Loss and Accuracy
			fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

			# Plot Loss
			axes[0].plot(epochs, loss, label="Loss", color='red', marker='o')
			axes[0].set_xlabel("Epochs")
			axes[0].set_ylabel("Loss")
			axes[0].set_title("Training Loss Over Epochs")
			axes[0].legend()
			axes[0].grid(True)

			# Plot Accuracy
			axes[1].plot(epochs, accuracy, label="Accuracy", color='blue', marker='s')
			axes[1].set_xlabel("Epochs")
			axes[1].set_ylabel("Accuracy")
			axes[1].set_title("Training Accuracy Over Epochs")
			axes[1].legend()
			axes[1].grid(True)

			# Adjust layout and save the plot
			plt.tight_layout()
			plt.savefig(self.plot_path, dpi=300, bbox_inches="tight")
			plt.close()
			print(f"Updated training plot saved at {self.plot_path}")


def exploratory_model(FILTERS=64, BLOCKS=4, SE_CHANNELS=16, lr=3e-3, dropout=0.3):
	inputs = keras.Input(shape=(8, 8, 19))

	# **Input Convolution**
	x = layers.Conv2D(FILTERS, (3, 3), padding="same", activation=None)(inputs)
	x = layers.BatchNormalization()(x)
	x = layers.ReLU()(x)
	x = layers.Dropout(dropout / 2)(x)  # Light dropout in early layers

	# **Residual Tower with SE Layers**
	def residual_block(x):
		skip = x  # Save for skip connection

		# Conv 1
		x = layers.Conv2D(FILTERS, (3, 3), padding="same", activation=None)(x)
		x = layers.BatchNormalization()(x)
		x = layers.ReLU()(x)
		x = layers.Dropout(dropout / 2)(x)  # Light dropout

		# Conv 2
		x = layers.Conv2D(FILTERS, (3, 3), padding="same", activation=None)(x)
		x = layers.BatchNormalization()(x)

		# **Fixed SE Layer**
		se = layers.GlobalAveragePooling2D()(x)  # Shape: (BATCH, FILTERS)
		se = layers.Dense(SE_CHANNELS, activation="relu")(se)
		se = layers.Dense(FILTERS, activation="sigmoid")(se)  # Match FILTERS, not FILTERS*2
		se = layers.Reshape((1, 1, FILTERS))(se)  # Expand dims for broadcasting
		x = layers.Multiply()([x, se])  # Ensure broadcastable shapes

		# **Skip Connection**
		if x.shape[-1] != skip.shape[-1]:  # Ensure matching shapes
			skip = layers.Conv2D(FILTERS, (1, 1), padding="same", activation=None)(skip)
		x = layers.Add()([skip, x])
		x = layers.ReLU()(x)
		x = layers.Dropout(dropout / 2)(x)  # Light dropout
		return x

	# Apply residual blocks
	for _ in range(BLOCKS):
		x = residual_block(x)

	# **Value Head**
	v = layers.Conv2D(32, (1, 1), padding="same", activation=None, name="last_conv")(x)
	v = layers.BatchNormalization()(v)
	v = layers.ReLU()(v)
	
	v = layers.Conv2D(1, (1, 1), padding="same", activation=None)(v)
	v = layers.BatchNormalization()(v)
	v = layers.ReLU()(v)
	v = layers.Flatten()(v)
	v = layers.Dropout(dropout)(v)  # Dropout before dense layer
	v = layers.Dense(128, activation="relu")(v)
	v = layers.Dropout(dropout)(v)  # Dropout after dense layer
	outputs = layers.Dense(3, activation="softmax")(v)  # VALUE_WDL format

	# **Build Model**
	model = keras.Model(inputs, outputs)
	model.compile(
		optimizer=keras.optimizers.Adam(learning_rate=lr),
		loss=keras.losses.SparseCategoricalCrossentropy(),
		metrics=['accuracy']
	)

	return model