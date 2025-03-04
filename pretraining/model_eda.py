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
	history = model.fit(dataset, epochs=50, steps_per_epoch=100000, callbacks=[checkpoint_callback, csv_logger, plot_callback])

	# Save model
	model.save("models/eda_model.h5")