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
	n_blocks = 3
	n_heads = 8
	n_dim = 128
	ff_dim = 128
	model = model_framework.create_chess_transformer(n_blocks, n_heads, n_dim, ff_dim, dropout_rate=0.1, lr=5e-4)
	model_name = f"transformer_{n_blocks}x{n_heads}-n_dim={n_dim}-ff_dim-{ff_dim}"

	zst_file = "data/LumbrasGigaBase 2024.pgn.zst"
	dataset = chess_database.get_tf_dataset(zst_file, batch_size=64).repeat()
	print(dataset.take(1))

	os.makedirs("models", exist_ok=True)
	os.makedirs("figures", exist_ok=True)  # Ensure figures directory exists

	model.summary()

	checkpoint_callback = ModelCheckpoint(f"models/{model_name}" + "_{epoch:03d}.h5", save_freq="epoch", save_weights_only=False)
	csv_logger = CSVLogger("logs/training_log.csv", append=True)
	plot_callback = model_framework.TrainingPlotCallback(save_interval=1, plot_path=f"figures/training/{model_name}_training.png")

	# Train model and capture history
	history = model.fit(dataset, epochs=1, steps_per_epoch=1000, callbacks=[checkpoint_callback, csv_logger, plot_callback], verbose=1)

	# Save model
	model.save(f"models/{model_name}.h5")