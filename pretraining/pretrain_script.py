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
	# 🚀 Enable Multi-GPU Strategy
	strategy = keras.mixed_precision.set_global_policy("float32")
	strategy = keras.distribute.MirroredStrategy()

	with strategy.scope():  # Ensures model is distributed across all GPUs
		n_blocks = int(os.getenv("N_BLOCKS"))
		n_heads = int(os.getenv("N_HEADS"))
		n_dim = int(os.getenv("N_DIM"))
		ff_dim = int(os.getenv("FF_DIM"))

		model = model_framework.create_chess_transformer(
			n_blocks, n_heads, n_dim, ff_dim, dropout_rate=0.1, lr=1e-5
		)

	model_name = f"transformer_{n_blocks}x{n_heads}-n_dim={n_dim}-ff_dim-{ff_dim}"

	# ✅ Increase batch size based on GPU count
	batch_size_per_gpu = 64
	num_gpus = strategy.num_replicas_in_sync  # Detects number of GPUs
	batch_size = batch_size_per_gpu * num_gpus

	zst_file = "data/LumbrasGigaBase 2024.pgn.zst"
	dataset = chess_database.get_tf_dataset(zst_file, batch_size=batch_size).repeat()
	
	print(f"🚀 Using {num_gpus} GPUs! Adjusted batch size: {batch_size}")
	print(dataset.take(1))

	os.makedirs("models", exist_ok=True)
	os.makedirs(f"models/{model_name}", exist_ok=True)
	os.makedirs("figures", exist_ok=True)

	model.summary()

	checkpoint_callback = ModelCheckpoint(
		f"models/{model_name}/{model_name}" + "_{epoch:03d}.h5",
		save_freq="epoch", save_weights_only=False
	)
	csv_logger = CSVLogger("logs/training_log.csv", append=True)
	plot_callback = model_framework.TrainingPlotCallback(
		save_interval=1, plot_path=f"figures/training/{model_name}_training.png"
	)

	# ✅ Start training with Multi-GPU strategy
	history = model.fit(
		dataset,
		epochs=1000,
		steps_per_epoch=10000,
		callbacks=[checkpoint_callback, csv_logger, plot_callback],
		verbose=2
	)

	# Save the trained model
	model.save(f"models/{model_name}.h5")