import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Model
import tensorflow as tf
import chess

import core.chess_database as chess_database
import core.chess_environment as chess_env
import core.model_framework as model_framework
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

if __name__ == "__main__":
	n_blocks = int(os.getenv("N_BLOCKS"))
	n_heads = int(os.getenv("N_HEADS"))
	n_dim = int(os.getenv("N_DIM"))
	ff_dim = int(os.getenv("FF_DIM"))

	strategy = tf.distribute.MirroredStrategy()
	BATCH_SIZE = 64
	GLOBAL_BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync

	print(f"Number of GPUS: {strategy.num_replicas_in_sync}\nBatch Size: {GLOBAL_BATCH_SIZE}")

	zst_file = "data/LumbrasGigaBase 2024.pgn.zst"
	dataset = chess_database.get_tf_dataset(zst_file, batch_size=GLOBAL_BATCH_SIZE).repeat()

	model_name = f"transformer_flat_{n_blocks}x{n_heads}-n_dim={n_dim}-ff_dim-{ff_dim}"
	print(dataset.take(1))

	os.makedirs("models", exist_ok=True)
	os.makedirs(f"models/{model_name}", exist_ok=True)
	os.makedirs("figures", exist_ok=True)

	with strategy.scope():
		model = model_framework.create_chess_transformer(
			n_blocks, n_heads, n_dim, ff_dim, dropout_rate=0.1, lr=1e-5
		)

	model.summary()

	checkpoint_callback = ModelCheckpoint(
		f"models/{model_name}/{model_name}" + "_{epoch:03d}.h5",
		save_freq="epoch", save_weights_only=False
	)
	csv_logger = CSVLogger("logs/training_log.csv", append=True)
	plot_callback = model_framework.TrainingPlotCallback(
		save_interval=1, plot_path=f"figures/training/{model_name}_training.png"
	)

	history = model.fit(
		dataset,
		epochs=1000,
		steps_per_epoch=10000,
		callbacks=[checkpoint_callback, csv_logger, plot_callback],
		verbose=2
	)

	# Save the trained model
	model.save(f"models/{model_name}.h5")