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
	n_filters = int(os.getenv("N_FILTERS"))
	lr = float(os.getenv("LR", "3e-4"))
	resume_from = os.getenv("RESUME_FROM")  # Checkpoint to continue training from (optional)
	resume_lr = float(os.getenv("RESUME_LR", "3e-5"))
	skip_games = int(os.getenv("SKIP_GAMES", "0"))  # Games the previous run already consumed

	strategy = tf.distribute.MirroredStrategy()
	BATCH_SIZE = 64
	GLOBAL_BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync

	print(f"Number of GPUS: {strategy.num_replicas_in_sync}\nBatch Size: {GLOBAL_BATCH_SIZE}")

	zst_file = "data/LumbrasGigaBase 2024.pgn.zst"
	dataset = chess_database.get_tf_dataset(zst_file, batch_size=GLOBAL_BATCH_SIZE, skip_games=skip_games).repeat()

	model_name = f"RESNET_36p_{n_blocks}x{n_filters}" + ("_PART2" if resume_from else "")
	print(dataset.take(1))

	os.makedirs("models", exist_ok=True)
	os.makedirs(f"models/{model_name}", exist_ok=True)
	os.makedirs("figures", exist_ok=True)

	with strategy.scope():
		if resume_from:
			if not os.path.exists(resume_from):
				raise FileNotFoundError(f"RESUME_FROM checkpoint not found: {resume_from}")
			print(f"Loading existing model from {resume_from}, skipping the first {skip_games} games")
			model = keras.models.load_model(resume_from)

			optimizer = keras.optimizers.Adam(learning_rate=resume_lr)  # Lower learning rate.
			model.compile(optimizer=optimizer,
				 loss=keras.losses.SparseCategoricalCrossentropy(),
				 metrics=['accuracy'],
				 weighted_metrics=['accuracy'])
		else:
			print("Creating new model.")
			model = model_framework.exploratory_model(
				FILTERS=n_filters, BLOCKS=n_blocks, dropout=0.25, lr=lr
			)

	model.summary()

	checkpoint_callback = ModelCheckpoint(
		f"models/{model_name}/{model_name}" + "_{epoch:03d}.h5",
		save_freq="epoch", save_weights_only=False
	)
	log_file = f"logs/{model_name}_training_log.csv"
	csv_logger = CSVLogger(log_file, append=True)
	plot_callback = model_framework.TrainingPlotCallback(
		save_interval=1, plot_path=f"figures/training/{model_name}_training.png", log_file=log_file
	)

	history = model.fit(
		dataset,
		epochs=1000,
		steps_per_epoch=100000,
		callbacks=[checkpoint_callback, csv_logger, plot_callback],
		verbose=2
	)

	# Save the trained model
	model.save(f"models/{model_name}.h5")