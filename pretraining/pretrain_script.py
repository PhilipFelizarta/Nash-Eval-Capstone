import tensorflow as tf
from tensorflow import keras
import core.chess_database as chess_database

if __name__ == "__main__":
	model = keras.Sequential([
		keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(8,8,17)),
		keras.layers.Conv2D(64, (3,3), activation='relu'),
		keras.layers.Flatten(),
		keras.layers.Dense(128, activation='relu'),
		keras.layers.Dense(3, activation='softmax')  # 3 categories: {0: Draw, 1: White Wins, 2: Black Wins}
	])

	model.compile(
		optimizer=keras.optimizers.Adam(),
		loss=keras.losses.SparseCategoricalCrossentropy(),
		metrics=['accuracy']
	)

	zst_file = "data/lichess_db_standard_rated_2025-02.pgn.zst"
	dataset = chess_database.get_tf_dataset(zst_file, batch_size=4096)

	model.fit(dataset, epochs=10, steps_per_epoch=1000)