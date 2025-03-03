from tensorflow import keras
from tensorflow.keras import layers, Model

def exploratory_model(dropout=0.3, lr=3e-3):
	inputs = keras.Input(shape=(8, 8, 19))

	# Convolutional feature extraction
	x = layers.Conv2D(32, (5, 5), padding="same", activation=None)(inputs)
	x = layers.BatchNormalization()(x)
	x = layers.ReLU()(x)

	x = layers.Conv2D(64, (3, 3), padding="same", activation=None)(x)
	x = layers.BatchNormalization()(x)
	x = layers.ReLU()(x)
	
	# Squeeze-and-Excitation (SE) Block
	se = layers.GlobalAveragePooling2D()(x)
	se = layers.Dense(64 // 8, activation="relu")(se)
	se = layers.Dense(64, activation="sigmoid")(se)
	x = layers.Multiply()([x, se])

	x = layers.Conv2D(128, (3, 3), padding="same", activation=None)(x)
	x = layers.BatchNormalization()(x)
	x = layers.ReLU()(x)
	
	# Residual Connection
	shortcut = layers.Conv2D(128, (1, 1), padding="same")(inputs)
	x = layers.Add()([x, shortcut])

	x = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)
	x = layers.GlobalAveragePooling2D()(x)  # GAP for interpretability

	x = layers.Dropout(dropout)(x)

	outputs = layers.Dense(3, activation="softmax")(x)  # Final classification

	model = Model(inputs, outputs)
	model.compile(
		optimizer=keras.optimizers.Adam(learning_rate=lr),
		loss=keras.losses.SparseCategoricalCrossentropy(),
		metrics=['accuracy']
	)
	
	return model