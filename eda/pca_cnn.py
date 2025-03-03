import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import core.chess_database as chess_database  # Ensure this module is available

if __name__ == "__main__":
	# Load trained CNN model
	model = tf.keras.models.load_model("models/eda_model.h5")
	
	print("---EDA Model---")
	model.summary()

	# Create a feature extractor by removing the last classification layer
	feature_extractor = tf.keras.Model(
		inputs=model.input,
		outputs=model.get_layer("global_average_pooling2d").output  # Extract from GAP layer
	)

	print("---Feature Extractor---")
	feature_extractor.summary()

	# Load dataset
	zst_file = "data/LumbrasGigaBase 2024.pgn.zst"
	dataset = chess_database.get_tf_dataset(zst_file, batch_size=256).take(100)  # Small sample for EDA

	# Extract features and labels
	feature_list, label_list, importance_list = [], [], []

	for batch in dataset:
		inputs, labels, importance = batch
		features = feature_extractor(inputs, training=False).numpy()  # Get CNN embeddings
		feature_list.append(features)
		label_list.append(labels.numpy())
		importance_list.append(importance.numpy())

	# Convert to arrays
	features = np.vstack(feature_list)
	labels = np.concatenate(label_list)
	importance_weights = np.concatenate(importance_list)

	# Apply PCA with importance-weighted mean centering
	pca = PCA(n_components=2)
	weighted_features = features * importance_weights[:, None]  # Apply importance scaling
	pca_result = pca.fit_transform(weighted_features)

	# Plot PCA visualization
	plt.figure(figsize=(10, 6))
	sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=labels, alpha=0.7, palette="viridis", edgecolor=None)
	plt.xlabel("PCA Component 1")
	plt.ylabel("PCA Component 2")
	plt.title("PCA of CNN Extracted Features (Importance Weighted)")
	plt.legend(title="Labels")
	plt.grid()
	plt.savefig("figures/pca_analysis/pca_cnn.png", dpi=300, bbox_inches="tight")