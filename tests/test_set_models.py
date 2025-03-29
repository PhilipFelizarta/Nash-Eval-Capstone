import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import core.chess_database as chess_database
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tensorflow.keras.losses import SparseCategoricalCrossentropy


if __name__ == "__main__":
	# Load dataset
	# Parameters
	PGN_FILE = "archive_data/LumbrasGigaBase 2024.pgn"
	MODEL_FOLDER = "models/final_model"
	BATCH_SIZE = 64
	MAX_BATCHES = 1000  # You can change this if needed

	# Load test dataset once
	print("⏳ Loading test set...")
	test_dataset = chess_database.get_tf_testset_from_end(PGN_FILE, batch_size=BATCH_SIZE, max_batches=MAX_BATCHES)

	# Cache the test set into memory (optional, makes reuse easier)
	print("⚡ Caching test set...")
	total_batches = 0.0 
	total_samples = 0.0
	x_all, y_all, w_all = [], [], []
	for x, y, w in test_dataset:
		batch_size_actual = x.shape[0]
		total_batches += 1
		total_samples += batch_size_actual
		x_all.append(x.numpy())
		y_all.append(y.numpy())
		w_all.append(w.numpy())

	x_all = np.concatenate(x_all, axis=0)
	y_all = np.concatenate(y_all, axis=0)
	w_all = np.concatenate(w_all, axis=0)

	print(f"✅ Total batches: {total_batches}")
	print(f"✅ Total samples: {total_samples}")

	print("Label distribution in test set:", np.bincount(y_all))
	print(f"✅ Loaded {x_all.shape[0]} samples.")

	# Get all model paths
	model_paths = sorted([os.path.join(MODEL_FOLDER, f) for f in os.listdir(MODEL_FOLDER) if f.endswith(".h5")])

	results = []

	loss_fn = SparseCategoricalCrossentropy(from_logits=True, reduction='none')

	for model_path in model_paths:
		model_name = os.path.basename(model_path)
		model = keras.models.load_model(model_path)

		# Predict and evaluate
		y_pred_logits = model.predict(x_all, batch_size=64, verbose=0)
		y_pred = np.argmax(y_pred_logits, axis=-1)

		# Accuracy and metrics
		acc = accuracy_score(y_all, y_pred)
		prec, rec, f1, _ = precision_recall_fscore_support(y_all, y_pred, average='macro')

		# NEW: Importance-weighted accuracy
		correct = (y_pred == y_all).astype(np.float32)
		weighted_acc = np.sum(correct * w_all) / np.sum(w_all)

		# Compute per-sample loss (no reduction)
		per_sample_loss = loss_fn(y_all, y_pred_logits).numpy()
		test_loss = np.mean(per_sample_loss)
		weighted_loss = np.sum(per_sample_loss * w_all) / np.sum(w_all)

		# Per-class scores
		prec_class, rec_class, f1_class, support_class = precision_recall_fscore_support(
			y_all, y_pred, labels=[0, 1, 2], average=None
		)

		# Total importance per class
		class_weights = np.array([
			np.sum(w_all[y_all == 0]),
			np.sum(w_all[y_all == 1]),
			np.sum(w_all[y_all == 2])
		])
		total_importance = np.sum(class_weights)

		# Weighted precision/recall/f1
		weighted_prec = np.sum(prec_class * class_weights) / total_importance
		weighted_rec = np.sum(rec_class * class_weights) / total_importance
		weighted_f1 = np.sum(f1_class * class_weights) / total_importance

		model_dict = {
			"model": model_name,
			"accuracy": acc,
			"weighted_accuracy": weighted_acc,
			"loss": test_loss,
			"weighted_loss": weighted_loss,
			"precision": prec,
			"recall": rec,
			"f1_score": f1,
			"weighted_precision": weighted_prec,
			"weighted_recall": weighted_rec,
			"weighted_f1_score": weighted_f1
		}

		print(model_dict)
		results.append(model_dict)

	# Save results to CSV
	df_results = pd.DataFrame(results)
	df_results.to_csv("figures/model_evaluation_results.csv", index=False)
	print("Saved results to model_evaluation_results.csv")