"""
Stockfish distillation from the Lichess evaluation database (https://database.lichess.org/#evals).

Each record is a position analysed by Stockfish at high depth with up to 5 principal variations. The
policy target spreads mass over the PVs' first moves by how close each line's expected score is to the
best; the value target is the best line's score turned into a WDL split. No engine runs locally.

	python pretraining/distill_lichess_evals.py prep  --mb 300 --out data/evals.npz
	python pretraining/distill_lichess_evals.py train --data data/evals.npz --out models/distill_6x64.keras

`prep` downloads only the first --mb megabytes of the (22 GB) zstd file with an HTTP range request.
"""
import argparse
import io
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "cpp", "mcts"))

import nash_mcts  # noqa: E402

URL = "https://database.lichess.org/lichess_db_eval.jsonl.zst"
MAX_PVS = 5
WDL_A, WDL_B = 150.0, 100.0  # W = sigmoid((cp - A) / B), L = sigmoid((-cp - A) / B), D = rest; cp for the side to move
POLICY_TAU = 0.04            # policy temperature on expected-score gaps between PVs
CASTLE_FIX = {"e1h1": "e1g1", "e1a1": "e1c1", "e8h8": "e8g8", "e8a8": "e8c8"}


def sigmoid(x):
	return 1.0 / (1.0 + math.exp(-x))


def wdl_from_pv(pv, stm_white):
	"""WDL for the side to move from one Lichess PV ({"cp": ..} or {"mate": ..}, White's point of view)."""
	if "mate" in pv:
		mate = pv["mate"] if stm_white else -pv["mate"]
		return (1.0, 0.0, 0.0) if mate > 0 else (0.0, 0.0, 1.0)
	cp = max(-2000.0, min(2000.0, float(pv["cp"]) * (1 if stm_white else -1)))
	w, l = sigmoid((cp - WDL_A) / WDL_B), sigmoid((-cp - WDL_A) / WDL_B)
	return (w, 1.0 - w - l, l)


def parse(record):
	"""-> (fen, [(policy index, prob)], (W, D, L)) or None."""
	fen = record["fen"] + " 0 1"
	ev = max(record["evals"], key=lambda e: (e.get("depth", 0), len(e["pvs"])))
	stm_white = fen.split()[1] == "w"
	legal = dict(nash_mcts.legal_moves(fen))
	moves, scores = [], []
	for pv in ev["pvs"][:MAX_PVS]:
		line = pv.get("line", "").split()
		if not line:
			continue
		uci = line[0]
		if uci not in legal:
			uci = CASTLE_FIX.get(uci, uci)
		if uci not in legal or legal[uci] in moves:
			continue
		w, d, l = wdl_from_pv(pv, stm_white)
		moves.append(legal[uci])
		scores.append(w + 0.5 * d)
	if not moves:
		return None
	best = max(scores)
	weights = [math.exp((s - best) / POLICY_TAU) for s in scores]
	total = sum(weights)
	return fen, [(m, w / total) for m, w in zip(moves, weights)], wdl_from_pv(ev["pvs"][0], stm_white)


def records(path):
	"""JSON records from a (possibly truncated) zstd stream."""
	import zstandard
	with open(path, "rb") as f:
		reader = zstandard.ZstdDecompressor().stream_reader(f, read_across_frames=True)
		tail = b""
		while True:
			try:
				chunk = reader.read(1 << 22)
			except zstandard.ZstdError:
				break  # truncated download: stop at the last complete chunk
			if not chunk:
				break
			lines = (tail + chunk).split(b"\n")
			tail = lines.pop()
			for line in lines:
				if line:
					yield json.loads(line)


def prep(args):
	raw = args.out + ".part.zst"
	os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
	if not os.path.exists(raw):
		subprocess.run(["curl", "-sSfL", "-r", f"0-{args.mb * 1024 * 1024 - 1}", "-o", raw, URL], check=True)
	fens, idx, prob, wdl = [], [], [], []
	start, skipped = time.time(), 0
	for rec in records(raw):
		out = parse(rec)
		if out is None:
			skipped += 1
			continue
		fen, policy, v = out
		fens.append(fen.encode())
		row_i = [-1] * MAX_PVS
		row_p = [0.0] * MAX_PVS
		for j, (m, p) in enumerate(policy):
			row_i[j], row_p[j] = m, p
		idx.append(row_i)
		prob.append(row_p)
		wdl.append(v)
		if len(fens) % 250_000 == 0:
			print(f"{len(fens):,} positions, {time.time() - start:.0f}s", flush=True)
	np.savez(args.out, fens=np.array(fens), idx=np.array(idx, np.int16), prob=np.array(prob, np.float32),
	         wdl=np.array(wdl, np.float32))
	os.remove(raw)
	print(f"saved {len(fens):,} positions ({skipped} skipped) to {args.out}", flush=True)


def train(args):
	import tensorflow as tf
	from tensorflow import keras
	for gpu in tf.config.list_physical_devices("GPU"):
		tf.config.experimental.set_memory_growth(gpu, True)
	keras.mixed_precision.set_global_policy("mixed_float16")
	from core import model_framework

	data = np.load(args.data)
	fens, idx, prob, wdl = data["fens"], data["idx"], data["prob"], data["wdl"]
	n = len(fens)
	order = np.random.default_rng(0).permutation(n)
	val, tr = order[:args.val], order[args.val:]
	steps = len(tr) // args.batch

	def batches(rows, shuffle):
		def gen():
			rng = np.random.default_rng()
			while True:
				r = rng.permutation(rows) if shuffle else rows
				for s in range(0, len(r) - args.batch + 1, args.batch):
					b = np.sort(r[s:s + args.batch])
					x = nash_mcts.encode_batch([f.decode() for f in fens[b]])
					p = np.zeros((len(b), nash_mcts.POLICY_SIZE), np.float32)
					ii, pp = idx[b], prob[b]
					rr = np.repeat(np.arange(len(b)), MAX_PVS).reshape(len(b), MAX_PVS)
					m = ii >= 0
					p[rr[m], ii[m]] = pp[m]
					yield x, {"policy": p, "value": wdl[b]}
		sig = (tf.TensorSpec((None, 8, 8, 36), tf.float32),
		       {"policy": tf.TensorSpec((None, nash_mcts.POLICY_SIZE), tf.float32), "value": tf.TensorSpec((None, 3), tf.float32)})
		return tf.data.Dataset.from_generator(gen, output_signature=sig).prefetch(16)

	class Progress(keras.callbacks.Callback):
		"""Writes training progress (step, running metrics, per-epoch history) to a JSON file for live dashboards."""
		def __init__(self, path, every=25):
			super().__init__()
			self.path, self.every, self.history, self.points = path, every, [], []
			self.start = time.time()

		def _write(self, epoch, step, logs):
			state = {"epoch": epoch, "epochs": args.epochs, "step": step, "steps": steps, "elapsed": time.time() - self.start,
			         "train": {k: float(v) for k, v in (logs or {}).items()}, "history": self.history, "points": self.points,
			         "params": model.count_params(), "batch": args.batch, "train_positions": len(tr), "val_positions": len(val)}
			tmp = self.path + ".tmp"
			with open(tmp, "w") as f:
				json.dump(state, f)
			os.replace(tmp, self.path)

		def on_epoch_begin(self, epoch, logs=None):
			self.epoch = epoch

		def on_train_batch_end(self, batch, logs=None):
			if (batch + 1) % self.every == 0:
				global_step = self.epoch * steps + batch + 1
				self.points.append([global_step, float(logs["loss"]), float(logs.get("policy_top1", 0)), float(logs.get("policy_top3", 0))])
				self._write(self.epoch, batch + 1, logs)

		def on_epoch_end(self, epoch, logs=None):
			self.history.append({k: float(v) for k, v in logs.items()})
			self._write(epoch + 1, 0, {})

	model = model_framework.exploratory_model(FILTERS=args.filters, BLOCKS=args.blocks, dropout=0.0, policy_head=True)
	schedule = keras.optimizers.schedules.CosineDecay(args.lr, decay_steps=steps * args.epochs, alpha=0.02,
	                                                  warmup_target=None)
	model.compile(
		optimizer=keras.optimizers.AdamW(learning_rate=schedule, weight_decay=1e-4),
		loss={"policy": keras.losses.CategoricalCrossentropy(from_logits=True), "value": keras.losses.CategoricalCrossentropy()},
		metrics={"policy": [keras.metrics.CategoricalAccuracy(name="top1"), keras.metrics.TopKCategoricalAccuracy(k=3, name="top3")]},
	)
	print(f"{model.count_params():,} params; {len(tr):,} train / {len(val):,} val positions; {steps} steps per epoch", flush=True)
	for path in (args.out, args.progress):
		if path:
			os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
	# Checkpoint after every epoch (weights + optimizer state), so a crash costs at most one epoch: --resume picks it up.
	last = args.out.replace(".keras", ".last.keras")
	initial_epoch = 0
	if args.resume and os.path.exists(last):
		model = keras.models.load_model(last)
		initial_epoch = int(model.optimizer.iterations.numpy()) // steps
		print(f"resuming from {last} at epoch {initial_epoch + 1}", flush=True)
	callbacks = [keras.callbacks.ModelCheckpoint(last)]
	if args.progress:
		callbacks.append(Progress(args.progress))
	model.fit(batches(tr, True), steps_per_epoch=steps, epochs=args.epochs, initial_epoch=initial_epoch,
	          validation_data=batches(val, False), validation_steps=len(val) // args.batch, callbacks=callbacks, verbose=2)
	model.save(args.out)
	if os.path.exists(last):
		os.remove(last)
	print(f"saved {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)", flush=True)


def main():
	parser = argparse.ArgumentParser()
	sub = parser.add_subparsers(dest="cmd", required=True)
	p = sub.add_parser("prep")
	p.add_argument("--mb", type=int, default=300)
	p.add_argument("--out", default="data/evals.npz")
	t = sub.add_parser("train")
	t.add_argument("--data", default="data/evals.npz")
	t.add_argument("--out", default="models/distill_6x64.keras")
	t.add_argument("--blocks", type=int, default=6)
	t.add_argument("--filters", type=int, default=64)
	t.add_argument("--epochs", type=int, default=6)
	t.add_argument("--batch", type=int, default=1024)
	t.add_argument("--lr", type=float, default=2e-3)
	t.add_argument("--val", type=int, default=20_480)  # a whole number of batches
	t.add_argument("--resume", action="store_true", help="continue from the per-epoch checkpoint next to --out")
	t.add_argument("--progress", default=None, help="JSON file updated during training (for the play UI's /train page)")
	args = parser.parse_args()
	prep(args) if args.cmd == "prep" else train(args)


if __name__ == "__main__":
	main()
