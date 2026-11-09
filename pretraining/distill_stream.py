"""
Stockfish distillation from the whole Lichess evaluation database, streamed. distill_lichess_evals.py trains on a
prefix that fits in RAM; this script never holds more than two shards.

	python pretraining/distill_stream.py prep  --out data/evals_full --workers 12
	python pretraining/distill_stream.py train --data data/evals_full --out out/v3/distill_8x96.keras --hours 5.75

`prep` streams the zstd file (curl | zstandard), parses records in --workers processes (same targets as
distill_lichess_evals.py) and writes shards of --shard positions to OUT/shard_XXXXXX.npz.

`train` can start as soon as prep has written its first shards. It loads two shards at a time, always picking
among the least-used shards present, so it moves on to new data as it arrives instead of repeating early shards.
The learning rate follows a cosine over wall-clock time, so the run ends after --hours whatever the throughput.
Validation is the first --val positions of shard --val-shard (beyond the 4M-position prefix the 6x64 v2 net was
trained on, so both can be scored on it); those positions are never trained on.
"""
import argparse
import glob
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
from pretraining.distill_lichess_evals import MAX_PVS, URL, parse  # noqa: E402


def parse_lines(lines):
	fens, idx, prob, wdl = [], [], [], []
	for line in lines:
		out = parse(json.loads(line))
		if out is None:
			continue
		fen, policy, v = out
		fens.append(fen.encode())
		row_i, row_p = [-1] * MAX_PVS, [0.0] * MAX_PVS
		for j, (m, p) in enumerate(policy):
			row_i[j], row_p[j] = m, p
		idx.append(row_i)
		prob.append(row_p)
		wdl.append(v)
	return (np.array(fens, "S100"), np.array(idx, np.int16).reshape(-1, MAX_PVS),
	        np.array(prob, np.float32).reshape(-1, MAX_PVS), np.array(wdl, np.float32).reshape(-1, 3), len(lines) - len(fens))


def prep(args):
	import zstandard
	from multiprocessing import Pool

	os.makedirs(args.out, exist_ok=True)
	curl = subprocess.Popen(["curl", "-sSfL", URL], stdout=subprocess.PIPE)
	reader = zstandard.ZstdDecompressor().stream_reader(curl.stdout, read_across_frames=True)

	def chunks():
		tail = b""
		while True:
			data = reader.read(1 << 24)
			if not data:
				break
			lines = (tail + data).split(b"\n")
			tail = lines.pop()
			yield [l for l in lines if l]
		if tail.strip():
			yield [tail]

	keys = ("fens", "idx", "prob", "wdl")
	buf = {k: [] for k in keys}  # lists of parsed chunks
	buffered = shard = total = skipped = 0
	start = time.time()

	def write(n):
		nonlocal shard, total, buffered
		path = os.path.join(args.out, f"shard_{shard:06d}.npz")
		arrays = {k: np.concatenate(buf[k]) for k in keys}
		np.savez(path + ".tmp.npz", **{k: a[:n] for k, a in arrays.items()})
		os.replace(path + ".tmp.npz", path)
		for k in keys:
			buf[k] = [arrays[k][n:]]
		buffered -= n
		shard += 1
		total += n
		rate = total / (time.time() - start)
		print(f"shard {shard - 1}: {total:,} positions ({skipped:,} skipped), {rate:,.0f}/s", flush=True)
		with open(os.path.join(args.out, "prep.json"), "w") as f:
			json.dump({"shards": shard, "positions": total, "skipped": skipped, "positions_per_s": rate,
			           "elapsed": time.time() - start, "done": False}, f)

	with Pool(args.workers) as pool:
		pending = []
		it = chunks()
		exhausted = False
		while pending or not exhausted:
			# At most 2 chunks per worker in flight, so a fast download cannot pile up in memory.
			while not exhausted and len(pending) < 2 * args.workers:
				try:
					pending.append(pool.apply_async(parse_lines, (next(it),)))
				except StopIteration:
					exhausted = True
			*parsed, skip = pending.pop(0).get()
			skipped += skip
			for k, a in zip(keys, parsed):
				buf[k].append(a)
			buffered += len(parsed[0])
			while buffered >= args.shard:
				write(args.shard)
			if args.max_positions and total >= args.max_positions:
				break
	if buffered and not (args.max_positions and total >= args.max_positions):
		write(buffered)
	curl.kill()
	with open(os.path.join(args.out, "prep.json"), "w") as f:
		json.dump({"shards": shard, "positions": total, "skipped": skipped, "elapsed": time.time() - start,
		           "done": True}, f)
	print(f"done: {total:,} positions in {shard} shards", flush=True)


def to_xy(fens, idx, prob, wdl):
	x = nash_mcts.encode_batch([f.decode() for f in fens])
	p = np.zeros((len(fens), nash_mcts.POLICY_SIZE), np.float32)
	rr = np.repeat(np.arange(len(fens)), MAX_PVS).reshape(len(fens), MAX_PVS)
	m = idx >= 0
	p[rr[m], idx[m]] = prob[m]
	return x, {"policy": p, "value": wdl}


def train(args):
	import tensorflow as tf
	from tensorflow import keras
	for gpu in tf.config.list_physical_devices("GPU"):
		tf.config.experimental.set_memory_growth(gpu, True)
	keras.mixed_precision.set_global_policy("mixed_float16")
	from core import model_framework

	def shard_path(i):
		return os.path.join(args.data, f"shard_{i:06d}.npz")

	print(f"waiting for {shard_path(args.val_shard)}", flush=True)
	while not os.path.exists(shard_path(args.val_shard)):
		time.sleep(10)
	d = np.load(shard_path(args.val_shard))
	val = tuple(d[k][:args.val] for k in ("fens", "idx", "prob", "wdl"))

	os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
	last = args.out.replace(".keras", ".last.keras")
	state_path = args.out.replace(".keras", ".state.json")
	state = {"elapsed": 0.0, "uses": {}}
	if args.resume and os.path.exists(last) and os.path.exists(state_path):
		with open(state_path) as f:
			state = json.load(f)
	uses = state["uses"]

	def load(name):
		d = np.load(os.path.join(args.data, name))
		arrays = [d[k] for k in ("fens", "idx", "prob", "wdl")]
		if name == os.path.basename(shard_path(args.val_shard)):
			arrays = [a[args.val:] for a in arrays]
		return arrays

	def pick(rng):
		for f in glob.glob(os.path.join(args.data, "shard_*.npz")):
			if not f.endswith(".tmp.npz"):
				uses.setdefault(os.path.basename(f), 0)
		low = min(uses.values())
		name = rng.choice(sorted(n for n, u in uses.items() if u == low))
		uses[name] += 1
		return name

	def batches(training):
		def gen():
			rng = np.random.default_rng()
			while True:
				if training:
					parts = [load(pick(rng)) for _ in range(2)]
					fens, idx, prob, wdl = (np.concatenate(a) for a in zip(*parts))
					rows = rng.permutation(len(fens))
				else:
					fens, idx, prob, wdl = val
					rows = np.arange(len(fens))
				for s in range(0, len(rows) - args.batch + 1, args.batch):
					b = np.sort(rows[s:s + args.batch])
					yield to_xy(fens[b], idx[b], prob[b], wdl[b])
		sig = (tf.TensorSpec((None, 8, 8, 36), tf.float32),
		       {"policy": tf.TensorSpec((None, nash_mcts.POLICY_SIZE), tf.float32), "value": tf.TensorSpec((None, 3), tf.float32)})
		return tf.data.Dataset.from_generator(gen, output_signature=sig).prefetch(16)

	budget = args.hours * 3600
	steps = args.epoch_positions // args.batch
	start = time.time() - state["elapsed"]

	def lr_at(elapsed):
		t = min(1.0, elapsed / budget)
		return args.lr * (args.lr_min + (1 - args.lr_min) * 0.5 * (1 + math.cos(math.pi * t)))

	class Clock(keras.callbacks.Callback):
		"""Cosine learning rate over wall-clock time; stops when the budget is spent. Also writes the progress JSON
		(same shape as distill_lichess_evals.py train --progress, for the play UI's /train page)."""
		def __init__(self):
			super().__init__()
			self.history, self.points, self.epoch = [], [], 0
			if os.path.exists(state_path) and args.resume:
				self.history = state.get("history", [])

		def _write(self, step, logs):
			elapsed = time.time() - start
			done = len(self.history) + step / steps
			epochs = max(done, done * budget / max(elapsed, 1)) if done > 0.05 else 0
			out = {"epoch": self.epoch, "epochs": round(epochs), "step": step, "steps": steps, "elapsed": elapsed,
			       "budget": budget, "lr": lr_at(elapsed), "train": {k: float(v) for k, v in (logs or {}).items()},
			       "history": self.history, "points": self.points, "params": self.model.count_params(),
			       "batch": args.batch, "train_positions": sum(1 for _ in uses) * args.shard_size,
			       "val_positions": len(val[0]), "shards_seen": sum(1 for u in uses.values() if u)}
			if args.progress:
				with open(args.progress + ".tmp", "w") as f:
					json.dump(out, f)
				os.replace(args.progress + ".tmp", args.progress)

		def on_epoch_begin(self, epoch, logs=None):
			self.epoch = epoch

		def on_train_batch_begin(self, batch, logs=None):
			if batch % 50 == 0:
				self.model.optimizer.learning_rate.assign(lr_at(time.time() - start))

		def on_train_batch_end(self, batch, logs=None):
			if (batch + 1) % 25 == 0:
				gs = self.epoch * steps + batch + 1
				self.points.append([gs, float(logs["loss"]), float(logs.get("policy_top1", 0)), float(logs.get("policy_top3", 0))])
				self.points = self.points[-4000:]
				self._write(batch + 1, logs)
			if time.time() - start > budget:
				self.model.stop_training = True

		def on_epoch_end(self, epoch, logs=None):
			self.history.append({**{k: float(v) for k, v in logs.items()}, "elapsed": time.time() - start})
			with open(state_path + ".tmp", "w") as f:
				json.dump({"elapsed": time.time() - start, "uses": uses, "history": self.history}, f)
			os.replace(state_path + ".tmp", state_path)
			self._write(0, {})

	if args.resume and os.path.exists(last):
		model = keras.models.load_model(last)
		initial_epoch = int(model.optimizer.iterations.numpy()) // steps
		print(f"resuming from {last} at epoch {initial_epoch + 1}, {state['elapsed'] / 3600:.2f} h in", flush=True)
	else:
		model = model_framework.exploratory_model(FILTERS=args.filters, BLOCKS=args.blocks, dropout=0.0, policy_head=True)
		model.compile(
			optimizer=keras.optimizers.AdamW(learning_rate=args.lr, weight_decay=1e-4),
			loss={"policy": keras.losses.CategoricalCrossentropy(from_logits=True), "value": keras.losses.CategoricalCrossentropy()},
			metrics={"policy": [keras.metrics.CategoricalAccuracy(name="top1"), keras.metrics.TopKCategoricalAccuracy(k=3, name="top3")]},
		)
		initial_epoch = 0
	print(f"{model.count_params():,} params; {steps} steps ({args.epoch_positions:,} positions) per epoch; "
	      f"{args.hours} h budget", flush=True)
	callbacks = [keras.callbacks.ModelCheckpoint(last), Clock()]
	if args.keep_epochs:
		os.makedirs(args.keep_epochs, exist_ok=True)
		callbacks.append(keras.callbacks.ModelCheckpoint(os.path.join(args.keep_epochs, "epoch_{epoch:02d}.keras")))
	model.fit(batches(True), steps_per_epoch=steps, epochs=10_000, initial_epoch=initial_epoch,
	          validation_data=batches(False), validation_steps=len(val[0]) // args.batch, callbacks=callbacks, verbose=2)
	model.save(args.out)
	print(f"saved {args.out} after {(time.time() - start) / 3600:.2f} h", flush=True)


def main():
	parser = argparse.ArgumentParser()
	sub = parser.add_subparsers(dest="cmd", required=True)
	p = sub.add_parser("prep")
	p.add_argument("--out", default="data/evals_full")
	p.add_argument("--workers", type=int, default=12)
	p.add_argument("--shard", type=int, default=2_000_000)
	p.add_argument("--max-positions", type=int, default=0, help="stop after this many (0: the whole file)")
	t = sub.add_parser("train")
	t.add_argument("--data", default="data/evals_full")
	t.add_argument("--out", default="models/distill_8x96.keras")
	t.add_argument("--blocks", type=int, default=8)
	t.add_argument("--filters", type=int, default=96)
	t.add_argument("--hours", type=float, default=5.75)
	t.add_argument("--batch", type=int, default=1024)
	t.add_argument("--lr", type=float, default=2e-3)
	t.add_argument("--lr-min", type=float, default=0.02, help="final learning rate as a fraction of --lr")
	t.add_argument("--epoch-positions", type=int, default=20_000_000, help="positions between checkpoints / val")
	t.add_argument("--shard-size", type=int, default=2_000_000, help="prep's --shard (for progress reporting)")
	t.add_argument("--val", type=int, default=20_480)  # a whole number of batches
	t.add_argument("--val-shard", type=int, default=3)
	t.add_argument("--keep-epochs", default=None, metavar="DIR", help="also save every epoch's model to DIR/epoch_XX.keras")
	t.add_argument("--resume", action="store_true", help="continue from the checkpoint and state next to --out")
	t.add_argument("--progress", default=None, help="JSON file updated during training (for the play UI's /train page)")
	args = parser.parse_args()
	prep(args) if args.cmd == "prep" else train(args)


if __name__ == "__main__":
	main()
