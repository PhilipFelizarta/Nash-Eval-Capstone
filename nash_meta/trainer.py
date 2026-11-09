"""
Trainer: the single training path of the Nash meta-game loop (see nash_meta/README.md).

Starts from the pretrained network (the 10-epoch Lichess distill) and trains continuously on the labeler's
Stockfish subgames: batches are drawn uniformly from a sliding window of the newest --window labeled positions.
Every --checkpoint-positions newly labeled positions it publishes a generation to RUN/pool/gen_XXXX.keras, which
the arena rates against the pool. Training never waits for the Nash; it only pauses when it has sampled more
than --replay times the positions it has ingested, so a slow labeler cannot make it overfit a small window.

	python nash_meta/trainer.py --run RUN --init out/v2/distill_6x64.keras

Subgames whose number is divisible by --val-every are held out (whole subgames, so no near-duplicate leaks).
Progress goes to RUN/trainer.json; RUN/trainer_last.keras (weights and optimizer) makes restarts resume.
"""
import argparse
import glob
import json
import os
import queue
import sys
import threading
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "cpp", "mcts"))

import nash_mcts  # noqa: E402


class Window:
	"""The newest labeled positions, split into train and validation by subgame."""

	def __init__(self, folder, size, val_every, val_size):
		self.folder, self.size, self.val_every, self.val_size = folder, size, val_every, val_size
		self.loaded = []
		self.fens = np.zeros(0, "S100")
		self.idx = np.zeros(0, np.int16)
		self.wdl = np.zeros((0, 3), np.float32)
		self.val = (np.zeros(0, "S100"), np.zeros(0, np.int16), np.zeros((0, 3), np.float32))
		self.lock = threading.Lock()

	def refresh(self, skip=()):
		"""Loads new shards; returns how many train positions they added."""
		new = [f for f in sorted(glob.glob(os.path.join(self.folder, "shard_*.npz")))
		       if f not in self.loaded and os.path.basename(f) not in skip]
		added = 0
		for f in new:
			d = np.load(f)
			held = d["subgame"] % self.val_every == 0
			tr = ~held
			with self.lock:
				self.fens = np.concatenate([self.fens, d["fens"][tr]])[-self.size:]
				self.idx = np.concatenate([self.idx, d["idx"][tr]])[-self.size:]
				self.wdl = np.concatenate([self.wdl, d["wdl"][tr]])[-self.size:]
				vf, vi, vw = self.val
				self.val = (np.concatenate([vf, d["fens"][held]])[-self.val_size:],
				            np.concatenate([vi, d["idx"][held]])[-self.val_size:],
				            np.concatenate([vw, d["wdl"][held]])[-self.val_size:])
			self.loaded.append(f)
			added += int(tr.sum())
		return added

	def sample(self, rng, n):
		with self.lock:
			r = rng.integers(0, len(self.fens), n)
			return self.fens[r], self.idx[r], self.wdl[r]


def to_batch(fens, idx, wdl):
	x = nash_mcts.encode_batch([f.decode() for f in fens])
	policy = np.zeros((len(idx), nash_mcts.POLICY_SIZE), np.float32)
	policy[np.arange(len(idx)), idx] = 1.0
	return x, {"policy": policy, "value": wdl}


def main():
	p = argparse.ArgumentParser()
	p.add_argument("--run", required=True)
	p.add_argument("--init", required=True, help="pretrained model (the start of the single training path)")
	p.add_argument("--batch", type=int, default=1024)
	p.add_argument("--lr", type=float, default=5e-4, help="constant: the data keeps moving")
	p.add_argument("--weight-decay", type=float, default=1e-4)
	p.add_argument("--window", type=int, default=2_000_000, help="train on the newest this many positions")
	p.add_argument("--min-positions", type=int, default=50_000, help="wait for this many before the first step")
	p.add_argument("--replay", type=float, default=8.0, help="max samples drawn per ingested position")
	p.add_argument("--checkpoint-positions", type=int, default=1_000_000, help="new labeled positions per generation")
	p.add_argument("--val-every", type=int, default=50, help="hold out subgames whose number is a multiple of this")
	p.add_argument("--val-size", type=int, default=20_480)
	p.add_argument("--eval-steps", type=int, default=500)
	args = p.parse_args()

	import tensorflow as tf
	from tensorflow import keras
	for gpu in tf.config.list_physical_devices("GPU"):
		tf.config.experimental.set_memory_growth(gpu, True)
	keras.mixed_precision.set_global_policy("mixed_float16")

	pool_dir, tmp_dir = os.path.join(args.run, "pool"), os.path.join(args.run, "tmp")
	for d in (pool_dir, tmp_dir):
		os.makedirs(d, exist_ok=True)
	state_path, last_path = os.path.join(args.run, "trainer.json"), os.path.join(args.run, "trainer_last.keras")

	state = {"gen": 0, "steps": 0, "samples": 0, "ingested": 0, "since_gen": 0, "gens": [], "evals": [], "points": []}
	if os.path.exists(state_path) and os.path.exists(last_path):
		with open(state_path) as f:
			state.update(json.load(f))
		model = keras.models.load_model(last_path)
		print(f"resuming at step {state['steps']:,}, gen {state['gen']}", flush=True)
	else:
		model = keras.models.load_model(args.init, compile=False)
		model.compile(
			optimizer=keras.optimizers.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay),
			loss={"policy": keras.losses.CategoricalCrossentropy(from_logits=True),
			      "value": keras.losses.CategoricalCrossentropy()},
			metrics={"policy": [keras.metrics.CategoricalAccuracy(name="top1"),
			                    keras.metrics.TopKCategoricalAccuracy(k=3, name="top3")]},
		)

	window = Window(os.path.join(args.run, "labeled"), args.window, args.val_every, args.val_size)
	# On resume, shards already counted are reloaded into the window without counting them again.
	window.refresh()
	if state["ingested"] == 0:
		state["ingested"] = len(window.fens)

	batches = queue.Queue(maxsize=8)
	stop = threading.Event()

	def producer():
		rng = np.random.default_rng()
		while not stop.is_set():
			if len(window.fens) < args.min_positions:
				time.sleep(5)
				continue
			batches.put(to_batch(*window.sample(rng, args.batch)))

	threading.Thread(target=producer, daemon=True).start()

	def save_state(extra=None):
		out = {**state, **(extra or {}), "window": len(window.fens), "val": len(window.val[0]), "batch": args.batch,
		       "lr": args.lr, "replay_cap": args.replay, "checkpoint_positions": args.checkpoint_positions,
		       "params": model.count_params(), "updated": time.time()}
		with open(state_path + ".tmp", "w") as f:
			json.dump(out, f)
		os.replace(state_path + ".tmp", state_path)

	def evaluate():
		vf, vi, vw = window.val
		if len(vf) < args.batch:
			return None
		logs = []
		for s in range(0, len(vf) - args.batch + 1, args.batch):
			x, y = to_batch(vf[s:s + args.batch], vi[s:s + args.batch], vw[s:s + args.batch])
			logs.append(model.test_on_batch(x, y, return_dict=True))
		return {k: float(np.mean([l[k] for l in logs])) for k in logs[0]}

	def publish():
		state["gen"] += 1
		name = f"gen_{state['gen']:04d}.keras"
		tmp = os.path.join(tmp_dir, name)
		model.save(tmp)
		os.replace(tmp, os.path.join(pool_dir, name))
		model.save(last_path)
		state["since_gen"] = 0
		state["gens"].append({"gen": state["gen"], "step": state["steps"], "ingested": state["ingested"], "time": time.time(),
		                      "val": evaluate()})
		print(f"published {name} at step {state['steps']:,} ({state['ingested']:,} positions ingested)", flush=True)

	last_refresh = last_write = 0.0
	running = {}
	waiting = False
	try:
		while True:
			now = time.time()
			if now - last_refresh > 15:
				last_refresh = now
				added = window.refresh()
				state["ingested"] += added
				state["since_gen"] = state["since_gen"] + added if state["steps"] else 0  # count from the first step
				if state["since_gen"] >= args.checkpoint_positions and state["steps"] > 0:
					publish()
			waiting = state["samples"] >= args.replay * max(state["ingested"], 1) or len(window.fens) < args.min_positions
			if waiting:
				if now - last_write > 5:
					last_write = now
					save_state({"waiting": True})
				time.sleep(1)
				continue
			x, y = batches.get()
			logs = model.train_on_batch(x, y, return_dict=True)
			state["steps"] += 1
			state["samples"] += args.batch
			for k, v in logs.items():
				running[k] = 0.98 * running.get(k, float(v)) + 0.02 * float(v)
			if state["steps"] % 50 == 0:
				state["points"].append([state["steps"], running["loss"], running.get("policy_top1", 0.0),
				                        running.get("value_loss", 0.0)])
				state["points"] = state["points"][-2000:]
			if state["steps"] % args.eval_steps == 0:
				val = evaluate()
				if val:
					state["evals"].append({"step": state["steps"], **val})
					print(f"step {state['steps']:,}: train loss {running['loss']:.3f} top1 {running.get('policy_top1', 0):.3f}"
					      f" | val loss {val['loss']:.3f} top1 {val.get('policy_top1', 0):.3f} | window {len(window.fens):,}",
					      flush=True)
			if time.time() - last_write > 5:
				last_write = time.time()
				save_state({"waiting": False, "train": running})
	finally:
		stop.set()
		save_state({"waiting": waiting})


if __name__ == "__main__":
	main()
