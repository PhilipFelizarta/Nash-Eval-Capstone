"""
Labeler: the CPU half of the Nash meta-game training loop (see nash_meta/README.md).

Every position of every actor game the arena writes to RUN/games/ becomes the root of a Stockfish subgame:
Stockfish plays both sides from that position to the end of the game (checkmate, stalemate, threefold,
fifty moves, insufficient material; a subgame still going after --max-plies counts as a draw). Every position
of every subgame is a training example: Stockfish's move is the policy target and the subgame's final result,
from the side to move, is the value target. Roots are deduplicated (a position is only played out once).

	taskset -c 0-5 python nash_meta/labeler.py --run RUN --stockfish sf/stockfish-bin --engines 6 --nodes 10000

Labeled positions go to RUN/labeled/shard_XXXXXX.npz (fens, idx = move index, wdl one-hot, subgame = global
subgame number, ply = ply within the subgame). Progress and the backlog the arena throttles on go to
RUN/labeler.json. Restarting is safe: roots already in a shard are skipped, the rest are queued again.
"""
import argparse
import glob
import io
import json
import os
import sys
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import chess
import chess.engine
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from core.move_encoding import move_to_index  # noqa: E402

_engine, _limit, _max_plies = None, None, None


def init_engine(path, nodes, hash_mb, max_plies):
	global _engine, _limit, _max_plies
	_engine = chess.engine.SimpleEngine.popen_uci(path)
	_engine.configure({"Threads": 1, "Hash": hash_mb})
	_limit, _max_plies = chess.engine.Limit(nodes=nodes), max_plies


def play_subgame(root):
	"""Stockfish vs itself from `root` to the end. Returns (fens, move indices, result for White)."""
	board = chess.Board(root)
	fens, moves = [], []
	game = object()  # one game id per subgame: Stockfish clears its hash on ucinewgame
	while not board.is_game_over(claim_draw=True) and len(moves) < _max_plies:
		move = _engine.play(board, _limit, game=game).move
		fens.append(board.fen())
		moves.append(move_to_index(move, board.turn == chess.WHITE))
		board.push(move)
	if board.is_checkmate():
		white = 0.0 if board.turn == chess.WHITE else 1.0
	else:
		white = 0.5
	return root, fens, moves, white


def epd(fen):
	return " ".join(fen.split()[:4])


def game_roots(rec):
	"""Every position of an actor game in which a move was played (the opening included)."""
	board = chess.Board(rec["opening"])
	for uci in rec["moves"]:
		yield board.fen()
		board.push_uci(uci)


class Shards:
	def __init__(self, folder):
		self.folder = folder
		os.makedirs(folder, exist_ok=True)
		files = sorted(glob.glob(os.path.join(folder, "shard_*.npz")))
		self.next = 1 + max([int(os.path.basename(f)[6:12]) for f in files], default=-1)
		self.subgames, self.positions, self.seen = 0, 0, set()
		for f in files:
			data = np.load(f)
			self.seen.update(epd(r.decode()) for r in data["roots"])
			self.subgames = max(self.subgames, int(data["subgame"].max()) + 1) if len(data["subgame"]) else self.subgames
			self.positions += len(data["fens"])
		self.clear()

	def clear(self):
		self.fens, self.idx, self.wdl, self.subgame, self.ply, self.roots = [], [], [], [], [], []

	def add(self, root, fens, moves, white):
		g = self.subgames
		self.subgames += 1
		self.roots.append(root.encode())
		for ply, (fen, m) in enumerate(zip(fens, moves)):
			score = white if fen.split()[1] == "w" else 1 - white
			self.fens.append(fen.encode())
			self.idx.append(m)
			self.wdl.append((float(score == 1.0), float(score == 0.5), float(score == 0.0)))
			self.subgame.append(g)
			self.ply.append(ply)

	def pending(self):
		return len(self.fens)

	def flush(self):
		if not self.roots:
			return None
		path = os.path.join(self.folder, f"shard_{self.next:06d}.npz")
		buf = io.BytesIO()
		np.savez(buf, fens=np.array(self.fens, dtype="S100"), idx=np.array(self.idx, np.int16),
		         wdl=np.array(self.wdl, np.float32).reshape(-1, 3), subgame=np.array(self.subgame, np.int64),
		         ply=np.array(self.ply, np.int16), roots=np.array(self.roots, dtype="S100"))
		with open(path + ".tmp", "wb") as f:
			f.write(buf.getvalue())
		os.replace(path + ".tmp", path)
		self.next += 1
		self.positions += len(self.fens)
		self.clear()
		return path


def main():
	p = argparse.ArgumentParser()
	p.add_argument("--run", required=True)
	p.add_argument("--stockfish", required=True)
	p.add_argument("--engines", type=int, default=6, help="Stockfish processes (one thread each)")
	p.add_argument("--nodes", type=int, default=10_000, help="Stockfish nodes per subgame move")
	p.add_argument("--hash", type=int, default=16, help="MB per engine")
	p.add_argument("--max-plies", type=int, default=400, help="subgame length cap (scored as a draw)")
	p.add_argument("--shard-positions", type=int, default=50_000)
	p.add_argument("--shard-seconds", type=float, default=300)
	args = p.parse_args()

	shards = Shards(os.path.join(args.run, "labeled"))
	print(f"{len(shards.seen):,} roots already labeled ({shards.positions:,} positions)", flush=True)
	queue, parsed = deque(), set()
	results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
	start = last_flush = last_write = time.time()
	done_here = positions_here = 0
	recent = deque(maxlen=200)  # (time, positions) per finished subgame, for the rates

	def scan():
		for f in sorted(glob.glob(os.path.join(args.run, "games", "*.jsonl"))):
			if f in parsed:
				continue
			parsed.add(f)
			with open(f) as fh:
				for line in fh:
					for root in game_roots(json.loads(line)):
						key = epd(root)
						if key not in shards.seen:
							shards.seen.add(key)
							queue.append(root)

	def write_state(in_flight):
		now = time.time()
		span = now - recent[0][0] if len(recent) > 1 else 0
		state = {"backlog": len(queue) + in_flight, "queued": len(queue), "in_flight": in_flight,
		         "subgames": shards.subgames, "positions": shards.positions + shards.pending(),
		         "subgames_here": done_here, "positions_here": positions_here, "results": results,
		         "subgames_per_s": (len(recent) - 1) / span if span else 0.0,
		         "positions_per_s": sum(n for _, n in list(recent)[1:]) / span if span else 0.0,
		         "mean_subgame_plies": positions_here / done_here if done_here else 0.0,
		         "engines": args.engines, "nodes": args.nodes, "shards": shards.next, "elapsed": now - start, "updated": now}
		with open(os.path.join(args.run, "labeler.json.tmp"), "w") as f:
			json.dump(state, f)
		os.replace(os.path.join(args.run, "labeler.json.tmp"), os.path.join(args.run, "labeler.json"))

	with ProcessPoolExecutor(args.engines, initializer=init_engine,
	                         initargs=(args.stockfish, args.nodes, args.hash, args.max_plies)) as pool:
		running = set()
		last_scan = 0.0
		while True:
			if time.time() - last_scan > 10:
				last_scan = time.time()
				scan()
			while queue and len(running) < 2 * args.engines:
				running.add(pool.submit(play_subgame, queue.popleft()))
			if running:
				finished, running = wait(running, timeout=1.0, return_when=FIRST_COMPLETED)
			else:
				finished = set()
				time.sleep(1.0)
			for fut in finished:
				root, fens, moves, white = fut.result()
				shards.add(root, fens, moves, white)
				results[{1.0: "1-0", 0.0: "0-1", 0.5: "1/2-1/2"}[white]] += 1
				done_here += 1
				positions_here += len(fens)
				recent.append((time.time(), len(fens)))
			now = time.time()
			if shards.pending() >= args.shard_positions or (shards.pending() and now - last_flush > args.shard_seconds):
				path = shards.flush()
				last_flush = now
				print(f"{os.path.basename(path)}: {shards.subgames:,} subgames / {shards.positions:,} positions, "
				      f"backlog {len(queue) + len(running):,}", flush=True)
			if now - last_write > 3:
				last_write = now
				write_state(len(running))


if __name__ == "__main__":
	main()
