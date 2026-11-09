"""
Benchmarks for nash_mcts (no neural network):
  1. MCTS simulations/second with a stub evaluator (uniform priors, constant draw), split into
     time spent in C++ (gather + backup) and in the stub.
  2. Encoding cost per position: nash_mcts.encode vs core.chess_environment.fast_fen_to_example.

	python cpp/mcts/bench.py [--sims 4000]
"""
import argparse
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import chess
import numpy as np

import nash_mcts
from core.chess_environment import fast_fen_to_example

POSITIONS = {
	"start": chess.STARTING_FEN,
	"middlegame": "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
	"endgame": "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
}


def stub(x):
	return np.zeros((len(x), nash_mcts.POLICY_SIZE), np.float32), np.tile(np.float32([0.0, 1.0, 0.0]), (len(x), 1))


def bench_search(fen, sims, batch):
	tree = nash_mcts.MCTS(seed=0)
	tree.reset(fen, [], False)
	t_tree = t_eval = 0.0
	start = time.perf_counter()
	while tree.root_visits() < sims:
		t0 = time.perf_counter()
		x = tree.gather(min(batch, sims - tree.root_visits()))
		t1 = time.perf_counter()
		policy, wdl = stub(x)
		t2 = time.perf_counter()
		tree.backup(policy, wdl)
		t3 = time.perf_counter()
		t_tree += (t1 - t0) + (t3 - t2)
		t_eval += t2 - t1
	total = time.perf_counter() - start
	return tree.root_visits() / total, tree.root_visits() / t_tree, tree.num_nodes()


def random_fens(n, seed=0):
	rng = random.Random(seed)
	fens = []
	while len(fens) < n:
		board = chess.Board()
		for _ in range(rng.randint(1, 120)):
			moves = list(board.legal_moves)
			if not moves:
				break
			board.push(rng.choice(moves))
		fens.append(board.fen())
	return fens


def per_call(fn, items, repeat=3):
	best = float("inf")
	for _ in range(repeat):
		t0 = time.perf_counter()
		for item in items:
			fn(item)
		best = min(best, (time.perf_counter() - t0) / len(items))
	return best


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--sims", type=int, default=4000)
	args = parser.parse_args()

	print(f"MCTS, {args.sims} simulations per search, stub evaluator (no NN)")
	print(f"{'position':<12}{'batch':>6}{'sims/s total':>15}{'sims/s C++ only':>18}{'nodes':>10}")
	for name, fen in POSITIONS.items():
		for batch in (1, 16, 64, 256):
			total, tree_only, nodes = bench_search(fen, args.sims, batch)
			print(f"{name:<12}{batch:>6}{total:>15,.0f}{tree_only:>18,.0f}{nodes:>10,}")

	fens = random_fens(500)
	t_cpp = per_call(nash_mcts.encode, fens)
	t_py = per_call(fast_fen_to_example, fens, repeat=1)
	t0 = time.perf_counter()
	nash_mcts.encode_batch(fens)
	t_batch = (time.perf_counter() - t0) / len(fens)
	print(f"\nEncoding, {len(fens)} random positions (per position)")
	print(f"  fast_fen_to_example (Python): {t_py * 1e6:9.1f} us")
	print(f"  nash_mcts.encode:             {t_cpp * 1e6:9.1f} us  ({t_py / t_cpp:,.0f}x)")
	print(f"  nash_mcts.encode_batch:       {t_batch * 1e6:9.1f} us  ({t_py / t_batch:,.0f}x)")


if __name__ == "__main__":
	main()
