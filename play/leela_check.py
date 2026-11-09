"""
Checks play/leela.py against lc0 itself: for positions from real games (with their move history), the adapter's
move priors and value must match lc0's own (VerboseMoveStats at one node, lc0's CPU backend on the original .pb.gz).

	python play/leela_check.py --lc0 leela/lc0/build/release/lc0 --weights leela/nets/net.pb.gz \
		--onnx leela/nets/net.onnx --pgn games/sf_match_v3/v3_elo_2400.pgn --positions 200
"""
import argparse
import os
import random
import re
import subprocess
import sys

import chess
import chess.pgn
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from play.leela import LeelaNetwork, encode, lc0_move_index  # noqa: E402

MOVE_RE = re.compile(r"info string (\S+)\s+\(\s*\d+\s*\).*?\(P:\s*([\d.]+)%\)")
NODE_RE = re.compile(r"info string node .*?\(V:\s*(-?[\d.]+)\)")


class Lc0:
	def __init__(self, path, weights):
		self.p = subprocess.Popen([path, f"--weights={weights}", "--threads=1", "--verbose-move-stats", "--smart-pruning-factor=0"],
		                          stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
		self.send("uci")
		self.wait("uciok")
		self.send("isready")
		self.wait("readyok")

	def send(self, s):
		self.p.stdin.write(s + "\n")

	def wait(self, token):
		lines = []
		while True:
			line = self.p.stdout.readline()
			if not line:
				raise RuntimeError("lc0 exited")
			lines.append(line.strip())
			if line.startswith(token):
				return lines

	def priors(self, start_fen, moves):
		self.send("ucinewgame")
		self.send(f"position fen {start_fen}" + (" moves " + " ".join(moves) if moves else ""))
		self.send("go nodes 1")
		lines = self.wait("bestmove")
		priors, value = {}, None
		for line in lines:
			m = MOVE_RE.match(line)
			if m and m.group(1) != "node":
				priors[m.group(1)] = float(m.group(2)) / 100
			n = NODE_RE.match(line)
			if n:
				value = float(n.group(1))
		return priors, value


def lc0_uci(board, move):
	"""lc0's verbose stats print castling as king-takes-rook."""
	if board.is_castling(move):
		rook_file = 7 if chess.square_file(move.to_square) > chess.square_file(move.from_square) else 0
		return chess.square_name(move.from_square) + chess.square_name(chess.square(rook_file, chess.square_rank(move.from_square)))
	return move.uci()


# Promotions (all four pieces, both colours, with captures) and en passant, which real-game samples rarely hit.
EXTRA = [
	("8/1P4k1/8/8/8/8/5Kp1/7R w - - 0 1", []), ("8/1P4k1/8/8/8/8/5Kp1/7R w - - 0 1", ["f2e3"]),
	("r3k3/1P6/8/8/8/8/6p1/4K2R w - - 0 1", []), ("r3k3/1P6/8/8/8/8/6p1/4K2R w - - 0 1", ["e1e2"]),
	("rnbqkbnr/ppp1pppp/8/8/3p4/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", ["e2e4"]),
	("rnbqkbnr/pppp1ppp/8/8/4p3/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1", ["d7d5", "d2d4"]),
]


def positions_from_pgns(paths, k, seed):
	out = []
	for path in paths:
		with open(path) as f:
			while (game := chess.pgn.read_game(f)) is not None:
				start = game.headers.get("FEN", chess.STARTING_FEN)
				moves = [m.uci() for m in game.mainline_moves()]
				out += [(start, moves[:i]) for i in range(len(moves))]
	random.Random(seed).shuffle(out)
	return out[:k]


def main():
	p = argparse.ArgumentParser()
	p.add_argument("--lc0", required=True)
	p.add_argument("--weights", required=True)
	p.add_argument("--onnx", required=True)
	p.add_argument("--pgn", nargs="+", required=True)
	p.add_argument("--positions", type=int, default=200)
	args = p.parse_args()

	lc0 = Lc0(args.lc0, args.weights)
	net = LeelaNetwork(args.onnx, device="cpu")
	max_err, top_agree, values, n = 0.0, 0, [], 0
	for start, moves in EXTRA + positions_from_pgns(args.pgn, args.positions, seed=1):
		board = chess.Board(start)
		for uci in moves:
			board.push_uci(uci)
		ref, ref_v = lc0.priors(start, moves)
		policy, wdl = net.run(encode(board)[None])
		legal = list(board.legal_moves)
		logits = np.array([policy[0, lc0_move_index(board, m)] for m in legal]) / net.policy_temperature
		ours = np.exp(logits - logits.max())
		ours /= ours.sum()
		mine = {lc0_uci(board, m): float(q) for m, q in zip(legal, ours)}
		if set(ref) != set(mine):
			print("move sets differ:", board.fen(), sorted(set(ref) ^ set(mine)))
			continue
		err = max(abs(mine[u] - ref[u]) for u in ref)
		max_err = max(max_err, err)
		top_agree += max(ref, key=ref.get) == max(mine, key=mine.get)
		v = float(wdl[0, 0] - wdl[0, 2])
		if ref_v is not None:
			values.append(abs(v - ref_v))
		n += 1
		if err > 0.02:
			print(f"prior mismatch {err:.3f}: {board.fen()}")
	print(f"positions {n}: max prior error {max_err:.4f}, top-move agreement {top_agree}/{n}, "
	      f"value |err| mean {np.mean(values) if values else float('nan'):.4f} max {max(values) if values else float('nan'):.4f}")
	lc0.send("quit")


if __name__ == "__main__":
	main()
