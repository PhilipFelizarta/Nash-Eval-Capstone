"""
Head-to-head between two networks, both searching with our C++ MCTS at a fixed number of new simulations per move,
from the same balanced openings as sf_match (each opening once with each colour). Either model can be one of our
Keras checkpoints or a Leela net converted to ONNX (play/leela.py). --nodes 1 is pure policy play.

	python self_play_agent/net_match.py --a out/v3/distill_8x96.keras --b leela/nets/11258-96x8-se.onnx \
		--openings data/evals.npz --games 20 --nodes 800 --out out/leela/v3_vs_leela_n800.json
"""
import argparse
import json
import os
import sys

import chess
import chess.pgn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "cpp", "mcts"))

from play.engine import Engine, load_network  # noqa: E402
from self_play_agent.checkpoint_round_robin import pick_openings  # noqa: E402


def main():
	p = argparse.ArgumentParser()
	p.add_argument("--a", required=True, help="model A (scores are reported for A)")
	p.add_argument("--b", required=True, help="model B")
	p.add_argument("--openings", required=True)
	p.add_argument("--games", type=int, default=20, help="even: each opening is played with both colours")
	p.add_argument("--nodes", type=int, default=800, help="new simulations per move, both sides")
	p.add_argument("--max-plies", type=int, default=300)
	p.add_argument("--out", default=None, help="JSON summary (and a PGN next to it)")
	args = p.parse_args()

	engines = {"a": Engine(load_network(args.a)), "b": Engine(load_network(args.b))}
	names = {k: e.net.name for k, e in engines.items()}
	openings = pick_openings(args.openings, args.games // 2, seed=7)
	points, results, pgns = 0.0, [], []
	for g in range(args.games):
		fen, a_white = openings[g // 2], g % 2 == 0
		board = chess.Board(fen)
		for e in engines.values():
			e.new_game(fen)
		while not board.is_game_over(claim_draw=True) and len(board.move_stack) < args.max_plies:
			side = "a" if (board.turn == chess.WHITE) == a_white else "b"
			move = chess.Move.from_uci(engines[side].think(board, 60, nodes=args.nodes))
			board.push(move)
			for e in engines.values():
				e.advance(move.uci())
		res = board.result(claim_draw=True)
		if res == "*":
			res = "1/2-1/2"
		score = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}[res]
		points += score if a_white else 1 - score
		results.append({"opening": fen, "a": "white" if a_white else "black", "result": res, "plies": len(board.move_stack)})
		game = chess.pgn.Game.from_board(board)
		game.headers.update({"White": names["a" if a_white else "b"], "Black": names["b" if a_white else "a"], "Result": res})
		pgns.append(str(game))
		print(f"game {g + 1}/{args.games}: A {'W' if a_white else 'B'} {res} ({len(board.move_stack)} plies) | A score {points}/{g + 1}", flush=True)
	summary = {"a": args.a, "b": args.b, "nodes": args.nodes, "games": args.games, "points_a": points,
	           "score_a": points / args.games, "results": results}
	print(json.dumps({k: v for k, v in summary.items() if k != "results"}), flush=True)
	if args.out:
		with open(args.out, "w") as f:
			json.dump(summary, f)
		with open(os.path.splitext(args.out)[0] + ".pgn", "w") as f:
			f.write("\n\n".join(pgns))


if __name__ == "__main__":
	main()
