"""
Strength check: our network + C++ MCTS at a fixed time per move against Stockfish limited to a given Elo
(UCI_LimitStrength / UCI_Elo, Stockfish's own calibration), from balanced openings with both colours.
--nodes limits our search to that many new simulations per move instead of time; --sf-nodes plays full-strength
Stockfish at a fixed node count per move (e.g. the labeler's 10k) instead of an Elo-limited one.

	python self_play_agent/sf_match.py --model out/v2/distill_6x64.keras --stockfish sf/stockfish-bin \
		--elo 2000 --openings data/evals.npz --games 10 --seconds 1
"""
import argparse
import json
import os
import sys
import time

import chess
import chess.engine
import chess.pgn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "cpp", "mcts"))

from play.engine import Engine, Network  # noqa: E402
from self_play_agent.checkpoint_round_robin import pick_openings  # noqa: E402


def main():
	p = argparse.ArgumentParser()
	p.add_argument("--model", required=True)
	p.add_argument("--stockfish", required=True)
	p.add_argument("--elo", type=int, default=None, help="Stockfish UCI_Elo (1320-3190)")
	p.add_argument("--nodes", type=int, default=None, help="our new simulations per move (instead of --seconds)")
	p.add_argument("--sf-nodes", type=int, default=None, help="full-strength Stockfish at this many nodes per move")
	p.add_argument("--openings", required=True)
	p.add_argument("--games", type=int, default=10, help="even: each opening is played with both colours")
	p.add_argument("--seconds", type=float, default=1.0, help="per move, both sides")
	p.add_argument("--max-plies", type=int, default=300)
	p.add_argument("--out", default=None, help="JSON summary (and a PGN next to it)")
	args = p.parse_args()
	if (args.elo is None) == (args.sf_nodes is None):
		p.error("give exactly one of --elo and --sf-nodes")
	sf_name = f"stockfish-{args.elo}" if args.elo else f"stockfish-{args.sf_nodes}n"
	sf_limit = chess.engine.Limit(nodes=args.sf_nodes) if args.sf_nodes else chess.engine.Limit(time=args.seconds)

	engine = Engine(Network(args.model))
	sf = chess.engine.SimpleEngine.popen_uci(args.stockfish)
	sf.configure({"Threads": 1, "Hash": 64, **({"UCI_LimitStrength": True, "UCI_Elo": args.elo} if args.elo else {})})
	openings = pick_openings(args.openings, args.games // 2, seed=7)
	points, results, nodes, think = 0.0, [], 0, 0.0
	pgns = []
	for g in range(args.games):
		fen, ours_white = openings[g // 2], g % 2 == 0
		board = chess.Board(fen)
		engine.new_game(fen)
		while not board.is_game_over(claim_draw=True) and len(board.move_stack) < args.max_plies:
			if (board.turn == chess.WHITE) == ours_white:
				v0, t0 = engine.tree.root_visits(), time.perf_counter()
				uci = engine.think(board, 60 if args.nodes else args.seconds, nodes=args.nodes)
				nodes += engine.tree.root_visits() - v0
				think += time.perf_counter() - t0
				move = chess.Move.from_uci(uci)
			else:
				move = sf.play(board, sf_limit).move
			board.push(move)
			engine.advance(move.uci())
		res = board.result(claim_draw=True)
		if res == "*":
			res = "1/2-1/2"
		score = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}[res]
		points += score if ours_white else 1 - score
		results.append({"opening": fen, "ours": "white" if ours_white else "black", "result": res, "plies": len(board.move_stack)})
		game = chess.pgn.Game.from_board(board)
		game.headers.update({"White": "nash-eval" if ours_white else sf_name,
		                     "Black": sf_name if ours_white else "nash-eval", "Result": res})
		pgns.append(str(game))
		print(f"game {g + 1}/{args.games}: ours {'W' if ours_white else 'B'} {res} ({len(board.move_stack)} plies) | "
		      f"score {points}/{g + 1} | {nodes / max(think, 1e-9):,.0f} nodes/s", flush=True)
	sf.quit()
	summary = {"model": args.model, "elo": args.elo, "sf_nodes": args.sf_nodes, "nodes": args.nodes, "seconds": args.seconds, "games": args.games, "points": points,
	           "score": points / args.games, "nodes_per_second": nodes / max(think, 1e-9), "results": results}
	print(json.dumps({k: v for k, v in summary.items() if k != "results"}), flush=True)
	if args.out:
		with open(args.out, "w") as f:
			json.dump(summary, f)
		with open(os.path.splitext(args.out)[0] + ".pgn", "w") as f:
			f.write("\n\n".join(pgns))


if __name__ == "__main__":
	main()
