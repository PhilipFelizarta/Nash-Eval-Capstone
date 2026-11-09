"""
Strength check: our network + C++ MCTS at a fixed time per move against Stockfish limited to a given Elo
(UCI_LimitStrength / UCI_Elo, Stockfish's own calibration), from balanced openings with both colours.

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
	p.add_argument("--elo", type=int, required=True, help="Stockfish UCI_Elo (1320-3190)")
	p.add_argument("--openings", required=True)
	p.add_argument("--games", type=int, default=10, help="even: each opening is played with both colours")
	p.add_argument("--seconds", type=float, default=1.0, help="per move, both sides")
	p.add_argument("--max-plies", type=int, default=300)
	p.add_argument("--out", default=None, help="JSON summary (and a PGN next to it)")
	args = p.parse_args()

	engine = Engine(Network(args.model))
	sf = chess.engine.SimpleEngine.popen_uci(args.stockfish)
	sf.configure({"Threads": 1, "Hash": 64, "UCI_LimitStrength": True, "UCI_Elo": args.elo})
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
				uci = engine.think(board, args.seconds)
				nodes += engine.tree.root_visits() - v0
				think += time.perf_counter() - t0
				move = chess.Move.from_uci(uci)
			else:
				move = sf.play(board, chess.engine.Limit(time=args.seconds)).move
			board.push(move)
			engine.advance(move.uci())
		res = board.result(claim_draw=True)
		if res == "*":
			res = "1/2-1/2"
		score = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}[res]
		points += score if ours_white else 1 - score
		results.append({"opening": fen, "ours": "white" if ours_white else "black", "result": res, "plies": len(board.move_stack)})
		game = chess.pgn.Game.from_board(board)
		game.headers.update({"White": "nash-eval" if ours_white else f"stockfish-{args.elo}",
		                     "Black": f"stockfish-{args.elo}" if ours_white else "nash-eval", "Result": res})
		pgns.append(str(game))
		print(f"game {g + 1}/{args.games}: ours {'W' if ours_white else 'B'} {res} ({len(board.move_stack)} plies) | "
		      f"score {points}/{g + 1} | {nodes / max(think, 1e-9):,.0f} nodes/s", flush=True)
	sf.quit()
	summary = {"model": args.model, "elo": args.elo, "seconds": args.seconds, "games": args.games, "points": points,
	           "score": points / args.games, "nodes_per_second": nodes / max(think, 1e-9), "results": results}
	print(json.dumps({k: v for k, v in summary.items() if k != "results"}), flush=True)
	if args.out:
		with open(args.out, "w") as f:
			json.dump(summary, f)
		with open(os.path.splitext(args.out)[0] + ".pgn", "w") as f:
			f.write("\n\n".join(pgns))


if __name__ == "__main__":
	main()
