"""
Play against the network + C++ MCTS in the browser.

	python play/server.py --model path/to/model.h5 --kind legacy   # old value-only checkpoint
	python play/server.py --model path/to/pv_model.keras           # policy/value model
	python play/server.py                                          # untrained placeholder net

Then open http://127.0.0.1:8765. The engine thinks --seconds per move (default 5).
"""
import argparse
import asyncio
import json
import os
import threading
import time

import chess
import chess.pgn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

from engine import Engine, Network

HERE = os.path.dirname(os.path.abspath(__file__))


class Game:
	def __init__(self, engine, seconds):
		self.engine, self.seconds = engine, seconds
		self.lock = threading.Lock()
		self.thread = None
		self.reset("white")

	def reset(self, human_color):
		self.board = chess.Board()
		self.human = chess.WHITE if human_color == "white" else chess.BLACK
		self.status = "human"
		self.engine_log = []  # per engine move: final search snapshot
		self.search = None
		self.engine.new_game()
		if self.board.turn != self.human:
			self._start_engine()

	def result(self):
		if self.board.is_checkmate():
			return "0-1" if self.board.turn == chess.WHITE else "1-0"
		if self.board.is_game_over(claim_draw=True):
			return "1/2-1/2"
		return None

	def human_move(self, uci):
		with self.lock:
			if self.status != "human":
				raise HTTPException(409, "not your turn")
			move = chess.Move.from_uci(uci)
			if move not in self.board.legal_moves:
				raise HTTPException(400, f"illegal move {uci}")
			self.board.push(move)
			self.engine.advance(uci)
			if self.result():
				self.status = "over"
				return
			self._start_engine()

	def _start_engine(self):
		self.status = "thinking"
		self.search = None
		self.thread = threading.Thread(target=self._think, daemon=True)
		self.thread.start()

	def _think(self):
		board = self.board.copy()
		best = self.engine.think(board, self.seconds, on_update=self._on_update)
		with self.lock:
			snap = dict(self.search or {})
			snap["played"] = board.san(chess.Move.from_uci(best))
			snap["ply"] = len(self.board.move_stack)
			self.engine_log.append(snap)
			self.board.push_uci(best)
			self.engine.advance(best)
			self.status = "over" if self.result() else "human"

	def _on_update(self, snap):
		self.search = snap

	def state(self):
		with self.lock:
			board = self.board
			pgn = chess.pgn.Game.from_board(board)
			return {
				"fen": board.fen(),
				"turn": "white" if board.turn == chess.WHITE else "black",
				"human": "white" if self.human == chess.WHITE else "black",
				"status": self.status,
				"result": self.result(),
				"check": board.is_check(),
				"legal": [m.uci() for m in board.legal_moves] if self.status == "human" else [],
				"last_move": board.move_stack[-1].uci() if board.move_stack else None,
				"moves_text": str(pgn.mainline_moves()),
				"search": self.search,
				"last_engine": self.engine_log[-1] if self.engine_log else None,
				"history": [{"ply": s["ply"], "played": s["played"], "value": s["root_value"],
				             "nps": s["nodes_per_second"], "visits": s["new_visits"]} for s in self.engine_log],
				"network": self.engine.net.name,
				"seconds": self.seconds,
			}


class MoveIn(BaseModel):
	uci: str


class NewIn(BaseModel):
	color: str = "white"


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", default=None)
	parser.add_argument("--kind", choices=["policy", "legacy"], default="policy")
	parser.add_argument("--seconds", type=float, default=5.0)
	parser.add_argument("--batch", type=int, default=256)
	parser.add_argument("--port", type=int, default=8765)
	parser.add_argument("--host", default="127.0.0.1")
	parser.add_argument("--tournament-progress", default=None, help="progress.json from self_play_agent/checkpoint_round_robin.py, shown at /tournament")
	parser.add_argument("--train-progress", default=None, help="progress JSON from distill_lichess_evals.py train --progress, shown at /train")
	args = parser.parse_args()

	net = Network(args.model, args.kind)
	engine = Engine(net, batch_size=args.batch if args.kind == "policy" else min(args.batch, 16))
	game = Game(engine, args.seconds)

	app = FastAPI()

	@app.get("/")
	def index():
		return FileResponse(os.path.join(HERE, "static", "index.html"))

	@app.get("/train")
	def train_page():
		return FileResponse(os.path.join(HERE, "static", "train.html"))

	def progress(path, stale_after):
		if not path or not os.path.exists(path):
			return {}
		with open(path) as f:
			data = json.load(f)
		data["stale"] = time.time() - os.path.getmtime(path) > stale_after
		return data

	@app.get("/api/train")
	def train_state():
		return progress(args.train_progress, 120)

	@app.get("/tournament")
	def tournament_page():
		return FileResponse(os.path.join(HERE, "static", "tournament.html"))

	@app.get("/api/tournament")
	def tournament_state():
		return progress(args.tournament_progress, 60)

	# Tournament game viewer: games.json is written next to progress.json by checkpoint_round_robin.py.
	games_cache = {"mtime": None, "games": []}

	def tournament_games():
		path = args.tournament_progress and os.path.join(os.path.dirname(args.tournament_progress), "games.json")
		if not path or not os.path.exists(path):
			return []
		mtime = os.path.getmtime(path)
		if mtime != games_cache["mtime"]:
			with open(path) as f:
				games_cache["games"] = {g["id"]: g for g in json.load(f)}
			games_cache["mtime"] = mtime
		return games_cache["games"]

	@app.get("/games")
	def games_page():
		return FileResponse(os.path.join(HERE, "static", "games.html"))

	@app.get("/api/games")
	def games_list():
		return [{"id": g["id"], "white": g["white"], "black": g["black"], "result": g["result"], "plies": len(g["moves"]),
		         "score": g["scores"][-1] if g["scores"] else None} for g in tournament_games().values()]

	@app.get("/api/games/{gid}")
	def game_detail(gid: int):
		g = tournament_games().get(gid)
		if g is None:
			raise HTTPException(404, "no such game")
		board = chess.Board(g["opening"])
		fens, sans = [board.fen()], []
		for uci in g["moves"]:
			move = chess.Move.from_uci(uci)
			sans.append(board.san(move))
			board.push(move)
			fens.append(board.fen())
		ending = None
		if g["result"]:
			outcome = board.outcome(claim_draw=True)
			ending = outcome.termination.name.replace("_", " ").lower() if outcome else \
				("move limit" if g["result"] == "1/2-1/2" else "resignation")
		return {**g, "fens": fens, "sans": sans, "ending": ending,
		        "first_white": chess.Board(g["opening"]).turn}

	@app.get("/api/state")
	def state():
		return game.state()

	@app.websocket("/api/ws")
	async def ws(socket: WebSocket):
		"""Pushes the state whenever it changes (about 8 per second while the engine thinks).
		A websocket rather than server-sent events: Cloudflare quick tunnels buffer SSE."""
		await socket.accept()
		last = None
		try:
			while True:
				data = json.dumps(game.state())
				if data != last:
					last = data
					await socket.send_text(data)
				await asyncio.sleep(0.12)
		except (WebSocketDisconnect, RuntimeError):
			pass

	@app.post("/api/move")
	def move(body: MoveIn):
		game.human_move(body.uci)
		return game.state()

	@app.post("/api/new")
	def new(body: NewIn):
		with game.lock:
			if game.status == "thinking":
				raise HTTPException(409, "engine is thinking")
		game.reset(body.color)
		return game.state()

	print(f"network: {net.name}; open http://{args.host}:{args.port}", flush=True)
	uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
	main()
