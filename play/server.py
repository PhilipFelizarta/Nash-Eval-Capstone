"""
Play against the network + C++ MCTS in the browser.

	python play/server.py --model path/to/model.h5 --kind legacy   # old value-only checkpoint
	python play/server.py --model path/to/pv_model.keras           # policy/value model
	python play/server.py                                          # untrained placeholder net

Then open http://127.0.0.1:8765. The engine thinks --seconds per move (default 5).
"""
import argparse
import os
import threading

import chess
import chess.pgn
from fastapi import FastAPI, HTTPException
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
	args = parser.parse_args()

	net = Network(args.model, args.kind)
	engine = Engine(net, batch_size=args.batch if args.kind == "policy" else min(args.batch, 16))
	game = Game(engine, args.seconds)

	app = FastAPI()

	@app.get("/")
	def index():
		return FileResponse(os.path.join(HERE, "static", "index.html"))

	@app.get("/api/state")
	def state():
		return game.state()

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
