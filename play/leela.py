"""
A Leela Chess Zero network as a drop-in for play.engine.Network, so it searches with our C++ MCTS.

The .pb.gz weights are converted once with `lc0 leela2onnx --input=net.pb.gz --output=net.onnx` and run with
onnxruntime. Our tree hands over the UCI path to every pending leaf; for each leaf this module rebuilds the
position plus its last 8 positions of history, encodes them in lc0's classical 112-plane format, and maps lc0's
1858-move policy onto our 4672-index layout (core/move_encoding.py) for the legal moves.

Encoding (lc0 src/neural/encoder.cc, INPUT_CLASSICAL_112_PLANE), all from the side to move's view, with the
board mirrored vertically when black is to move:
	- 8 history steps x 13 planes: our P N B R Q K, their P N B R Q K, "position occurred before"
	  (missing history before the game's starting FEN repeats the oldest position, lc0's FEN_ONLY fill)
	- 104-107: we can castle queenside / kingside, they can castle queenside / kingside
	- 108: black to move; 109: rule-50 ply counter; 110: zeros; 111: ones
Policy: lc0 indexes moves in the mirrored frame, castling as king-takes-rook (e1h1), queen/rook/bishop
promotions with a suffix and knight promotions without one.

Leela's networks and lc0 are GPL-3.0 (https://github.com/LeelaChessZero/lc0); the 1858-move table is
lczero-training's tf/policy_index.py.
"""
import os

import chess
import numpy as np

from core import move_encoding
from play.leela_policy_index import POLICY_INDEX

LC0_POLICY_TEMPERATURE = 1.359  # lc0's default PolicyTemperature; Leela nets are meant to be read through it

_LC0_INDEX = {uci: i for i, uci in enumerate(POLICY_INDEX)}
_PIECES = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)
_SUFFIX = {None: "", chess.QUEEN: "q", chess.ROOK: "r", chess.BISHOP: "b", chess.KNIGHT: ""}


def lc0_move_index(board, move):
	"""lc0 policy index of a legal move on `board`."""
	frm, to = move.from_square, move.to_square
	if board.is_castling(move):  # king takes own rook
		to = chess.square(7 if chess.square_file(to) > chess.square_file(frm) else 0, chess.square_rank(frm))
	if board.turn == chess.BLACK:
		frm, to = chess.square_mirror(frm), chess.square_mirror(to)
	return _LC0_INDEX[chess.square_name(frm) + chess.square_name(to) + _SUFFIX[move.promotion]]


def _bits(masks):
	"""uint64 bitboards -> (k, 8, 8) float planes, row = rank, col = file (lc0's plane layout)."""
	b = np.array(masks, dtype="<u8").view(np.uint8)
	return np.unpackbits(b, bitorder="little").reshape(-1, 8, 8).astype(np.float32)


def encode(board):
	"""lc0 classical 112-plane input (112, 8, 8) for `board`, using its move stack as history."""
	us, them = board.turn, not board.turn
	flip = us == chess.BLACK
	b = board.copy()
	masks = []
	for step in range(8):
		ms = []
		for colour in (us, them):
			for piece in _PIECES:
				m = b.pieces_mask(piece, colour)
				ms.append(chess.flip_vertical(m) if flip else m)
		masks += ms + [0xFFFFFFFFFFFFFFFF if b.is_repetition(2) else 0]
		if not b.move_stack:
			# lc0's FEN_ONLY fill: repeat the oldest known position (unless it is the standard start)
			if b.board_fen() != chess.STARTING_BOARD_FEN:
				masks += (ms + [0]) * (7 - step)
			break
		b.pop()
	planes = np.zeros((112, 8, 8), np.float32)
	planes[:len(masks)] = _bits(masks)
	planes[104] = board.has_queenside_castling_rights(us)
	planes[105] = board.has_kingside_castling_rights(us)
	planes[106] = board.has_queenside_castling_rights(them)
	planes[107] = board.has_kingside_castling_rights(them)
	planes[108] = float(flip)
	planes[109] = board.halfmove_clock
	planes[111] = 1.0
	return planes


class LeelaNetwork:
	"""Same interface as play.engine.Network for kind == "leela": evaluate(board, paths) -> (logits, wdl)."""

	kind = "leela"

	def __init__(self, path, policy_temperature=LC0_POLICY_TEMPERATURE, threads=4, device=None):
		import onnxruntime as ort
		providers = ["CPUExecutionProvider"]
		if (device or os.environ.get("LEELA_DEVICE", "cuda")) == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
			providers.insert(0, "CUDAExecutionProvider")
		opts = ort.SessionOptions()
		opts.intra_op_num_threads = threads
		self.session = ort.InferenceSession(path, opts, providers=providers)
		self.providers = self.session.get_providers()
		self.input_name = self.session.get_inputs()[0].name
		self.outputs = [o.name for o in self.session.get_outputs()]
		self.policy_out = next(n for n in self.outputs if "policy" in n)
		self.wdl_out = next((n for n in self.outputs if "wdl" in n), None)
		self.value_out = None if self.wdl_out else next(n for n in self.outputs if "value" in n)
		self.policy_temperature = policy_temperature
		self.name = os.path.basename(path)
		self.positions = 0
		self.run(np.zeros((1, 112, 8, 8), np.float32))  # warm-up

	def run(self, planes):
		"""Raw net outputs for (k, 112, 8, 8) planes: lc0 policy logits (k, 1858) and WDL (k, 3) for the side to move."""
		out = dict(zip(self.outputs, self.session.run(self.outputs, {self.input_name: planes})))
		policy = out[self.policy_out].astype(np.float32)
		if self.wdl_out:
			wdl = out[self.wdl_out].astype(np.float32)
		else:  # value-only net: q in [-1, 1] -> a win/loss split with the same expected score
			q = out[self.value_out].reshape(-1).astype(np.float32)
			wdl = np.stack([(1 + q) / 2, np.zeros_like(q), (1 - q) / 2], axis=1)
		return policy, wdl

	def evaluate(self, root_board, paths):
		"""Our-layout policy logits (k, 4672) and WDL (k, 3) for the leaves at `paths` (UCI lists) below root_board."""
		leaves = []
		for path in paths:
			b = root_board.copy()
			for uci in path:
				b.push_uci(uci)
			leaves.append(b)
		logits = np.full((len(leaves), move_encoding.POLICY_SIZE), -1e9, np.float32)
		if not leaves:
			return logits, np.zeros((0, 3), np.float32)
		self.positions += len(leaves)
		policy, wdl = self.run(np.stack([encode(b) for b in leaves]))
		for i, b in enumerate(leaves):
			white = b.turn == chess.WHITE
			for move in b.legal_moves:
				logits[i, move_encoding.move_to_index(move, white)] = policy[i, lc0_move_index(b, move)] / self.policy_temperature
		return logits, wdl
