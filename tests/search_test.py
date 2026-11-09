import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import chess
import numpy as np
from core.self_play import select_depth2_move, select_stochastic_move


class ConstantModel:
	"""Stands in for a Keras model: every position gets the same [win, draw, loss]."""
	def __init__(self, value=(0.3, 0.4, 0.3)):
		self.value = np.array(value)
		self.calls = 0

	def predict(self, x, batch_size=None, verbose=0):
		self.calls += 1
		return np.tile(self.value, (len(x), 1))


class LegalityCheckingBoard(chess.Board):
	"""Fails if the search ever pushes a move that isn't pseudo-legal in the current position."""
	def push(self, move):
		assert self.is_pseudo_legal(move), f"{move} pushed on {self.fen()}"
		return super().push(move)


def test_depth2_only_pushes_real_moves():
	board = LegalityCheckingBoard("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
	select_depth2_move(board, ConstantModel())


class LovesG4Pawn(ConstantModel):
	"""Rates every position with a hero pawn on g4 as nearly won, so 2.g4 looks great."""
	def predict(self, x, batch_size=None, verbose=0):
		has_g4_pawn = x[:, 4, 6, 0] == 1
		return np.where(has_g4_pawn[:, None], [0.9, 0.1, 0.0], [0.2, 0.6, 0.2])


def test_depth2_avoids_mate_in_one():
	# After 1.f3 e5, 2.g4?? allows Qh4#. The model loves g4, so only the search can refute it.
	board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/8/5P2/PPPPP1PP/RNBQKBNR w KQkq - 0 2")
	move, value = select_depth2_move(board, LovesG4Pawn())
	assert move != chess.Move.from_uci("g2g4")
	assert value[2] < 1.0


def test_depth2_finds_mate_in_one():
	board = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")  # Ra8#
	move, value = select_depth2_move(board, ConstantModel())
	assert move == chess.Move.from_uci("a1a8")
	assert value[0] == 1.0


def test_depth2_scores_stalemate_as_draw():
	# b5-b6 stalemates the black king on a8. If the model says we lose everywhere else,
	# the drawn leaf must be the best option.
	board = chess.Board("k7/2K5/8/1P6/8/8/8/8 w - - 0 1")
	move, value = select_depth2_move(board, ConstantModel((0.0, 0.0, 1.0)))
	assert move == chess.Move.from_uci("b5b6")
	assert np.array_equal(value, [0.0, 1.0, 0.0])


def test_stochastic_takes_mate_in_one():
	board = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
	move, value = select_stochastic_move(board, ConstantModel(), temp=1.0)
	assert move == chess.Move.from_uci("a1a8")
