import os
import sys
import random

import chess

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import move_encoding


TRICKY_FENS = [
	chess.STARTING_FEN,
	"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",  # Castling both sides
	"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1",
	"n1n5/PPPk4/8/8/8/8/4Kppp/5N1N w - - 0 1",  # Promotions and under-promotions with captures
	"n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1",
	"8/8/8/2k5/3Pp3/8/8/4K3 b - d3 0 1",  # En passant
]


def random_positions(n, seed=0):
	rng = random.Random(seed)
	positions = []
	while len(positions) < n:
		board = chess.Board()
		for _ in range(rng.randint(0, 120)):
			moves = list(board.legal_moves)
			if not moves:
				break
			board.push(rng.choice(moves))
		positions.append(board)
	return positions


def all_boards():
	return [chess.Board(fen) for fen in TRICKY_FENS] + random_positions(300)


def test_indices_are_unique_and_in_range():
	for board in all_boards():
		_, indices = move_encoding.legal_move_indices(board)
		assert len(set(indices.tolist())) == len(indices), board.fen()
		assert ((indices >= 0) & (indices < move_encoding.POLICY_SIZE)).all()


def test_index_round_trips_to_the_same_move():
	for board in all_boards():
		moves, indices = move_encoding.legal_move_indices(board)
		for move, index in zip(moves, indices):
			assert move_encoding.index_to_move(index, board) == move, (board.fen(), move.uci())


def test_mirrored_positions_share_indices():
	# The encoding is hero-relative, so a position and its colour-mirror use the same indices
	for board in all_boards()[:100]:
		_, indices = move_encoding.legal_move_indices(board)
		_, mirrored = move_encoding.legal_move_indices(board.mirror())
		assert sorted(indices.tolist()) == sorted(mirrored.tolist()), board.fen()


def test_hero_frame_matches_the_board_tensor():
	# e2e4 for white and e7e5 for black both start on row 6 (hero's second rank), col 4, going "up"
	assert move_encoding.move_to_index(chess.Move.from_uci("e2e4"), True) == (6 * 8 + 4) * 73 + 1
	assert move_encoding.move_to_index(chess.Move.from_uci("e7e5"), False) == (6 * 8 + 4) * 73 + 1
	# Knight b1c3 is (-2, +1): plane 56
	assert move_encoding.move_to_index(chess.Move.from_uci("b1c3"), True) == (7 * 8 + 1) * 73 + 56
	# Under-promotion a7b8=N captures to the right: plane 64 + 0 * 3 + 2
	assert move_encoding.move_to_index(chess.Move.from_uci("a7b8n"), True) == (1 * 8 + 0) * 73 + 66
