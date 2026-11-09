import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from core.chess_precomputed import generate_pseudo_legal_moves
from core.chess_environment import fast_fen_to_example, tensor_to_fen

def test_initial_position():
	"""
	Test that the initial chess position is correctly encoded.
	"""
	start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
	tensor = fast_fen_to_example(start_fen)

	# Check piece placement
	assert np.sum(tensor[:, :, :7]) == 16  # 16 white pieces
	assert np.sum(tensor[:, :, 9:16]) == 16  # 16 black pieces
	
	# Check castling rights
	assert tensor[:, :, 7].sum() == 64  # White kingside castling
	assert tensor[:, :, 8].sum() == 64  # White queenside castling
	assert tensor[:, :, 16].sum() == 64  # Black kingside castling
	assert tensor[:, :, 17].sum() == 64  # Black queenside castling
	
	# Ensure en passant is empty
	assert tensor[:, :, 18].sum() == 0  # No en passant squares

def test_empty_board():
	"""
	Test that an empty board results in a tensor of all zeros (except legal move planes).
	"""
	empty_fen = "8/8/8/8/8/8/8/8 w - - 0 1"
	tensor = fast_fen_to_example(empty_fen)

	# Ensure no pieces exist on the board
	assert np.sum(tensor[:, :, :19]) == 0  # No piece placements

	# Ensure no move encodings
	assert np.sum(tensor) == 0

def test_pawn_moves():
	"""
	Test that pawns generate correct pseudo-legal moves.
	"""
	fen = "8/8/8/3P4/8/8/8/8 w - - 0 1"  # Single white pawn in the center
	tensor = fast_fen_to_example(fen)

	# Ensure pawn is placed correctly
	assert tensor[3, 3, 0] == 1  # Pawn at (3,3)
	
	# Ensure pseudo-legal moves are correct
	assert tensor[2, 3, 19] == 1  # Forward move
	assert tensor[1, 3, 19] == 0  # Double move

	fen = "8/8/8/8/8/8/3P4/8 w - - 0 1"  # Single white pawn in the center
	tensor = fast_fen_to_example(fen)

	# Ensure pawn is placed correctly
	assert tensor[6, 3, 0] == 1  # Pawn at (3,3)
	
	# Ensure pseudo-legal moves are correct
	assert tensor[5, 3, 19] == 1  # Forward move
	assert tensor[4, 3, 19] == 1  # Double move

def test_knight_moves():
	"""
	Test that knights generate correct pseudo-legal moves.
	"""
	fen = "8/8/8/4N3/8/8/8/8 w - - 0 1"  # Single white knight in center
	tensor = fast_fen_to_example(fen)

	# Ensure knight is placed correctly
	assert tensor[3, 4, 1] == 1  # Knight at (3,4)

	# Ensure pseudo-legal moves are correct (8 possible)
	expected_knight_moves = [(1, 3), (1, 5), (2, 2), (2, 6), (4, 2), (4, 6), (5, 3), (5, 5)]
	for (r, c) in expected_knight_moves:
		assert tensor[r, c, 20] == 1  # White knight move plane

	fen = "7n/8/8/8/8/8/8/8 w - - 0 1" # villain knight on the rim
	tensor = fast_fen_to_example(fen)

	# Ensure knight is placed correctly
	assert tensor[0, 7, 10] == 1  # Knight at (3,4)

	expected_knight_moves = [(1, 5), (2, 6)]
	for (r, c) in expected_knight_moves:
		assert tensor[r, c, 27] == 1  # Villain Knight


def test_castling_rights():
	"""
	Test that castling rights are correctly encoded.
	"""
	fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"  # Both kings with full castling rights
	tensor = fast_fen_to_example(fen)

	# Ensure castling rights are properly encoded
	assert tensor[:, :, 7].sum() == 64  # White kingside castling
	assert tensor[:, :, 8].sum() == 64  # White queenside castling
	assert tensor[:, :, 16].sum() == 64  # Black kingside castling
	assert tensor[:, :, 17].sum() == 64  # Black queenside castling

def test_en_passant():
	"""
	Test that en passant squares are correctly encoded.
	"""
	fen = "8/8/8/3pP3/8/8/8/8 w - d6 0 1"  # White pawn just moved
	tensor = fast_fen_to_example(fen)

	# Ensure en passant square is marked
	print(tensor[:, :, 18])
	assert tensor[2, 3, 18] == 1  # d6 en passant square

def test_legal_moves_correctness():
	"""
	Test that pseudo-legal moves match expected results.
	"""
	fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"  # White e4 move
	tensor = fast_fen_to_example(fen)

	# Regenerating the move planes from the piece planes is idempotent
	assert np.array_equal(generate_pseudo_legal_moves(tensor.copy()), tensor)

	# Expect e5 to be a legal move for the pawn
	assert tensor[3, 4, 19] == 1  # e5 move

def test_move_planes_are_hero_relative():
	"""
	A position and its colour-mirrored twin (black to move) must encode identically,
	move planes included.
	"""
	pairs = [
		("4k3/8/8/8/8/8/4P3/4K1N1 w - - 0 1", "4k1n1/4p3/8/8/8/8/8/4K3 b - - 0 1"),
		("r3k2r/pppq1ppp/2n2n2/3pp3/1b2P3/2NP1N2/PPPBQPPP/R3K2R w KQkq - 0 1",
		 "r3k2r/pppbqppp/2np1n2/1B2p3/3PP3/2N2N2/PPPQ1PPP/R3K2R b KQkq - 0 1"),
	]
	for white_fen, black_fen in pairs:
		assert np.array_equal(fast_fen_to_example(white_fen), fast_fen_to_example(black_fen))

def test_sliding_moves_stop_at_blockers():
	fen = "8/8/8/8/1p1R2P1/8/8/8 w - - 0 1"  # Rook on d4, villain pawn b4, hero pawn g4
	tensor = fast_fen_to_example(fen)

	rook_plane = tensor[4, :, 23]
	assert list(rook_plane) == [0, 1, 1, 0, 1, 1, 1, 0]  # Ray includes the first blocker and stops there
	assert tensor[:, 3, 23].sum() == 7  # Open file

	# Every slider plane is populated in the start position
	start = fast_fen_to_example("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
	for plane in (21, 22, 23, 24, 28, 29, 30, 31):
		assert start[:, :, plane].sum() > 0

def test_blocked_pawn_push():
	tensor = fast_fen_to_example("8/8/8/8/8/4n3/4P3/8 w - - 0 1")  # e2 pawn blocked by e3 knight
	assert tensor[:, :, 19].sum() == 0  # No pushes, single or double
	assert tensor[5, 3, 33] == 1 and tensor[5, 5, 33] == 1  # Still attacks d3 and f3

def test_fen_loop():
	fens = [
		"rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1",
		"rnr3k1/pq1b1pbp/3p2p1/1p1Pp3/4PPn1/P1NNB1P1/1P4BP/2RQ1RK1 w - - 1 17",
		"rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b Kq e3 0 1"
	]

	for fen in fens:
		tensor = fast_fen_to_example(fen)
		reconstruction_fen = tensor_to_fen(tensor[:, :, :19], is_white_turn=fen.split()[1] == "w")

		parts = fen.split()
		board_part, turn, castling_rights, en_passant = parts[0], parts[1], parts[2], parts[3]
		recon_parts = reconstruction_fen.split()

		assert recon_parts[0] == parts[0]
		assert recon_parts[1] == parts[1]
		assert recon_parts[2] == parts[2]
		assert recon_parts[3] == parts[3]


if __name__ == "__main__":
	pytest.main()