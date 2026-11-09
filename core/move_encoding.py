import numpy as np
import chess

"""
Policy encoding shared by the network's policy head and the C++ MCTS (cpp/mcts).

The policy is an (8, 8, 73) grid flattened channels-last to 4672 logits, in the same hero frame
as fast_fen_to_example: row 0 is the far rank from the side to move, col 0 is the a-file,
and the hero moves "up" (toward row 0).

	index = (from_row * 8 + from_col) * 73 + plane

Planes:
	- 0-55: queen-like moves, plane = direction * 7 + (distance - 1), directions in QUEEN_DIRECTIONS.
	        Queen promotions, castling (king moves two squares) and en passant use these planes.
	- 56-63: knight moves, plane = 56 + index in KNIGHT_OFFSETS
	- 64-72: under-promotions, plane = 64 + piece * 3 + (dc + 1), piece in (N, B, R), dc in (-1, 0, +1)
"""

POLICY_PLANES = 73
POLICY_SIZE = 8 * 8 * POLICY_PLANES  # 4672

# (d_row, d_col) in the hero frame, clockwise from "forward"
QUEEN_DIRECTIONS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
KNIGHT_OFFSETS = [(-2, 1), (-1, 2), (1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1)]
UNDERPROMOTIONS = [chess.KNIGHT, chess.BISHOP, chess.ROOK]


def square_to_hero(square, white_to_move):
	"""python-chess square -> (row, col) in the hero frame."""
	rank, file = chess.square_rank(square), chess.square_file(square)
	return (7 - rank if white_to_move else rank), file


def hero_to_square(row, col, white_to_move):
	return chess.square(col, 7 - row if white_to_move else row)


def move_to_index(move, white_to_move):
	"""Index of a (pseudo-)legal move in the flattened 4672 policy vector."""
	from_row, from_col = square_to_hero(move.from_square, white_to_move)
	to_row, to_col = square_to_hero(move.to_square, white_to_move)
	dr, dc = to_row - from_row, to_col - from_col

	if move.promotion is not None and move.promotion != chess.QUEEN:
		plane = 64 + UNDERPROMOTIONS.index(move.promotion) * 3 + (dc + 1)
	elif (dr, dc) in KNIGHT_OFFSETS:
		plane = 56 + KNIGHT_OFFSETS.index((dr, dc))
	else:
		distance = max(abs(dr), abs(dc))
		direction = QUEEN_DIRECTIONS.index((dr // distance, dc // distance))
		plane = direction * 7 + (distance - 1)

	return (from_row * 8 + from_col) * POLICY_PLANES + plane


def index_to_move(index, board):
	"""Inverse of move_to_index for the given board. Returns a chess.Move (not checked for legality)."""
	white = board.turn == chess.WHITE
	square_index, plane = divmod(int(index), POLICY_PLANES)
	from_row, from_col = divmod(square_index, 8)
	promotion = None

	if plane < 56:
		direction, distance = divmod(plane, 7)
		dr, dc = QUEEN_DIRECTIONS[direction]
		to_row, to_col = from_row + dr * (distance + 1), from_col + dc * (distance + 1)
	elif plane < 64:
		dr, dc = KNIGHT_OFFSETS[plane - 56]
		to_row, to_col = from_row + dr, from_col + dc
	else:
		piece, dc = divmod(plane - 64, 3)
		promotion = UNDERPROMOTIONS[piece]
		to_row, to_col = from_row - 1, from_col + dc - 1

	from_square = hero_to_square(from_row, from_col, white)
	to_square = hero_to_square(to_row, to_col, white)
	if promotion is None and to_row == 0 and board.piece_type_at(from_square) == chess.PAWN:
		promotion = chess.QUEEN
	return chess.Move(from_square, to_square, promotion)


def legal_move_indices(board):
	"""Legal moves and their policy indices, in board.legal_moves order."""
	white = board.turn == chess.WHITE
	moves = list(board.legal_moves)
	return moves, np.array([move_to_index(m, white) for m in moves], dtype=np.int64)


def masked_policy(logits, indices):
	"""Softmax of the policy logits restricted to the given legal-move indices."""
	legal = np.asarray(logits, dtype=np.float64)[indices]
	legal = np.exp(legal - legal.max())
	return legal / legal.sum()
