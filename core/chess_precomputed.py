'''
This python file does a precomputation on import.
Legal move generation is slow, so we precompute psuedo legal move variables 
for fast embeddings. This should help inform the neural networks how pieces move.
'''

import numpy as np

# Move lookup tables
KNIGHT_MOVES = {}
KING_MOVES = {}
PAWN_MOVES = {True: {}, False: {}}  # Separate for white & black pawns
SLIDING_MOVES = {"R": {}, "B": {}, "Q": {}}  # Rook, Bishop, Queen


def precompute_psuedo_moves():
	"""
	Precomputes legal move masks for each piece type on an empty board.
	"""
	global KNIGHT_MOVES, KING_MOVES, PAWN_MOVES, SLIDING_MOVES

	# Precompute knight moves
	knight_offsets = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]
	for square in range(64):
		r, c = divmod(square, 8)
		KNIGHT_MOVES[square] = [(r + dr, c + dc) for dr, dc in knight_offsets if 0 <= r + dr < 8 and 0 <= c + dc < 8]

	# Precompute king moves
	king_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
	for square in range(64):
		r, c = divmod(square, 8)
		KING_MOVES[square] = [(r + dr, c + dc) for dr, dc in king_offsets if 0 <= r + dr < 8 and 0 <= c + dc < 8]

	# Precompute pawn moves (for both colors)
	for square in range(64):
		r, c = divmod(square, 8)
		if r > 0:
			PAWN_MOVES[True][square] = [(r - 1, c)]  # White pawn 1-step
			if r == 6:
				PAWN_MOVES[True][square].append((r - 2, c))  # White double move
		if r < 7:
			PAWN_MOVES[False][square] = [(r + 1, c)]  # Black pawn 1-step
			if r == 1:
				PAWN_MOVES[False][square].append((r + 2, c))  # Black double move

	# Precompute sliding piece moves (Rook, Bishop, Queen)
	directions = {
		"R": [(-1, 0), (1, 0), (0, -1), (0, 1)],  # Rook: Up, Down, Left, Right
		"B": [(-1, -1), (-1, 1), (1, -1), (1, 1)],  # Bishop: Diagonals
		"Q": [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)],  # Queen: Rook + Bishop
	}
	
	for piece, dirs in directions.items():
		for square in range(64):
			r, c = divmod(square, 8)
			SLIDING_MOVES[piece][square] = []
			for dr, dc in dirs:
				ray = []
				nr, nc = r + dr, c + dc
				while 0 <= nr < 8 and 0 <= nc < 8:
					ray.append((nr, nc))
					nr += dr
					nc += dc
				SLIDING_MOVES[piece][square].append(ray)

# Call once at initialization
precompute_psuedo_moves()

def generate_pseudo_legal_moves(tensor, piece_positions):
	""" Populates tensor with precomputed pseudo-legal moves for both White and Black using NumPy. """

	# Move plane mapping (same for both sides)
	move_channels = {
		"P": (19, 26),  # Pawn: (White Plane, Black Plane)
		"N": (20, 27),  # Knight
		"B": (21, 28),  # Light-Square Bishop
		"D": (22, 29),  # Dark-Square Bishop
		"R": (23, 30),  # Rook
		"Q": (24, 31),  # Queen
		"K": (25, 32),  # King
	}

	def batch_update(tensor, positions, move_table, plane):
		""" Update multiple move positions at once using NumPy indexing. """
		if not positions:
			return
		indices = np.array([r * 8 + c for r, c in positions])
		valid_indices = indices[indices < 64]  # Ensure indices stay within bounds

		moves = np.array([move_table[idx] for idx in valid_indices if idx in move_table])
		if moves.size > 0:
			moves = np.concatenate(moves, axis=0)  # Flatten list of lists
			valid_moves = moves[(moves[:, 0] >= 0) & (moves[:, 0] < 8) & (moves[:, 1] >= 0) & (moves[:, 1] < 8)]
			tensor[valid_moves[:, 0], valid_moves[:, 1], plane] = 1

	# Process Knights
	batch_update(tensor, piece_positions.get("N_white", []), KNIGHT_MOVES, move_channels["N"][0])
	batch_update(tensor, piece_positions.get("N_black", []), KNIGHT_MOVES, move_channels["N"][1])

	# Process Kings
	batch_update(tensor, piece_positions.get("K_white", []), KING_MOVES, move_channels["K"][0])
	batch_update(tensor, piece_positions.get("K_black", []), KING_MOVES, move_channels["K"][1])

	# Process Pawns
	batch_update(tensor, piece_positions.get("P_white", []), PAWN_MOVES[True], move_channels["P"][0])
	batch_update(tensor, piece_positions.get("P_black", []), PAWN_MOVES[False], move_channels["P"][1])

	# Process Sliding Pieces (Rook, Queen, Light-Square & Dark-Square Bishops)
	for piece, (white_plane, black_plane) in [("R", (23, 30)), ("Q", (24, 31)), ("B", (21, 28)), ("D", (22, 29))]:
		batch_update(tensor, piece_positions.get(f"{piece}_white", []), SLIDING_MOVES["B" if piece in ["B", "D"] else piece], white_plane)
		batch_update(tensor, piece_positions.get(f"{piece}_black", []), SLIDING_MOVES["B" if piece in ["B", "D"] else piece], black_plane)

	return tensor