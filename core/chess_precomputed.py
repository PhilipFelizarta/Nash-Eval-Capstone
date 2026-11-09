'''
This python file does a precomputation on import.
Legal move generation is slow, so we precompute pseudo-legal move variables 
for fast embeddings. This should help inform the neural networks how pieces move.
'''

import numpy as np

# Move lookup tables
KNIGHT_MOVES = {}
KING_MOVES = {}
PAWN_MOVES = {True: {}, False: {}}  # Separate for white & black pawns
PAWN_ATTACKS = {True: {}, False: {}}  # Pawn attacks for white and black
SLIDING_MOVES = {"R": {}, "B": {}, "Q": {}}  # Rook, Bishop, Queen


def precompute_psuedo_moves():
	"""
	Precomputes legal move masks for each piece type on an empty board.
	"""
	global KNIGHT_MOVES, KING_MOVES, PAWN_MOVES, PAWN_ATTACKS, SLIDING_MOVES

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

	# Precompute pawn attacks
	pawn_attack_offsets = {
		True: [(-1, -1), (-1, 1)],  # White pawn attacks
		False: [(1, -1), (1, 1)]  # Black pawn attacks
	}

	for color in [True, False]:
		for square in range(64):
			r, c = divmod(square, 8)
			PAWN_ATTACKS[color][square] = [(r + dr, c + dc) for dr, dc in pawn_attack_offsets[color]
										   if 0 <= r + dr < 8 and 0 <= c + dc < 8]

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


# Move planes are hero-relative (hero = side to move) and use the tensor's own,
# already-flipped coordinates, so the hero's pawns always move toward row 0.
HERO_MOVE_PLANES = {"P": 19, "N": 20, "B": 21, "D": 22, "R": 23, "Q": 24, "K": 25}
VILLAIN_MOVE_PLANES = {piece: plane + 7 for piece, plane in HERO_MOVE_PLANES.items()}
PAWN_ATTACK_PLANES = (33, 34)  # (Hero, Villain)
PIECE_CHANNELS = {"P": 0, "N": 1, "B": 2, "D": 3, "R": 4, "Q": 5, "K": 6}
SLIDING_TABLES = {"B": "B", "D": "B", "R": "R", "Q": "Q"}  # Both bishop colours slide like a bishop


def generate_pseudo_legal_moves(tensor):
	"""
	Fills move planes 19-34 from the piece planes already in the tensor.

	Knight, king, slider and pawn attack planes mark every square the piece attacks;
	slider rays stop at (and include) the first occupied square. Pawn push planes only
	mark empty squares, and a blocked single push also blocks the double push.
	"""
	occupied = tensor[:, :, 0:7].any(axis=2) | tensor[:, :, 9:16].any(axis=2)

	sides = (
		(0, HERO_MOVE_PLANES, PAWN_ATTACK_PLANES[0], True),  # Hero pawns move toward row 0
		(9, VILLAIN_MOVE_PLANES, PAWN_ATTACK_PLANES[1], False),
	)

	for channel_offset, move_planes, attack_plane, toward_row_0 in sides:
		for piece, channel in PIECE_CHANNELS.items():
			plane = move_planes[piece]
			rows, cols = np.nonzero(tensor[:, :, channel + channel_offset])

			for square in (int(r) * 8 + int(c) for r, c in zip(rows, cols)):
				if piece == "P":
					for r, c in PAWN_ATTACKS[toward_row_0][square]:
						tensor[r, c, attack_plane] = 1
					for r, c in PAWN_MOVES[toward_row_0].get(square, []):
						if occupied[r, c]:
							break
						tensor[r, c, plane] = 1
				elif piece == "N":
					for r, c in KNIGHT_MOVES[square]:
						tensor[r, c, plane] = 1
				elif piece == "K":
					for r, c in KING_MOVES[square]:
						tensor[r, c, plane] = 1
				else:
					for ray in SLIDING_MOVES[SLIDING_TABLES[piece]][square]:
						for r, c in ray:
							tensor[r, c, plane] = 1
							if occupied[r, c]:
								break

	return tensor
