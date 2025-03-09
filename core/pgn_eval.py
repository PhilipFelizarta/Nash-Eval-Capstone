import chess.pgn
import numpy as np
import matplotlib.pyplot as plt

def extract_wdl_from_pgn(pgn_path):
	"""
	Reads a PGN file and extracts WDL probabilities for each move.

	Args:
		pgn_path (str): Path to the PGN file.

	Returns:
		list: WDL values for White's perspective.
		list: WDL values for Black's perspective.
		list: Move numbers.
	"""
	with open(pgn_path, "r", encoding="utf-8") as pgn_file:
		game = chess.pgn.read_game(pgn_file)

	white_wdl = []
	black_wdl = []
	move_numbers = []

	board = game.board()
	move_count = 0

	for node in game.mainline():
		move_count += 1
		move_numbers.append(move_count)

		# Extract WDL from comment
		if node.comment.startswith("WDL:"):
			wdl_values = node.comment.split("WDL:")[1].strip().split("/")
			wdl_values = list(map(float, wdl_values))  # Convert to float

			if board.turn == chess.WHITE:
				white_wdl.append(wdl_values)
			else:
				black_wdl.append(wdl_values)

		board.push(node.move)

	return np.array(white_wdl), np.array(black_wdl), move_numbers


def plot_wdl_area_chart(pgn_path, save_path):
	"""
	Reads a PGN and plots WDL area charts for White and Black.

	Args:
		pgn_path (str): Path to the PGN file.
	"""
	white_wdl, black_wdl, move_numbers = extract_wdl_from_pgn(pgn_path)

	if len(white_wdl) == 0 or len(black_wdl) == 0:
		print("No WDL data found in PGN.")
		return

	# Create figure with two subplots
	fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)

	# **Plot White's WDL**
	axes[0].stackplot(
		np.arange(white_wdl.shape[0])+1, white_wdl[:, 0], white_wdl[:, 1], white_wdl[:, 2],
		labels=["Win", "Draw", "Loss"], colors=["green", "blue", "red"], alpha=0.6
	)
	axes[0].set_ylabel("White's Probability")
	axes[0].set_title("Win-Draw-Loss Probability (White)")
	axes[0].legend(loc="upper right")

	# **Plot Black's WDL**
	axes[1].stackplot(
		np.arange(black_wdl.shape[0])+1, black_wdl[:, 0], black_wdl[:, 1], black_wdl[:, 2],
		labels=["Win", "Draw", "Loss"], colors=["green", "blue", "red"], alpha=0.6
	)
	axes[1].set_xlabel("Move Number")
	axes[1].set_ylabel("Black's Probability")
	axes[1].set_title("Win-Draw-Loss Probability (Black)")
	axes[1].legend(loc="upper right")

	plt.tight_layout()
	plt.savefig(save_path, dpi=300, bbox_inches="tight")
	plt.close()