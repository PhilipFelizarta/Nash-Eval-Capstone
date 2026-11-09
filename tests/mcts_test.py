import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "cpp", "mcts")))

import chess
import numpy as np
import pytest

nash_mcts = pytest.importorskip("nash_mcts", reason="build the extension first: python cpp/mcts/build.py")

from core import mcts
from core.chess_environment import fast_fen_to_example
from core.move_encoding import POLICY_SIZE, move_to_index


# ---------------------------------------------------------------------------------------------
# Stub evaluators (no TensorFlow): evaluate(batch) -> (policy_logits (k, 4672), wdl (k, 3))

def uniform_draw(x):
	"""Uniform priors, every position a dead draw."""
	return np.zeros((len(x), POLICY_SIZE), np.float32), np.tile([0.0, 1.0, 0.0], (len(x), 1)).astype(np.float32)


def constant(wdl):
	def evaluate(x):
		return np.zeros((len(x), POLICY_SIZE), np.float32), np.tile(wdl, (len(x), 1)).astype(np.float32)
	return evaluate


def loves_g4_pawn(x):
	"""Like LovesG4Pawn in search_test.py: a hero pawn on g4 looks nearly winning, so 2.g4?? is bait."""
	has_g4_pawn = x[:, 4, 6, 0] == 1
	wdl = np.where(has_g4_pawn[:, None], [0.9, 0.1, 0.0], [0.2, 0.6, 0.2]).astype(np.float32)
	return np.zeros((len(x), POLICY_SIZE), np.float32), wdl


class KerasLikeModel:
	"""Quacks like a Keras model with dict outputs."""
	layers = []

	def __init__(self):
		self.calls = 0

	def predict(self, x, **kwargs):
		raise AssertionError("the driver should call the model directly")

	def __call__(self, x, training=None):
		assert training is False
		self.calls += 1
		policy, value = uniform_draw(np.asarray(x))
		return {"policy": policy, "value": value}


# ---------------------------------------------------------------------------------------------
# Positions

TRICKY_FENS = [
	chess.STARTING_FEN,
	"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",  # Castling both sides
	"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1",
	"r3k2r/8/8/8/8/8/8/R3K2R w Kq - 0 1",  # Partial castling rights
	"r3k2r/8/8/8/8/8/8/R3K2R b Qk - 0 1",
	"n1n5/PPPk4/8/8/8/8/4Kppp/5N1N w - - 0 1",  # Promotions and under-promotions with captures
	"n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1",
	"8/8/8/2k5/3Pp3/8/8/4K3 b - d3 0 1",  # En passant
	"rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
	"8/8/8/8/k2Pp2Q/8/8/3K4 b - d3 0 1",  # En passant square in the FEN, but the capture is illegal (pin)
	"4k3/8/8/8/8/8/8/R3K2R w KQkq - 0 1",  # Black castling rights without the pieces
	"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
	"rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
	"6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1",
]


def random_positions(n, seed=0):
	"""Every position along seeded random games, until n positions (both sides to move)."""
	rng = random.Random(seed)
	boards = []
	while len(boards) < n:
		board = chess.Board()
		for _ in range(rng.randint(1, 160)):
			moves = list(board.legal_moves)
			if not moves:
				break
			board.push(rng.choice(moves))
			boards.append(board.copy(stack=False))
	return boards[:n]


@pytest.fixture(scope="module")
def parity_boards():
	return [chess.Board(fen) for fen in TRICKY_FENS] + random_positions(2400)


# ---------------------------------------------------------------------------------------------
# Encoder and move indexing parity

def test_encoder_parity(parity_boards):
	sides = set()
	for board in parity_boards:
		fen = board.fen()
		ours, ref = nash_mcts.encode(fen), fast_fen_to_example(fen)
		assert ours.shape == (8, 8, 36) and ours.dtype == np.float32
		assert np.array_equal(ours[..., :35], ref[..., :35]), fen
		assert np.abs(ours[..., 35] - ref[..., 35]).max() <= 1e-6, fen
		sides.add(board.turn)
	assert sides == {chess.WHITE, chess.BLACK}


def test_encoder_parity_on_raw_fen_fields():
	# fast_fen_to_example reads castling / en passant verbatim, so encode() must too
	fens = TRICKY_FENS + [
		"8/8/8/8/8/8/8/8 w - - 0 1",
		"8/8/8/8/8/8/8/8 b KQkq e3 0 1",
		"P6p/8/8/8/8/8/8/p6P w - - 0 1",  # Pawns on the back ranks
		"P6p/8/8/8/8/8/8/p6P b - - 0 1",
		"bBbB4/8/8/8/8/8/8/4BbBb b Kq - 0 1",  # Bishops on both colours, both sides
		"bBbB4/8/8/8/8/8/8/4BbBb w kQ a6 0 1",
	]
	for fen in fens:
		assert np.array_equal(nash_mcts.encode(fen), fast_fen_to_example(fen)), fen
	batch = nash_mcts.encode_batch(fens)
	assert np.array_equal(batch, np.stack([fast_fen_to_example(f) for f in fens]))


def test_legal_moves_parity(parity_boards):
	for board in parity_boards:
		ours = sorted(nash_mcts.legal_moves(board.fen()))
		ref = sorted((m.uci(), move_to_index(m, board.turn)) for m in board.legal_moves)
		assert ours == ref, board.fen()


def test_castling_is_king_two_squares():
	moves = dict(nash_mcts.legal_moves("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"))
	assert "e1g1" in moves and "e1c1" in moves
	assert moves["e1g1"] == move_to_index(chess.Move.from_uci("e1g1"), chess.WHITE)


def leaf_boards(root_fen, history, paths):
	for path in paths:
		board = chess.Board(root_fen)
		for uci in list(history) + list(path):
			board.push_uci(uci)
		yield board


def test_tree_leaves_encode_like_python_chess():
	# Leaves inside the tree must encode exactly as fast_fen_to_example(board.fen()) of the same game
	history = ["e2e4", "c7c5", "e4e5", "d7d5"]  # e5xd6 en passant is available at the root
	for fen, hist in [(chess.STARTING_FEN, history), (TRICKY_FENS[2], []), (TRICKY_FENS[5], [])]:
		tree = nash_mcts.MCTS(seed=1)
		tree.reset(fen, hist)
		checked = 0
		for _ in range(12):
			x = tree.gather(16)
			for leaf, board in zip(x, leaf_boards(fen, hist, tree.pending_paths())):
				assert np.array_equal(leaf, fast_fen_to_example(board.fen())), board.fen()
				checked += 1
			k = len(x)
			rng = np.random.default_rng(checked)
			tree.backup(rng.normal(size=(k, POLICY_SIZE)).astype(np.float32), np.tile([0.3, 0.4, 0.3], (k, 1)).astype(np.float32))
		assert checked > 50


# ---------------------------------------------------------------------------------------------
# Search behaviour

def stats_by_move(tree):
	return {s[0]: s for s in tree.root_stats()}


@pytest.mark.parametrize("fen, mate", [
	("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", "a1a8"),
	("r5k1/8/8/8/8/8/5PPP/6K1 b - - 0 1", "a8a1"),
])
def test_finds_mate_in_one_with_uniform_evaluator(fen, mate):
	tree = mcts.run_mcts(fen, uniform_draw, num_simulations=200, batch_size=16, add_noise=False)
	assert tree.select_move(0.0) == mate
	assert tree.root_proven() == "win"
	assert stats_by_move(tree)[mate][3] == 1.0
	move, _ = mcts.select_mcts_move(chess.Board(fen), uniform_draw, num_simulations=200, batch_size=16)
	assert move == chess.Move.from_uci(mate)


def test_mate_dominates_even_with_temperature_and_noise():
	tree = mcts.run_mcts("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", uniform_draw, 200, 16, add_noise=True, seed=3)
	assert all(tree.select_move(1.0) == "a1a8" for _ in range(10))


def test_avoids_mate_in_one_bait():
	# After 1.f3 e5, 2.g4?? allows Qh4#. The evaluator loves g4, only the search can refute it.
	board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/8/5P2/PPPPP1PP/RNBQKBNR w KQkq - 0 2")
	tree = mcts.run_mcts(board, loves_g4_pawn, num_simulations=400, batch_size=16, add_noise=False)
	assert tree.select_move(0.0) != "g2g4"
	stats = stats_by_move(tree)
	assert stats["g2g4"][3] == 0.0
	assert tree.root_move_proven()[[s[0] for s in tree.root_stats()].index("g2g4")] == "loss"
	moves, _, visits = mcts.visit_distribution(tree)
	assert visits[moves.index("g2g4")] == 0  # Not a policy target once refuted


def test_stalemate_is_a_draw():
	# b5-b6 stalemates the black king on a8
	tree = mcts.run_mcts("k7/2K5/8/1P6/8/8/8/8 w - - 0 1", constant([1.0, 0.0, 0.0]), 200, 16, add_noise=False)
	stats = stats_by_move(tree)
	assert stats["b5b6"][2] >= 1 and stats["b5b6"][3] == 0.5

	tree = nash_mcts.MCTS()
	tree.reset("k7/2K5/1P6/8/8/8/8/8 b - - 0 1")  # Root is stalemate
	assert len(tree.gather(8)) == 0
	assert tree.root_terminal() and tree.root_visits() == 8 and tree.root_value() == 0.5


def test_insufficient_material_and_fifty_moves_are_draws():
	tree = mcts.run_mcts("8/8/8/8/8/k7/2p5/1K6 w - - 0 1", constant([1.0, 0.0, 0.0]), 100, 8, add_noise=False)
	assert stats_by_move(tree)["b1c2"][3] == 0.5  # Kxc2 leaves bare kings
	assert tree.root_move_proven()[[s[0] for s in tree.root_stats()].index("b1c2")] == "draw"

	# Half-move clock 99: every rook or king move (none are zeroing) hits the fifty-move rule
	tree = mcts.run_mcts("8/8/8/4k3/8/8/R7/4K3 w - - 99 80", constant([1.0, 0.0, 0.0]), 100, 8, add_noise=False)
	assert tree.root_proven() == "draw"
	assert all(s[3] == 0.5 for s in tree.root_stats())


def test_threefold_repetition_uses_history():
	history = ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"]
	board = chess.Board()
	for uci in history + ["f6g8"]:
		board.push_uci(uci)
	assert board.is_repetition(3)

	def f6g8(tree):
		moves = [st[0] for st in tree.root_stats()]
		return tree.root_stats()[moves.index("f6g8")], tree.root_move_proven()[moves.index("f6g8")]

	tree = mcts.run_mcts(chess.STARTING_FEN, constant([0.0, 0.0, 1.0]), 100, 8, history_uci=history, add_noise=False)
	stat, proven = f6g8(tree)
	assert proven == "draw" and stat[2] >= 1 and stat[3] == 0.5

	# Same thing from a chess.Board: its move_stack is the history
	board.pop()
	tree = mcts.run_mcts(board, constant([0.0, 0.0, 1.0]), 100, 8, add_noise=False)
	assert f6g8(tree)[1] == "draw"

	# Without the history the same position is not a repetition
	tree = mcts.run_mcts(board.fen(), constant([0.0, 0.0, 1.0]), 100, 8, add_noise=False)
	stat, proven = f6g8(tree)
	assert proven == "unknown" and stat[2] >= 1


@pytest.mark.parametrize("fen", [chess.STARTING_FEN, TRICKY_FENS[2]])  # White and black to move
@pytest.mark.parametrize("wdl, leaf_score", [([0.0, 0.0, 1.0], 0.0), ([1.0, 0.0, 0.0], 1.0), ([0.2, 0.6, 0.2], 0.5)])
def test_value_sign_convention(fen, wdl, leaf_score):
	# WDL index 0 = the side to move at the leaf wins; root Q is for the root's side to move,
	# edge Q is for the player who made the move.
	tree = nash_mcts.MCTS()
	tree.reset(fen, [], False)
	tree.backup(*constant(wdl)(tree.gather(1)))
	assert tree.root_visits() == 1 and tree.root_value() == pytest.approx(leaf_score)
	tree.backup(*constant(wdl)(tree.gather(1)))  # One child, whose side to move is the opponent
	(child,) = [st for st in tree.root_stats() if st[2] == 1]
	assert child[3] == pytest.approx(1.0 - leaf_score)
	assert tree.root_value() == pytest.approx((leaf_score + 1.0 - leaf_score) / 2)


# ---------------------------------------------------------------------------------------------
# Batching, virtual loss, collisions

def test_visits_and_virtual_loss_bookkeeping():
	tree = nash_mcts.MCTS(seed=0)
	tree.reset(chess.STARTING_FEN)
	for _ in range(20):
		before = tree.root_visits()
		x = tree.gather(32)
		stats = tree.last_gather_stats()
		assert stats["selections"] == 32
		assert stats["leaves"] == len(x)
		assert stats["leaves"] + stats["terminals"] + stats["collisions"] == stats["selections"]
		if len(x):
			assert tree.total_virtual_loss() > 0
		tree.backup(*uniform_draw(x))
		assert tree.total_virtual_loss() == 0
		assert tree.root_visits() == before + len(x) + stats["terminals"]
	assert sum(s[2] for s in tree.root_stats()) == tree.root_visits() - 1  # The root's own evaluation


def test_collisions_when_batch_exceeds_root_moves():
	fen = "k7/8/8/8/8/8/8/K6R b - - 0 1"  # Black has 3 legal moves
	root = chess.Board(fen)
	assert root.legal_moves.count() == 3

	tree = nash_mcts.MCTS(seed=0)
	tree.reset(fen)
	x = tree.gather(64)  # Fresh tree: only the root can be a leaf
	assert len(x) == 1 and tree.last_gather_stats()["collisions"] == 63
	tree.backup(*uniform_draw(x))
	assert tree.total_virtual_loss() == 0 and tree.root_visits() == 1

	x = tree.gather(64)  # Only the 3 children are reachable
	assert len(x) == 3 and tree.last_gather_stats()["collisions"] == 61
	tree.backup(*uniform_draw(x))
	assert tree.total_virtual_loss() == 0 and tree.root_visits() == 4

	for _ in range(15):
		nodes_before = tree.num_nodes()
		x = tree.gather(64)
		paths = [tuple(p) for p in tree.pending_paths()]
		assert len(paths) == len(x) == len(set(paths))  # Distinct leaves, no duplicates in the batch
		# Each leaf is expanded exactly once: the node count grows by exactly their legal move counts
		expected_children = sum(b.legal_moves.count() for b in leaf_boards(fen, [], paths))
		tree.backup(*uniform_draw(x))
		assert tree.num_nodes() == nodes_before + expected_children
		assert tree.total_virtual_loss() == 0
	assert sum(s[2] for s in tree.root_stats()) == tree.root_visits() - 1


def test_gather_twice_without_backup_fails_and_cancel_restores():
	tree = nash_mcts.MCTS()
	tree.reset(chess.STARTING_FEN)
	tree.backup(*uniform_draw(tree.gather(1)))
	x = tree.gather(8)
	with pytest.raises(RuntimeError):
		tree.gather(8)
	with pytest.raises(ValueError):
		tree.backup(*uniform_draw(x[:-1]))
	tree.cancel()
	assert tree.total_virtual_loss() == 0 and tree.num_pending() == 0
	x = tree.gather(8)
	assert len(x) == 8  # The cancelled leaves are selectable again
	tree.backup(*uniform_draw(x))


def test_noise_flag():
	def priors(add_noise, seed):
		tree = nash_mcts.MCTS(seed=seed)
		tree.reset(chess.STARTING_FEN, [], add_noise)
		tree.backup(*uniform_draw(tree.gather(1)))
		return np.array([s[4] for s in tree.root_stats()])

	assert np.allclose(priors(False, 0), 1 / 20)
	noisy = priors(True, 0)
	assert not np.allclose(noisy, 1 / 20) and np.isclose(noisy.sum(), 1.0, atol=1e-5)
	assert np.allclose(noisy, priors(True, 0))  # Seeded


# ---------------------------------------------------------------------------------------------
# Tree reuse

def test_advance_keeps_the_subtree():
	tree = mcts.run_mcts(chess.STARTING_FEN, uniform_draw, 300, 16, add_noise=False)
	move = tree.select_move(0.0)
	child_visits = stats_by_move(tree)[move][2]
	assert child_visits > 1
	tree.advance(move)
	assert tree.root_visits() == child_visits
	board = chess.Board()
	board.push_uci(move)
	assert tree.root_fen().split()[:2] == board.fen().split()[:2]
	assert sum(s[2] for s in tree.root_stats()) == child_visits - 1
	assert sorted(s[0] for s in tree.root_stats()) == sorted(m.uci() for m in board.legal_moves)

	# Keep searching from the reused root; leaves still encode like python-chess with the full history
	mcts.run_mcts(None, uniform_draw, child_visits + 64, 16, tree=tree)
	x = tree.gather(16)
	for leaf, b in zip(x, leaf_boards(chess.STARTING_FEN, [move], tree.pending_paths())):
		assert np.array_equal(leaf, fast_fen_to_example(b.fen()))
	tree.backup(*uniform_draw(x))


def test_advance_outside_the_tree_resets():
	tree = nash_mcts.MCTS()
	tree.reset(chess.STARTING_FEN)
	tree.advance("e2e4")  # Root never expanded
	assert tree.root_visits() == 0 and tree.root_fen().split()[1] == "b"
	tree.backup(*uniform_draw(tree.gather(1)))
	assert len(tree.root_stats()) == 20
	with pytest.raises(ValueError):
		tree.advance("e2e5")


def test_advance_tracks_repetition_history():
	tree = nash_mcts.MCTS()
	tree.reset(chess.STARTING_FEN, [], False)
	for uci in ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"]:
		tree.advance(uci)
	mcts.search(tree, mcts.make_evaluator(constant([0.0, 0.0, 1.0])), 100, 8)
	assert stats_by_move(tree)["f6g8"][3] == 0.5


# ---------------------------------------------------------------------------------------------
# Drivers

def test_keras_like_model_and_multi_tree_batching():
	model = KerasLikeModel()
	tree = mcts.run_mcts(chess.STARTING_FEN, model, num_simulations=64, batch_size=16)
	assert tree.root_visits() == 64 and model.calls > 0

	calls = []

	def counting(x):
		calls.append(len(x))
		return uniform_draw(x)

	trees = []
	for i, fen in enumerate([chess.STARTING_FEN, TRICKY_FENS[1], TRICKY_FENS[2]]):
		t = nash_mcts.MCTS(seed=i)
		t.reset(fen)
		trees.append(t)
	mcts.MultiMCTS(trees).search(counting, num_simulations=100, batch_size=16)
	assert all(t.root_visits() >= 100 for t in trees)
	assert all(t.total_virtual_loss() == 0 for t in trees)
	assert max(calls) > 16  # Leaves of several trees shared one network call


def test_play_game_mcts_writes_pgn(tmp_path):
	path = tmp_path / "game.pgn"
	result, records = mcts.play_game_mcts(uniform_draw, uniform_draw, str(path), num_simulations=32, batch_size=8, max_moves=3)
	assert result == "*" and len(records) == 7
	assert '[Result "*"]' in path.read_text()
	for record in records:
		assert record["move"] in record["moves"] and record["visits"].sum() > 0

	result, records = mcts.play_game_mcts(uniform_draw, uniform_draw, str(path), num_simulations=64, batch_size=8,
										  starting_fen="6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", add_noise=False)
	assert result == "1-0" and [r["move"] for r in records] == ["a1a8"]
	assert '[Result "1-0"]' in path.read_text()


def test_principal_variation_and_depth():
	tree = nash_mcts.MCTS(dirichlet_epsilon=0.0, seed=0)
	tree.reset(chess.STARTING_FEN, [], False)
	assert tree.principal_variation() == [] and tree.max_depth() == 0
	while tree.root_visits() < 400:
		x = tree.gather(16)
		tree.backup(np.zeros((len(x), nash_mcts.POLICY_SIZE), np.float32), np.tile(np.float32([0, 1, 0]), (len(x), 1)))
	pv = tree.principal_variation(8)
	assert 1 <= len(pv) <= 8 and tree.max_depth() >= len(pv) - 1
	board = chess.Board()
	for uci in pv:
		move = chess.Move.from_uci(uci)
		assert move in board.legal_moves
		board.push(move)
	assert pv[0] == tree.select_move(0.0)
	tree.advance(pv[0])  # reused subtree keeps parent-before-child order
	assert tree.principal_variation(7) == pv[1:8] and tree.max_depth() >= 0
