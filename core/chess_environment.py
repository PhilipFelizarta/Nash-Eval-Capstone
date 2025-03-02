import chess
import numpy as np

def fen_to_example(fen):
    """
    Converts a FEN string into a (8, 8, 14) tensor where:
        - First 6 channels: Current player's pieces
        - Channels 7 & 8: Current player's kingside & queenside castling rights
        - Next 6 channels: Opponent's pieces
        - Channels 13 & 14: Opponent's kingside & queenside castling rights

    Args:
        fen (str): Chess position in FEN notation.

    Returns:
        np.ndarray: (8, 8, 14) tensor representation of the board.
    """
    board = chess.Board(fen)
    tensor = np.zeros((8, 8, 14), dtype=np.float32)

    # Define piece indices (relative to the current player)
    piece_map = {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5}
    opponent_offset = 8  # Opponent pieces are stored 8 indices later

    # Identify the player to move
    is_white_turn = board.turn

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row, col = divmod(square, 8)
            piece_index = piece_map.get(piece.symbol().upper(), None)

            if piece_index is not None:
                if (piece.color and is_white_turn) or (not piece.color and not is_white_turn):
                    # Current player
                    tensor[row, col, piece_index] = 1
                else:
                    # Opponent
                    tensor[row, col, piece_index + opponent_offset] = 1

    # Encode castling rights
    if is_white_turn:
        tensor[:, :, 6] = board.has_kingside_castling_rights(chess.WHITE)  # Current player's kingside castling
        tensor[:, :, 7] = board.has_queenside_castling_rights(chess.WHITE)  # Current player's queenside castling
        tensor[:, :, 12] = board.has_kingside_castling_rights(chess.BLACK)  # Opponent's kingside castling
        tensor[:, :, 13] = board.has_queenside_castling_rights(chess.BLACK)  # Opponent's queenside castling
    else:
        tensor[:, :, 6] = board.has_kingside_castling_rights(chess.BLACK)  # Current player's kingside castling
        tensor[:, :, 7] = board.has_queenside_castling_rights(chess.BLACK)  # Current player's queenside castling
        tensor[:, :, 12] = board.has_kingside_castling_rights(chess.WHITE)  # Opponent's kingside castling
        tensor[:, :, 13] = board.has_queenside_castling_rights(chess.WHITE)  # Opponent's queenside castling

    return tensor

# Example usage
if __name__ == "__main__":
    example_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    tensor = fen_to_example(example_fen)
    print("Tensor shape:", tensor.shape)  # Should print (8, 8, 14)
    print("Piece Tensor Representation:\n", tensor[:, :, :6])  # Print player piece channels
