import chess
import chess.pgn
import torch
import numpy as np

PIECE_TO_CHANNEL = {
    (chess.PAWN,   chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK,   chess.WHITE): 3,
    (chess.QUEEN,  chess.WHITE): 4,
    (chess.KING,   chess.WHITE): 5,
    (chess.PAWN,   chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK,   chess.BLACK): 9,
    (chess.QUEEN,  chess.BLACK): 10,
    (chess.KING,   chess.BLACK): 11,
}

def board_to_tensor(board):
    """Convert a chess.Board to a (12, 8, 8) float tensor.
    
    Each of the 12 channels represents one piece type per color:
      Channels 0-5: white pawns, knights, bishops, rooks, queens, kings
      Channels 6-11: black pawns, knights, bishops, rooks, queens, kings
    A cell is 1.0 if that piece occupies that square, 0.0 otherwise.
    """
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        channel = PIECE_TO_CHANNEL[(piece.piece_type, piece.color)]
        row = square // 8
        col = square % 8
        tensor[channel, row, col] = 1.0
    return torch.tensor(tensor).float()

def move_to_label(move):
    """Encode a chess.Move as an integer 0-4095.
    
    We have 64 possible from-squares and 64 possible to-squares,
    giving 64 * 64 = 4096 total combinations.
    
    NOTE: This encoding does NOT handle promotions (e.g. pawn promoting
    to a queen vs knight). That would require extending the label space.
    """
    return move.from_square * 64 + move.to_square

def label_to_move(label):
    """Decode an integer label back into from/to squares."""
    from_square = label // 64
    to_square = label % 64
    return chess.Move(from_square, to_square)

def parse_pgn(pgn_path, max_games=10000):
    """Yields (board_tensor, move_label) pairs from a PGN file.
    
    Each position encountered in every game becomes one training sample.
    Both sides' moves are included (the model learns for both colors).
    """
    with open(pgn_path) as f:
        games_parsed = 0
        while games_parsed < max_games:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            board = game.board()
            for move in game.mainline_moves():
                yield board_to_tensor(board), move_to_label(move)
                board.push(move)
            games_parsed += 1
            if games_parsed % 500 == 0:
                print(f"  Parsed {games_parsed} games...")

def count_games(pgn_path):
    """Count total games in the PGN file (used for wrap-around)."""
    count = 0
    with open(pgn_path) as f:
        while True:
            offset = chess.pgn.read_game(f)
            if offset is None:
                break
            count += 1
    return count

def parse_pgn_chunk(pgn_path, start_game=0, num_games=20000):
    """Yields (board_tensor, move_label) pairs from a chunk of games.

    Skips the first `start_game` games, then reads `num_games` games.
    This lets each training run read a different slice of the file.
    """
    with open(pgn_path) as f:
        for _ in range(start_game):
            if chess.pgn.read_game(f) is None:
                return
        games_parsed = 0
        while games_parsed < num_games:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            board = game.board()
            for move in game.mainline_moves():
                yield board_to_tensor(board), move_to_label(move)
                board.push(move)
            games_parsed += 1
            if games_parsed % 500 == 0:
                print(f"  Parsed {games_parsed} games (from game {start_game})...")
