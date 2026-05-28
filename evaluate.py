import chess
import torch
import torch.nn.functional as F
from model import ChessCNN
from data import board_to_tensor, label_to_move

def load_model(path="chess_model.pt", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessCNN().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model, device

def predict_move(model, board, device, top_k=5):
    """Predict the best legal move for the current board position.
    
    We get logits for all 4096 from/to pairs, filter to only legal moves,
    then pick the highest-scoring legal move.
    """
    tensor = board_to_tensor(board).unsqueeze(0).to(device)  # (1, 12, 8, 8)
    with torch.no_grad():
        logits = model(tensor).squeeze(0)  # (4096,)

    # Get all legal moves and find the highest-scoring one
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None

    best_move, best_score = None, float("-inf")
    for move in legal_moves:
        label = move.from_square * 64 + move.to_square
        score = logits[label].item()
        if score > best_score:
            best_score = score
            best_move = move

    return best_move

def play_game(model, device, you_play="white"):
    """Play an interactive game against the trained model in the terminal."""
    board = chess.Board()
    your_color = chess.WHITE if you_play == "white" else chess.BLACK

    print("\nStarting game! Enter moves in UCI format (e.g. e2e4, g1f3)")
    print("Type 'quit' to exit, 'board' to reprint the board.\n")

    while not board.is_game_over():
        print(board)
        print()

        if board.turn == your_color:
            # Human turn
            while True:
                move_str = input("Your move: ").strip()
                if move_str == "quit":
                    return
                if move_str == "board":
                    print(board)
                    continue
                try:
                    move = chess.Move.from_uci(move_str)
                    if move in board.legal_moves:
                        board.push(move)
                        break
                    else:
                        print("Illegal move, try again.")
                except ValueError:
                    print("Invalid format. Use UCI notation like e2e4.")
        else:
            # Model turn
            move = predict_move(model, board, device)
            print(f"Model plays: {move.uci()}")
            board.push(move)

    print(board)
    print(f"\nGame over: {board.result()}")

def top1_accuracy(model, device, pgn_path, max_games=200):
    """Measure top-1 move prediction accuracy on held-out games."""
    from data import parse_pgn
    correct, total = 0, 0
    model.eval()

    for board_tensor, true_label in parse_pgn(pgn_path, max_games):
        inp = board_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(inp).squeeze(0)
        pred = logits.argmax().item()
        if pred == true_label:
            correct += 1
        total += 1

    print(f"Top-1 accuracy over {total} positions: {correct/total:.3f} ({correct}/{total})")

if __name__ == "__main__":
    import sys
    model, device = load_model()
    if len(sys.argv) > 1 and sys.argv[1] == "accuracy":
        pgn = sys.argv[2] if len(sys.argv) > 2 else "dataset/lichess.pgn"
        top1_accuracy(model, device, pgn)
    else:
        play_game(model, device)
