import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import ChessCNN
from data import parse_pgn
import os

def build_dataset(pgn_path, max_games=5000):
    """Load PGN and convert all positions to tensors."""
    print(f"Building dataset from {pgn_path} (max {max_games} games)...")
    boards, labels = [], []
    for board_tensor, label in parse_pgn(pgn_path, max_games):
        boards.append(board_tensor)
        labels.append(label)
    print(f"  Total positions: {len(boards)}")
    return TensorDataset(torch.stack(boards), torch.tensor(labels))

def train(pgn_path="dataset/lichess.pgn", max_games=5000, epochs=20, batch_size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = build_dataset(pgn_path, max_games)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model     = ChessCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Why CrossEntropyLoss and not MSE?
    # We're doing classification over 4096 classes, not regression.
    # CrossEntropyLoss = log-softmax + negative log likelihood,
    # which is the standard loss for multi-class classification.

    print(f"\nStarting training: {epochs} epochs, batch size {batch_size}")
    print("-" * 50)

    for epoch in range(epochs):
        model.train()
        total_loss, correct = 0, 0

        for boards, labels in loader:
            boards, labels = boards.to(device), labels.to(device)

            logits = model(boards)
            loss   = criterion(logits, labels)

            # Why zero_grad()? PyTorch accumulates gradients by default.
            # We zero them each step so previous batches don't interfere.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct    += (logits.argmax(1) == labels).sum().item()

        acc = correct / len(dataset)
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1:2d}/{epochs} | loss: {avg_loss:.4f} | acc: {acc:.3f}")

    torch.save(model.state_dict(), "chess_model.pt")
    print("\nModel saved to chess_model.pt")

if __name__ == "__main__":
    import sys
    pgn = sys.argv[1] if len(sys.argv) > 1 else "dataset/lichess.pgn"
    if not os.path.exists(pgn):
        print(f"ERROR: PGN file not found at '{pgn}'")
        print("Download a dataset from: https://database.lichess.org/")
        print(f"Then place it at: {pgn}")
        sys.exit(1)
    train(pgn_path=pgn)
