import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from model import ChessCNN
from data import parse_pgn_chunk

CHUNK_SIZE = 6000          # games loaded per run
PROGRESS_FILE = "progress.json"
TOTAL_GAMES = 3000000       # approx games in your file; used for wrap-around

def get_start_game():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            data = json.load(f)
        return data.get("next_start", 0), data.get("run", 0)
    return 0, 0

def save_progress(next_start, run):
    if next_start >= TOTAL_GAMES:
        next_start = 0  # wrap back to beginning
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"next_start": next_start, "run": run}, f)

def build_dataset(pgn_path, start_game, num_games):
    print(f"Building dataset: games {start_game} to {start_game+num_games}...")
    boards, labels = [], []
    for board_tensor, label in parse_pgn_chunk(pgn_path, start_game, num_games):
        boards.append(board_tensor)
        labels.append(label)
    print(f"  Total positions: {len(boards)}")
    return TensorDataset(torch.stack(boards), torch.tensor(labels))

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for boards, labels in loader:
            boards, labels = boards.to(device), labels.to(device)
            logits = model(boards)
            total_loss += criterion(logits, labels).item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), correct / total

def train(pgn_path="dataset/lichess.pgn", epochs=20, batch_size=64,
          patience=4, lr=1e-3, weight_decay=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    start_game, run = get_start_game()
    run += 1
    print(f">>> Training run #{run} | starting at game {start_game}")

    full_dataset = build_dataset(pgn_path, start_game, CHUNK_SIZE)

    val_size = int(len(full_dataset) * 0.2)
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])
    print(f"  Train: {train_size} positions | Validation: {val_size} positions")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    model = ChessCNN().to(device)
    if os.path.exists("chess_model.pt"):
        model.load_state_dict(torch.load("chess_model.pt", map_location=device))
        print(">>> Loaded existing model — continuing training")
    else:
        print(">>> No existing model found — starting fresh")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for boards, labels in train_loader:
            boards, labels = boards.to(device), labels.to(device)
            logits = model(boards)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"train_loss: {train_loss:.4f} train_acc: {train_acc:.3f} | "
              f"val_loss: {val_loss:.4f} val_acc: {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), "chess_model.pt")
            print(f"   >>> New best val_acc {val_acc:.3f} — model saved")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"   >>> Early stopping (no improvement for {patience} epochs)")
                break

    save_progress(start_game + CHUNK_SIZE, run)
    print(f">>> Run #{run} done. Best val_acc: {best_val_acc:.3f}")
    print(f">>> Next run will start at game {start_game + CHUNK_SIZE}")

if __name__ == "__main__":
    import sys
    pgn = sys.argv[1] if len(sys.argv) > 1 else "dataset/lichess.pgn"
    if not os.path.exists(pgn):
        print(f"ERROR: PGN file not found at '{pgn}'")
        sys.exit(1)
    train(pgn_path=pgn)
