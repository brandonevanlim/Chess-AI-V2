# Chess AI — Stage 1: Supervised Learning

A CNN trained to predict moves from real chess games.

## Setup

```bash
pip install -r requirements.txt
```

## Get Training Data

Download a PGN file from Lichess open database:
https://database.lichess.org/

Place the file at `dataset/lichess.pgn`.
A monthly file has millions of games — start with a small one (e.g. 2013-01).

## Train

```bash
python train.py dataset/lichess.pgn
```

Training runs for 20 epochs and saves `chess_model.pt`.

## Play Against It

```bash
python evaluate.py
```

Enter moves in UCI format: `e2e4`, `g1f3`, etc.

## Check Accuracy

```bash
python evaluate.py accuracy dataset/lichess.pgn
```

## Project Structure

| File | Purpose |
|------|---------|
| `data.py` | PGN parsing + board → tensor encoding |
| `model.py` | CNN architecture (ResNet-style) |
| `train.py` | Training loop |
| `evaluate.py` | Play against the model, check accuracy |

## Things to Understand Before Running

**data.py**
- Why 12 channels? (6 piece types × 2 colors)
- Why `from_square * 64 + to_square` as the label?
- What moves does this encoding miss? (promotions)

**model.py**
- Why does `x + residual` help training? (vanishing gradients)
- What does BatchNorm do?

**train.py**
- Why CrossEntropyLoss and not MSE?
- Why `logits.argmax(1)` for accuracy?
- What does `optimizer.zero_grad()` do and why is it necessary?

Write down your answers — that reflection is the whole point.

## Next Steps (Stage 2)

Add a value head to estimate who's winning from a given board position.
Train it on game outcomes: 1.0 = white wins, 0.0 = black wins, 0.5 = draw.
