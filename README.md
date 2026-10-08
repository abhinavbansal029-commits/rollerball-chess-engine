# Rollerball Chess Engine

A game-playing AI for **Rollerball**, a chess variant played on a 7×7 board with the central 3×3 block removed. Pieces move around the 40-square ring that's left.

## Rules implemented

- Each side has a King, Rook, Bishop and two Pawns, set up symmetrically (White at the bottom, Black at the top).
- **Rook / Bishop / King** move as in chess but cannot enter or jump over the central hole.
- **Pawns** move one square **clockwise** around the ring and capture on the two forward diagonals. A pawn promotes to a rook on reaching the opponent's home band.
- The game is won by **capturing the king**. It is a draw on threefold repetition or after 200 plies.

## Engine

| Component | File |
|---|---|
| Board, move generation, Zobrist hashing | `board.py` |
| Iterative-deepening **negamax with alpha-beta pruning** and principal variation search | `engine.py` |
| **Transposition table** keyed by Zobrist hash | `engine.py` |
| **Late move reduction** for late quiet moves, with re-search on fail-high | `engine.py` |
| Move ordering: TT move → MVV-LVA captures → killer moves | `engine.py` |
| Quiescence search over captures | `engine.py` |
| Positional evaluation: material + pawn advancement | `engine.py` |
| **Neural value network** (NumPy MLP, 393 → 64 → 1) | `value_net.py` |
| Self-play training with **bootstrapped TD / Q-learning-style targets** | `train.py` |

The learned evaluation is `material + 200 × net(position)`. The network is trained on self-play games. Each position is trained towards the network's value of the next position, and the final position towards the game result. ε-greedy random moves are mixed in for exploration.

## Usage

```bash
pip install -r requirements.txt

python play.py                        # play as White against the engine
python play.py --color black --time 3 # play as Black, 3 s per engine move
python play.py --selfplay             # watch engine vs engine

python train.py --games 200 --out weights.npz   # train the value network
python play.py --weights weights.npz            # play against the learned evaluation
```

Moves are typed as from-square + to-square (for example `c2b2`). Type `moves` to list the legal moves.

```
7  . . r b k . .
6  . . . p p . .
5  . . # # # . .
4  . . # # # . .
3  . . # # # . .
2  . . P P . . .
1  . . K B R . .
   a b c d e f g
```
