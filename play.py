"""Play Rollerball against the engine, or watch two engines play.

    python play.py                       # you are White vs the engine
    python play.py --color black         # you are Black
    python play.py --selfplay            # engine vs engine
    python play.py --weights weights.npz # use the learned evaluation

Moves are entered as from-square + to-square, e.g. "c2b2".
"""
import argparse

from board import WHITE, Board, move_str, parse_sq
from engine import Engine, material_eval
from value_net import ValueNet, hybrid_eval


def ask_move(b):
    legal = {move_str(m): m for m in b.moves()}
    while True:
        s = input("your move> ").strip().lower()
        if s in ("moves", "?"):
            print(" ".join(sorted(legal)))
        elif s in ("quit", "exit"):
            raise SystemExit
        elif s in legal:
            return legal[s]
        else:
            try:
                parse_sq(s[:2]), parse_sq(s[2:4])
                print("illegal move (type 'moves' to list legal moves)")
            except (ValueError, IndexError):
                print("format: e.g. c2b2   ('moves' lists legal moves)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--color", choices=["white", "black"], default="white")
    ap.add_argument("--selfplay", action="store_true")
    ap.add_argument("--time", type=float, default=2.0, help="seconds per engine move")
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--weights", default=None)
    args = ap.parse_args()

    evaluate = hybrid_eval(ValueNet.load(args.weights)) if args.weights else material_eval
    engine = Engine(evaluate, max_depth=args.depth, time_limit=args.time)
    human = None if args.selfplay else (WHITE if args.color == "white" else -WHITE)

    b = Board()
    print(b)
    while (res := b.result()) is None:
        side = "White" if b.side == WHITE else "Black"
        if b.side == human:
            m = ask_move(b)
        else:
            m = engine.search(b, verbose=True)
            print(f"{side} plays {move_str(m)}")
        b.make(m)
        print(b, "\n")
    print({1: "White wins", -1: "Black wins", 0: "Draw"}[res])


if __name__ == "__main__":
    main()
