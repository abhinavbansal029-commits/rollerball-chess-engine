"""Train the value network by self-play with temporal-difference (bootstrapped) targets.

Each position's target is the network's value of the next position (Q-learning style
bootstrapping); the final position is trained towards the actual game result.

    python train.py --games 200 --out weights.npz
"""
import argparse
import random

from board import Board
from engine import Engine
from value_net import ValueNet, features, hybrid_eval


def self_play_game(net, depth, epsilon, max_ply):
    b = Board()
    engine = Engine(hybrid_eval(net), max_depth=depth, time_limit=0.5)
    states = [features(b)]
    while b.result(max_ply) is None:
        moves = b.moves()
        m = random.choice(moves) if random.random() < epsilon else engine.search(b)
        b.make(m)
        states.append(features(b))
    return states, b.result(max_ply)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--epsilon", type=float, default=0.1, help="exploration rate")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--max-ply", type=int, default=150)
    ap.add_argument("--out", default="weights.npz")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    net = ValueNet.load(args.resume) if args.resume else ValueNet()
    results = {1: 0, -1: 0, 0: 0}
    for g in range(1, args.games + 1):
        states, result = self_play_game(net, args.depth, args.epsilon, args.max_ply)
        results[result] += 1
        loss = 0.0
        for t in reversed(range(len(states))):
            target = result if t == len(states) - 1 else net.forward(states[t + 1])[0]
            loss += net.update(states[t], target, args.lr)
        if g % 10 == 0 or g == args.games:
            print(f"game {g:4d}  plies {len(states) - 1:3d}  loss {loss / len(states):.4f}  "
                  f"W/B/D {results[1]}/{results[-1]}/{results[0]}")
            net.save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
