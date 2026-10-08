"""Search engine: iterative-deepening negamax with alpha-beta pruning.

Features
- Principal variation search (null-window re-search)
- Zobrist-keyed transposition table
- Late move reduction for quiet moves searched late
- Move ordering: TT move, MVV-LVA captures, killer moves
- Quiescence search over captures
"""
import time
from collections import defaultdict

from board import (EMPTY, PAWN, SQUARES, VALUES, ORDER_VALUES, PAWN_STEPS, Board, move_str)

MATE = 100_000
INF = 10 ** 9
EXACT, LOWER, UPPER = 0, 1, 2


class SearchTimeout(Exception):
    pass


def material_eval(b: Board) -> int:
    """Handcrafted evaluation from the side to move's perspective."""
    score = 0
    for i in SQUARES:
        p = b.sq[i]
        if not p:
            continue
        color = 1 if p > 0 else -1
        t = abs(p)
        v = VALUES[t]
        if t == PAWN:
            v += 12 * (12 - min(PAWN_STEPS[color][i], 12))  # reward advanced pawns
        score += color * v
    return score * b.side


class Engine:
    def __init__(self, evaluate=material_eval, max_depth=8, time_limit=2.0):
        self.evaluate = evaluate
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.tt = {}
        self.killers = defaultdict(list)
        self.nodes = 0

    # ---------------------------------------------------------------- public
    def search(self, b: Board, verbose=False):
        self.deadline = time.time() + self.time_limit
        self.nodes = 0
        self.killers.clear()
        best, score = None, 0
        moves = b.moves()
        if len(moves) == 1:
            return moves[0]
        for depth in range(1, self.max_depth + 1):
            try:
                score = self.negamax(b, depth, -INF, INF, 0)
            except SearchTimeout:
                break
            entry = self.tt.get(b.hash)
            if entry and entry[3]:
                best = entry[3]
            if verbose:
                print(f"depth {depth:2d}  score {score:7d}  nodes {self.nodes:8d}  best {move_str(best)}")
            if abs(score) > MATE - 100:
                break
        self.last_score = score
        return best or moves[0]

    # ---------------------------------------------------------------- search
    def _tick(self):
        self.nodes += 1
        if self.nodes & 1023 == 0 and time.time() > self.deadline:
            raise SearchTimeout

    def order(self, b, moves, tt_move, ply):
        killers = self.killers[ply]

        def key(m):
            if m == tt_move:
                return 10_000
            victim = abs(b.sq[m[1]])
            if victim:
                return 1_000 + 10 * ORDER_VALUES[victim] - ORDER_VALUES[abs(b.sq[m[0]])]
            if m in killers:
                return 500
            return 0

        return sorted(moves, key=key, reverse=True)

    def negamax(self, b, depth, alpha, beta, ply):
        self._tick()
        if b.winner():
            return -MATE + ply  # our king has been captured
        if ply and b.repetitions():
            return 0
        if depth <= 0:
            return self.quiesce(b, alpha, beta, ply, 0)

        alpha_orig = alpha
        tt_move = None
        entry = self.tt.get(b.hash)
        if entry:
            e_depth, e_score, e_flag, tt_move = entry
            if ply and e_depth >= depth:
                if e_flag == EXACT:
                    return e_score
                if e_flag == LOWER:
                    alpha = max(alpha, e_score)
                else:
                    beta = min(beta, e_score)
                if alpha >= beta:
                    return e_score

        moves = b.moves()
        if not moves:
            return -MATE + ply
        moves = self.order(b, moves, tt_move, ply)

        best, best_move = -INF, None
        for i, m in enumerate(moves):
            capture = b.sq[m[1]] != EMPTY
            b.make(m)
            if i == 0:
                score = -self.negamax(b, depth - 1, -beta, -alpha, ply + 1)
            else:
                # Late move reduction: quiet moves late in the list are searched shallower first.
                r = 1 if (i >= 3 and depth >= 3 and not capture) else 0
                score = -self.negamax(b, depth - 1 - r, -alpha - 1, -alpha, ply + 1)
                if score > alpha and (r or score < beta):
                    score = -self.negamax(b, depth - 1, -beta, -alpha, ply + 1)
            b.undo()
            if score > best:
                best, best_move = score, m
            alpha = max(alpha, score)
            if alpha >= beta:
                if not capture and m not in self.killers[ply]:
                    self.killers[ply] = ([m] + self.killers[ply])[:2]
                break

        flag = UPPER if best <= alpha_orig else LOWER if best >= beta else EXACT
        self.tt[b.hash] = (depth, best, flag, best_move)
        return best

    def quiesce(self, b, alpha, beta, ply, qdepth):
        self._tick()
        if b.winner():
            return -MATE + ply
        stand = self.evaluate(b)
        if stand >= beta or qdepth >= 6:
            return stand
        alpha = max(alpha, stand)
        captures = [m for m in b.moves() if b.sq[m[1]] != EMPTY]
        for m in self.order(b, captures, None, ply):
            b.make(m)
            score = -self.quiesce(b, -beta, -alpha, ply + 1, qdepth + 1)
            b.undo()
            if score >= beta:
                return score
            alpha = max(alpha, score)
        return alpha
