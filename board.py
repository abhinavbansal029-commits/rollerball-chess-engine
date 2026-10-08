"""Rollerball board representation, move generation and Zobrist hashing.

The board is 7x7 with the central 3x3 block removed, leaving a 40-square ring.
Pieces are stored as signed ints: positive = White, negative = Black.
"""
import random

N = 7
EMPTY, PAWN, BISHOP, ROOK, KING = 0, 1, 2, 3, 4
WHITE, BLACK = 1, -1
SYMBOLS = {PAWN: "P", BISHOP: "B", ROOK: "R", KING: "K"}
VALUES = {PAWN: 100, BISHOP: 320, ROOK: 500, KING: 20000}
ORDER_VALUES = {EMPTY: 0, PAWN: 1, BISHOP: 3, ROOK: 5, KING: 50}

ORTHO = [(0, 1), (1, 0), (0, -1), (-1, 0)]
DIAG = [(1, 1), (1, -1), (-1, 1), (-1, -1)]


def idx(r, c):
    return r * N + c


def in_hole(r, c):
    return 2 <= r <= 4 and 2 <= c <= 4


def on_board(r, c):
    return 0 <= r < N and 0 <= c < N and not in_hole(r, c)


SQUARES = [idx(r, c) for r in range(N) for c in range(N) if on_board(r, c)]


def clockwise(r, c):
    """Direction a pawn moves from (r, c): both sides travel clockwise around the ring."""
    if r <= 1 and c < 5:
        return (0, 1)
    if c >= 5 and r < 5:
        return (1, 0)
    if r >= 5 and c > 1:
        return (0, -1)
    return (-1, 0)


def promotes(color, r, c):
    """A pawn promotes to a rook on reaching the opponent's home band."""
    if color == WHITE:
        return r <= 1 and c >= 2
    return r >= 5 and c <= 4


def _pawn_steps():
    """Steps each square is away from promotion, following the clockwise path."""
    steps = {WHITE: [0] * (N * N), BLACK: [0] * (N * N)}
    for color in (WHITE, BLACK):
        for sq in SQUARES:
            r, c = divmod(sq, N)
            n = 0
            while not promotes(color, r, c) and n < 40:
                dr, dc = clockwise(r, c)
                r, c = r + dr, c + dc
                n += 1
            steps[color][sq] = n
    return steps


PAWN_STEPS = _pawn_steps()

_rng = random.Random(2026)
ZOBRIST = [[_rng.getrandbits(64) for _ in range(9)] for _ in range(N * N)]
Z_SIDE = _rng.getrandbits(64)

FILES = "abcdefg"


def sq_name(i):
    r, c = divmod(i, N)
    return f"{FILES[c]}{N - r}"


def parse_sq(s):
    c = FILES.index(s[0])
    r = N - int(s[1])
    return idx(r, c)


def move_str(m):
    return sq_name(m[0]) + sq_name(m[1])


class Board:
    def __init__(self):
        self.sq = [EMPTY] * (N * N)
        setup = {(6, 2): KING, (6, 3): BISHOP, (6, 4): ROOK, (5, 2): PAWN, (5, 3): PAWN}
        for (r, c), p in setup.items():
            self.sq[idx(r, c)] = WHITE * p
            self.sq[idx(6 - r, 6 - c)] = BLACK * p  # 180-degree symmetric setup
        self.side = WHITE
        self.ply = 0
        self.undo_stack = []
        self.hashes = []
        self.hash = self._full_hash()

    def _full_hash(self):
        h = 0
        for i in SQUARES:
            if self.sq[i]:
                h ^= ZOBRIST[i][self.sq[i] + 4]
        return h ^ (Z_SIDE if self.side == BLACK else 0)

    def copy(self):
        b = Board.__new__(Board)
        b.sq = self.sq[:]
        b.side, b.ply, b.hash = self.side, self.ply, self.hash
        b.undo_stack, b.hashes = [], self.hashes[:]
        return b

    # ---------------------------------------------------------------- moves
    def moves(self):
        out = []
        s = self.side
        sq = self.sq
        for i in SQUARES:
            p = sq[i]
            if p * s <= 0:
                continue
            t = abs(p)
            r, c = divmod(i, N)
            if t == PAWN:
                dr, dc = clockwise(r, c)
                nr, nc = r + dr, c + dc
                if on_board(nr, nc) and sq[idx(nr, nc)] == EMPTY:
                    out.append((i, idx(nr, nc)))
                for pr, pc in ((dc, dr), (-dc, -dr)):  # forward diagonals
                    nr, nc = r + dr + pr, c + dc + pc
                    if on_board(nr, nc) and sq[idx(nr, nc)] * s < 0:
                        out.append((i, idx(nr, nc)))
            elif t == KING:
                for dr, dc in ORTHO + DIAG:
                    nr, nc = r + dr, c + dc
                    if on_board(nr, nc) and sq[idx(nr, nc)] * s <= 0:
                        out.append((i, idx(nr, nc)))
            else:
                for dr, dc in (ORTHO if t == ROOK else DIAG):
                    nr, nc = r + dr, c + dc
                    while on_board(nr, nc):
                        j = idx(nr, nc)
                        if sq[j] * s > 0:
                            break
                        out.append((i, j))
                        if sq[j] != EMPTY:
                            break
                        nr, nc = nr + dr, nc + dc
        return out

    def make(self, m):
        frm, to = m
        piece, captured = self.sq[frm], self.sq[to]
        self.undo_stack.append((frm, to, piece, captured, self.hash))
        self.hashes.append(self.hash)
        h = self.hash ^ ZOBRIST[frm][piece + 4]
        if captured:
            h ^= ZOBRIST[to][captured + 4]
        r, c = divmod(to, N)
        new_piece = piece
        if abs(piece) == PAWN and promotes(self.side, r, c):
            new_piece = self.side * ROOK
        self.sq[frm], self.sq[to] = EMPTY, new_piece
        h ^= ZOBRIST[to][new_piece + 4] ^ Z_SIDE
        self.hash = h
        self.side = -self.side
        self.ply += 1

    def undo(self):
        frm, to, piece, captured, h = self.undo_stack.pop()
        self.hashes.pop()
        self.sq[frm], self.sq[to] = piece, captured
        self.hash = h
        self.side = -self.side
        self.ply -= 1

    # ------------------------------------------------------------ game state
    def has_king(self, color):
        return any(self.sq[i] == color * KING for i in SQUARES)

    def winner(self):
        """WHITE/BLACK if one king has been captured, else 0."""
        w, b = self.has_king(WHITE), self.has_king(BLACK)
        if w and b:
            return 0
        return WHITE if w else BLACK

    def repetitions(self):
        return self.hashes.count(self.hash)

    def result(self, max_ply=200):
        """None while the game is running; otherwise +1 / -1 / 0 from White's view."""
        w = self.winner()
        if w:
            return w
        if not self.moves():
            return -self.side  # no legal move loses
        if self.ply >= max_ply or self.repetitions() >= 2:
            return 0
        return None

    def __str__(self):
        rows = []
        for r in range(N):
            line = f"{N - r} "
            for c in range(N):
                if in_hole(r, c):
                    line += " #"
                    continue
                p = self.sq[idx(r, c)]
                ch = "." if p == 0 else SYMBOLS[abs(p)]
                line += " " + (ch.lower() if p < 0 else ch)
            rows.append(line)
        rows.append("   " + " ".join(FILES))
        return "\n".join(rows)
