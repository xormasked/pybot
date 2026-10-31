import argparse
import ctypes
import os
import queue
import threading
import sys
import time
from pathlib import Path

import chess
import chess.engine
import numpy as np

STOCKFISH_PATH = r"PATH TO YOUR STOCKFISH"
ENGINE_DEPTH = 16
ENGINE_TIME = 0.35
ENGINE_THREADS = 4
ENGINE_HASH_MB = 128
READ_SAMPLES = 2
POLL_INTERVAL = 0.25

CALIBRATION = Path(__file__).resolve().parent / "board_view_calibration.npz" <- make ur own...
FORMAT_VERSION = 1

START = {7: "rnbqkbnr", 6: "pppppppp", 1: "PPPPPPPP", 0: "RNBQKBNR"}
QUANT = 16
FEAT = 32
EMPTY = 0.04
MAX_DIST = 0.22
LUMA = np.array([0.299, 0.587, 0.114], np.float32)
START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


class BoardError(Exception):
    pass


def make_dpi_aware():
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def screenshot(image_path=None):
    if image_path:
        from PIL import Image
        with Image.open(image_path) as im:
            return np.array(im.convert("RGB"))

    import mss
    with mss.MSS() as sct:
        shot = sct.grab(sct.monitors[1])
        return np.array(shot)[:, :, 2::-1].copy()


def _frame_means(gray, p):
    ii = np.pad(gray, ((1, 0), (1, 0))).cumsum(0).cumsum(1)

    def box_sum(off, n):
        a = ii[off + n:, off + n:]
        h, w = a.shape
        return (
            a
            - ii[off:off + h, off + n:off + n + w]
            - ii[off + n:off + n + h, off:off + w]
            + ii[off:off + h, off:off + w]
        )

    b = max(1, p // 7)
    outer = box_sum(0, p)
    inner = box_sum(b, p - 2 * b)[:outer.shape[0] - 2 * b,
                                      :outer.shape[1] - 2 * b]
    h, w = inner.shape
    return (outer[:h, :w] - inner) / (p * p - (p - 2 * b) ** 2)


def _checker(means, p):
    offs = [round(i * p) for i in range(8)]
    h = means.shape[0] - offs[7]
    w = means.shape[1] - offs[7]

    if h <= 0 or w <= 0:
        return None, None

    sq = np.stack([
        means[a:a + h, b:b + w]
        for a in offs for b in offs
    ]).reshape(8, 8, h, w)

    sign = np.where(
        np.add.outer(np.arange(8), np.arange(8)) % 2 == 0,
        1.0, -1.0
    )

    s = sq * sign[:, :, None, None]

    pairs = np.concatenate([
        (s[:, :-1] + s[:, 1:]).reshape(56, h, w),
        (s[:-1] + s[1:]).reshape(56, h, w)
    ])

    contrast = s.mean((0, 1))
    agree = np.where(
        contrast > 0,
        (pairs > 0).sum(0),
        (pairs < 0).sum(0)
    )

    return np.abs(contrast), agree


def find_board(rgb):
    pad = 16
    gray = rgb.astype(np.float32) @ LUMA
    gray = np.pad(gray, pad, constant_values=float(np.median(gray)))

    k = 4
    h = gray.shape[0] // k * k
    w = gray.shape[1] // k * k
    small = gray[:h, :w].reshape(h // k, k, w // k, k).mean((1, 3))
    best = None

    for p in np.arange(4, min(small.shape) / 8, 0.5):
        contrast, agree = _checker(_frame_means(small, int(p)), p)

        if contrast is None:
            continue

        score = contrast * (agree >= 100)
        i = score.argmax()

        if score.flat[i] > 5 and (best is None or score.flat[i] > best[0]):
            y, x = np.unravel_index(i, score.shape)
            best = (score.flat[i], x * k, y * k, p * k)

    if best is None:
        return None

    _, x0, y0, p0 = best
    m = 2 * k
    cx = max(0, x0 - m)
    cy = max(0, y0 - m)

    area = gray[
        cy:y0 + int(8 * (p0 + k)) + m,
        cx:x0 + int(8 * (p0 + k)) + m
    ]

    top = None

    for p in np.arange(p0 - k, p0 + k + 0.01, 0.125):
        contrast, agree = _checker(_frame_means(area, int(p)), p)

        if contrast is None:
            continue

        sub = contrast[:2 * m + 1, :2 * m + 1] * (
            agree[:2 * m + 1, :2 * m + 1] >= 100
        )

        i = sub.argmax()

        if sub.flat[i] > 0 and (top is None or sub.flat[i] > top[0]):
            y, x = np.unravel_index(i, sub.shape)
            top = (sub.flat[i], cx + x, cy + y, p)

    if top is None:
        return None

    x, y, size = _polish(gray, top[1], top[2], top[3])

    if x is None:
        return None

    x -= pad
    y -= pad
    x = max(0, x)
    y = max(0, y)
    size = min(size, rgb.shape[1] - x, rgb.shape[0] - y)

    return (int(x), int(y), int(size)) if size >= 80 else None


def _score_at(ii, x, y, p, b):
    pos = [round(i * p) for i in range(9)]
    n = int(p)

    if (
        x < 0 or y < 0
        or y + pos[7] + n >= ii.shape[0]
        or x + pos[7] + n >= ii.shape[1]
        or n - 2 * b < 1
    ):
        return 0.0

    def box_sums(off, size):
        ys = np.array([y + pos[i] + off for i in range(8)])[:, None]
        xs = np.array([x + pos[j] + off for j in range(8)])[None, :]

        return (
            ii[ys + size, xs + size] - ii[ys, xs + size]
            - ii[ys + size, xs] + ii[ys, xs]
        )

    means = (box_sums(0, n) - box_sums(b, n - 2 * b)) / (
        n * n - (n - 2 * b) ** 2
    )

    sign = np.where(
        np.add.outer(np.arange(8), np.arange(8)) % 2 == 0,
        1.0, -1.0
    )

    s = means * sign
    contrast = s.mean()

    pairs = np.concatenate([
        (s[:, :-1] + s[:, 1:]).ravel(),
        (s[:-1] + s[1:]).ravel()
    ])

    agree = (pairs > 0).sum() if contrast > 0 else (pairs < 0).sum()

    return abs(contrast) if agree >= 100 else 0.0


def _polish(gray, x, y, p):
    ii = np.pad(gray, ((1, 0), (1, 0))).cumsum(0).cumsum(1)

    best = max((
        _score_at(
            ii,
            round(x + dx * p),
            round(y + dy * p),
            p,
            max(1, int(p) // 7)
        ),
        round(x + dx * p),
        round(y + dy * p)
    ) for dx in (-1, 0, 1) for dy in (-1, 0, 1))

    if best[0] == 0:
        return None, None, None

    _, x, y = best
    b = max(1, round(p / 25))

    for step, span in ((2, 8), (1, 2)):
        cands = [
            (
                _score_at(ii, x + dx, y + dy, q, b),
                x + dx,
                y + dy,
                q
            )
            for dx in range(-span, span + 1, step)
            for dy in range(-span, span + 1, step)
            for q in np.arange(p - 1.5, p + 1.51, 0.125)
        ]

        score, x, y, p = max(cands)

        if score == 0:
            return None, None, None

    return x, y, round(p * 8)


def board_image(rgb, region):
    x, y, size = region
    return rgb[y:y + size, x:x + size]


def squares(board):
    s = board.shape[0] / 8

    for row in range(8):
        for col in range(8):
            yield (
                row,
                col,
                board[
                    round(row * s):round((row + 1) * s),
                    round(col * s):round((col + 1) * s)
                ]
            )


def square_rank_file(row, col, white_bottom):
    return (7 - row, col) if white_bottom else (row, 7 - col)


def _bins(pixels):
    q = pixels.reshape(-1, 3).astype(np.int32) // QUANT
    n = 256 // QUANT
    return (q[:, 0] * n + q[:, 1]) * n + q[:, 2]


def _colour_table(pixels, min_count):
    n = 256 // QUANT
    seen = (
        np.bincount(_bins(pixels), minlength=n ** 3) >= min_count
    ).reshape(n, n, n)

    out = seen.copy()
    out[1:, :, :] |= seen[:-1, :, :]
    out[:-1, :, :] |= seen[1:, :, :]
    out[:, 1:, :] |= seen[:, :-1, :]
    out[:, :-1, :] |= seen[:, 1:, :]
    out[:, :, 1:] |= seen[:, :, :-1]
    out[:, :, :-1] |= seen[:, :, 1:]

    return out.ravel()


def _ring(sq):
    r = max(2, sq.shape[0] // 20)

    return np.concatenate([
        sq[:r].reshape(-1, 3),
        sq[-r:].reshape(-1, 3),
        sq[:, :r].reshape(-1, 3),
        sq[:, -r:].reshape(-1, 3)
    ])


def piece_mask(sq, background):
    h, w, _ = sq.shape
    table = background | _colour_table(_ring(sq), 3)
    mask = ~table[_bins(sq)].reshape(h, w)

    m = max(3, h // 12)
    mask[:m] = False
    mask[-m:] = False
    mask[:, :m] = False
    mask[:, -m:] = False

    return mask


def _resize(a):
    h, w = a.shape
    ys = np.linspace(0, h, FEAT + 1).astype(int)
    xs = np.linspace(0, w, FEAT + 1).astype(int)

    pooled = np.add.reduceat(
        np.add.reduceat(a, ys[:-1], 0),
        xs[:-1],
        1
    )

    return pooled / np.outer(np.diff(ys), np.diff(xs))


def features(sq, mask):
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    cy = int(round(ys.mean()))
    cx = int(round(xs.mean()))

    lum = sq.astype(np.float32) @ LUMA
    vals = lum[mask]
    detail = np.where(
        mask,
        (lum - vals.mean()) / (vals.std() + 1e-6),
        0
    )

    def centred(a):
        p = np.pad(a, ((h, h), (w, w)))
        top = cy + h - h // 2
        left = cx + w - w // 2
        return _resize(p[top:top + h, left:left + w])

    return (
        centred(mask.astype(np.float32)),
        centred(detail.astype(np.float32))
    )


def distance(shape_a, detail_a, shape_b, detail_b):
    return (
        np.abs(shape_a - shape_b).mean(axis=(-1, -2))
        + 0.15 * np.abs(detail_a - detail_b).mean(axis=(-1, -2))
    )


def detect_bottom_side(board, background):
    top_colours = []
    bottom_colours = []

    for row, col, sq in squares(board):
        if row not in (0, 1, 6, 7):
            continue

        mask = piece_mask(sq, background)

        if mask.mean() < EMPTY:
            continue

        colour = np.median(sq[mask], axis=0)

        if row in (6, 7):
            bottom_colours.append(colour)
        else:
            top_colours.append(colour)

    if len(top_colours) < 8 or len(bottom_colours) < 8:
        raise BoardError(
            "Couldn't identify the piece colours at the top and bottom "
            "of the board. Make sure the game is in its starting position."
        )

    top_colour = np.median(top_colours, axis=0)
    bottom_colour = np.median(bottom_colours, axis=0)

    top_brightness = float(top_colour @ LUMA)
    bottom_brightness = float(bottom_colour @ LUMA)

    if abs(bottom_brightness - top_brightness) < 25:
        raise BoardError(
            "Couldn't reliably distinguish the white and black pieces."
        )

    return bottom_brightness > top_brightness

def calibrate(rgb):
    region = find_board(rgb)

    if region is None:
        raise BoardError(
            "Couldn't find a chess board. Make sure the whole board is visible."
        )

    board = board_image(rgb, region)

    background = _colour_table(
        np.concatenate([_ring(sq) for _, _, sq in squares(board)]),
        5
    )

    white_bottom = detect_bottom_side(board, background)

    labels = []
    shapes = []
    details = []
    colours = []
    problems = []

    for row, col, sq in squares(board):
        rank, file = square_rank_file(row, col, white_bottom)
        mask = piece_mask(sq, background)
        name = "abcdefgh"[file] + str(rank + 1)
        has_piece = mask.mean() >= EMPTY

        if rank not in START:
            if has_piece:
                problems.append(f"{name} should be empty")
            continue

        if not has_piece:
            problems.append(f"{name} should have a piece")
            continue

        shape, detail = features(sq, mask)

        labels.append(START[rank][file])
        shapes.append(shape)
        details.append(detail)
        colours.append(np.median(sq[mask], axis=0))

    if problems:
        raise BoardError(
            "Calibration needs the STARTING position, but: "
            + ", ".join(problems[:6])
            + (" ..." if len(problems) > 6 else "")
            + "\nStart a new game and calibrate before anyone moves."
        )

    labels = np.array(labels)
    colours = np.array(colours, np.float32)

    white = colours[np.char.isupper(labels)].mean(0)
    black = colours[np.char.islower(labels)].mean(0)

    if np.linalg.norm(white - black) < 40:
        raise BoardError(
            "Calibration failed: white and black pieces look too similar."
        )

    cal = dict(
        version=np.int32(FORMAT_VERSION),
        background=background,
        labels=labels,
        shapes=np.array(shapes),
        details=np.array(details),
        white=white,
        black=black,
        white_bottom=np.bool_(white_bottom)
    )

    grid, unsure = read_grid(board, cal, white_bottom)
    expected = START_FEN

    if unsure or to_fen(grid) != expected:
        raise BoardError(
            "Calibration failed its self-check (read back: "
            + to_fen(grid) + ")."
        )

    return cal, region


def save_calibration(cal):
    tmp = CALIBRATION.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **cal)
    os.replace(tmp, CALIBRATION)


def load_calibration():
    if not CALIBRATION.exists():
        raise BoardError(
            "Not calibrated yet. Open a game at the starting position and run:\n"
            "  python board_view.py calibrate"
        )

    try:
        with np.load(CALIBRATION) as f:
            cal = {k: f[k] for k in f.files}

        if int(cal["version"]) != FORMAT_VERSION:
            raise ValueError("old format")

    except Exception as exc:
        raise BoardError(
            "The calibration file is damaged or outdated. "
            "Run: python board_view.py calibrate"
        ) from exc

    return cal


def classify(sq, cal):
    mask = piece_mask(sq, cal["background"])

    if mask.mean() < EMPTY:
        return None

    colour = np.median(sq[mask], axis=0)

    is_white = (
        np.linalg.norm(colour - cal["white"])
        < np.linalg.norm(colour - cal["black"])
    )

    same = np.char.isupper(cal["labels"]) == is_white
    shape, detail = features(sq, mask)

    d = distance(
        shape,
        detail,
        cal["shapes"][same],
        cal["details"][same]
    )

    best = int(d.argmin())

    return "?" if d[best] > MAX_DIST else str(cal["labels"][same][best])


def read_grid(board, cal, white_bottom):
    grid = [[None] * 8 for _ in range(8)]
    unsure = []

    for row, col, sq in squares(board):
        rank, file = square_rank_file(row, col, white_bottom)
        grid[rank][file] = classify(sq, cal)

        if grid[rank][file] == "?":
            unsure.append("abcdefgh"[file] + str(rank + 1))

    return grid, unsure


def _captured_icons_brightness(rgb, region, below):
    x, y, size = region
    gap = int(0.012 * size)
    h = int(0.12 * size)

    band = (
        rgb[y + size + gap:y + size + h]
        if below else rgb[max(0, y - h):max(0, y - gap)]
    )

    band = band[:, x:x + int(0.75 * size)].astype(np.float32)

    if band.shape[0] < 5 or band.shape[1] < 5:
        return None

    fg = np.abs(
        band - np.median(band.reshape(-1, 3), 0)
    ).sum(2) > 60

    def runs(flags, gap_ok=0):
        out = []

        for i in np.flatnonzero(flags):
            if out and i - out[-1][1] <= gap_ok + 1:
                out[-1][1] = i + 1
            else:
                out.append([i, i + 1])

        return out

    cols = runs(fg.any(0))

    if cols:
        a, b = cols[0]

        if (
            b - a >= 0.03 * size
            and fg[:, a:b].any(1).sum() >= 0.04 * size
        ):
            fg[:, :b] = False

    lines = runs(fg.any(1), gap_ok=1)

    if len(lines) < 2:
        return None

    a, b = lines[-1]

    return float(np.median(band[a:b][fg[a:b]] @ LUMA))


def white_at_bottom(rgb, region, cal):
    return bool(cal["white_bottom"])


def read_screen(cal, region=None, image_path=None, samples=3):
    last = None
    agreeing = 0

    for _ in range(samples + 12):
        rgb = screenshot(image_path)

        if (
            region is None
            or region[1] + region[2] > rgb.shape[0]
            or region[0] + region[2] > rgb.shape[1]
        ):
            region = find_board(rgb)

            if region is None:
                raise BoardError(
                    "Couldn't find a chess board. "
                    "Make sure the whole board is visible."
                )

        wb = white_at_bottom(rgb, region, cal)
        grid, unsure = read_grid(board_image(rgb, region), cal, wb)
        result = (grid, unsure, wb)

        if image_path:
            return grid, unsure, wb, region

        agreeing = agreeing + 1 if result == last else 1
        last = result

        if agreeing >= samples:
            return grid, unsure, wb, region

        time.sleep(0.15)

    raise BoardError(
        "The board kept changing - is a piece being dragged or animating?"
    )


def problems_with(grid):
    pieces = [
        p for row in grid for p in row
        if p and p != "?"
    ]

    issues = []

    for piece, side in (("K", "white"), ("k", "black")):
        if pieces.count(piece) != 1:
            issues.append(f"{pieces.count(piece)} {side} kings")

    for piece, side in (("P", "white"), ("p", "black")):
        if pieces.count(piece) > 8:
            issues.append(f"more than 8 {side} pawns")

    if any(grid[r][f] in ("P", "p") for r in (0, 7) for f in range(8)):
        issues.append("a pawn on the first or last rank")

    return issues


def to_fen(grid):
    rows = []

    for rank in range(7, -1, -1):
        row = ""
        empty = 0

        for file in range(8):
            piece = grid[rank][file]

            if piece is None:
                empty += 1
            else:
                row += (str(empty) if empty else "") + piece
                empty = 0

        rows.append(row + (str(empty) if empty else ""))

    return "/".join(rows)


def report(grid, unsure, white_bottom):
    lines = [
        f"{rank + 1} " + " ".join(
            grid[rank][f] or "." for f in range(8)
        )
        for rank in range(7, -1, -1)
    ]

    lines.append("  a b c d e f g h")
    lines.append("FEN: " + to_fen(grid))

    pieces = [
        p for row in grid for p in row
        if p and p != "?"
    ]

    values = {
        "p": 1, "n": 3, "b": 3,
        "r": 5, "q": 9, "k": 0
    }

    names = {
        "q": "queen", "r": "rook",
        "b": "bishop", "n": "knight",
        "p": "pawn"
    }

    for taker, case in (("White", str.lower), ("Black", str.upper)):
        gone = []

        for piece, start in zip("qrbnp", (1, 2, 2, 2, 8)):
            n = start - pieces.count(case(piece))

            if n > 0:
                gone.append(
                    f"{n} {names[piece]}{'s' if n > 1 else ''}"
                )

        lines.append(
            f"{taker} has captured: {', '.join(gone) or 'nothing'}"
        )

    balance = sum(
        values[p.lower()] * (1 if p.isupper() else -1)
        for p in pieces
    )

    lines.append(
        "Material: " + (
            "equal" if balance == 0
            else f"{'white' if balance > 0 else 'black'} +{abs(balance)}"
        )
    )

    lines.append(
        f"You are playing {'white' if white_bottom else 'black'}."
    )

    if unsure:
        lines.append(
            "Not sure about: " + ", ".join(unsure) + " (shown as ?)"
        )

    issues = problems_with(grid)

    if issues:
        lines.append(
            "Warning, this looks like a misread: " + "; ".join(issues)
        )

    return "\n".join(lines)


class ChessOverlay:

    def __init__(self):
        self.commands = queue.Queue()
        self.ready = threading.Event()
        self.error = None
        self.region = None
        self.hwnd = None
        self.user32 = ctypes.windll.user32

        self.thread = threading.Thread(
            target=self._run,
            daemon=True
        )

        self.thread.start()

        if not self.ready.wait(5):
            raise RuntimeError("Move overlay failed to initialize.")

        if self.error:
            raise RuntimeError(
                f"Move overlay failed: {self.error}"
            )

    def _run(self):
        try:
            import tkinter as tk

            self.root = tk.Tk()
            self.root.overrideredirect(True)
            self.root.attributes("-topmost", True)
            self.root.configure(bg="#ff00ff")
            self.root.wm_attributes("-transparentcolor", "#ff00ff")
            self.root.geometry("1x1+0+0")

            self.canvas = tk.Canvas(
                self.root,
                bg="#ff00ff",
                highlightthickness=0,
                borderwidth=0
            )

            self.canvas.pack(fill="both", expand=True)
            self.root.update_idletasks()

            self.hwnd = self.user32.GetParent(
                self.root.winfo_id()
            )

            if not self.hwnd:
                self.hwnd = self.root.winfo_id()

            get_style = self.user32.GetWindowLongW
            set_style = self.user32.SetWindowLongW

            style = get_style(self.hwnd, -20)
            style |= 0x00080000
            style |= 0x00000020
            style |= 0x00000080
            style |= 0x08000000

            set_style(self.hwnd, -20, style)

            self.ready.set()
            self.root.after(20, self._process)
            self.root.mainloop()

        except Exception as exc:
            self.error = exc
            self.ready.set()

    def _position_window(self, x, y, size):
        self.root.geometry(f"{size}x{size}+{x}+{y}")
        self.canvas.config(width=size, height=size)
        self.root.update_idletasks()

        self.user32.SetWindowPos(
            self.hwnd,
            -1,
            x, y, size, size,
            0x0010 | 0x0040
        )

    def _process(self):
        import tkinter as tk

        try:
            while True:
                command = self.commands.get_nowait()
                kind = command[0]

                if kind == "close":
                    self.root.destroy()
                    return

                if kind == "region":
                    _, x, y, size = command
                    self._position_window(x, y, size)
                    self.canvas.delete("all")

                elif kind == "clear":
                    self.canvas.delete("all")

                elif kind == "draw":
                    points, width = command[1], command[2]
                    self.canvas.delete("all")

                    for i in range(len(points) - 1):
                        x1, y1 = points[i]
                        x2, y2 = points[i + 1]

                        self.canvas.create_line(
                            x1, y1, x2, y2,
                            fill="#ff3030",
                            width=width,
                            capstyle=tk.ROUND,
                            joinstyle=tk.ROUND,
                            arrow=tk.LAST if i == len(points) - 2 else tk.NONE,
                            arrowshape=(18, 22, 9)
                        )

                    x, y = points[0]

                    self.canvas.create_oval(
                        x - width, y - width,
                        x + width, y + width,
                        fill="#ff3030",
                        outline=""
                    )

                    self.user32.SetWindowPos(
                        self.hwnd,
                        -1,
                        0, 0, 0, 0,
                        0x0001 | 0x0002 | 0x0010 | 0x0040
                    )

        except queue.Empty:
            pass

        self.root.after(20, self._process)

    def square_center(self, square, region, white_bottom):
        file_index = chess.square_file(square)
        rank_index = chess.square_rank(square)

        if white_bottom:
            col = file_index
            row = 7 - rank_index
        else:
            col = 7 - file_index
            row = rank_index

        square_size = region[2] / 8.0

        return (
            int((col + 0.5) * square_size),
            int((row + 0.5) * square_size)
        )

    def draw_move(self, move, region, white_bottom):
        if region != self.region:
            self.region = region
            self.commands.put(("region", *region))

        start = self.square_center(
            move.from_square,
            region,
            white_bottom
        )

        end = self.square_center(
            move.to_square,
            region,
            white_bottom
        )

        dx = abs(
            chess.square_file(move.to_square)
            - chess.square_file(move.from_square)
        )

        dy = abs(
            chess.square_rank(move.to_square)
            - chess.square_rank(move.from_square)
        )

        points = [start]

        if (dx, dy) == (1, 2):
            bend = chess.square(
                chess.square_file(move.from_square),
                chess.square_rank(move.to_square)
            )

            points.append(
                self.square_center(bend, region, white_bottom)
            )

        elif (dx, dy) == (2, 1):
            bend = chess.square(
                chess.square_file(move.to_square),
                chess.square_rank(move.from_square)
            )

            points.append(
                self.square_center(bend, region, white_bottom)
            )

        points.append(end)

        width = max(4, int((region[2] / 8.0) * 0.055))
        self.commands.put(("draw", points, width))

    def clear(self):
        self.commands.put(("clear",))

    def close(self):
        self.commands.put(("close",))


def sync_position(board, detected_fen, max_plies=3):
    if board.board_fen() == detected_fen:
        return None

    nodes = 0
    node_limit = 20000

    def search(remaining):
        nonlocal nodes

        if board.board_fen() == detected_fen:
            return []

        if remaining == 0 or nodes >= node_limit:
            return None

        for move in board.legal_moves:
            nodes += 1
            board.push(move)

            result = search(remaining - 1)

            if result is not None:
                return [move] + result

            board.pop()

            if nodes >= node_limit:
                break

        return None

    for depth in range(1, max_plies + 1):
        result = search(depth)

        if result is not None:
            return result

    return False


def analyze_position(engine, board):
    if board.is_game_over():
        return None, None

    info = engine.analyse(
        board,
        chess.engine.Limit(
            time=ENGINE_TIME,
            depth=ENGINE_DEPTH
        )
    )

    if not info.get("pv"):
        return None, None

    move = info["pv"][0]
    score = info["score"].pov(board.turn)

    return move, score


def format_score(score):
    if score.is_mate():
        mate = score.mate()
        return f"Mate in {mate:+d}"

    return f"{score.score() / 100:+.2f}"


def start_engine():
    if not Path(STOCKFISH_PATH).is_file():
        raise BoardError(
            "Stockfish executable not found. Edit STOCKFISH_PATH near the top "
            "of board_view.py and point it to your stockfish.exe.\n"
            f"Current path: {STOCKFISH_PATH}"
        )

    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

        options = {}

        if "Threads" in engine.options:
            options["Threads"] = ENGINE_THREADS

        if "Hash" in engine.options:
            options["Hash"] = ENGINE_HASH_MB

        if options:
            engine.configure(options)

        return engine

    except Exception as exc:
        raise BoardError(
            f"Could not start Stockfish: {exc}"
        ) from exc


def detect_starting_position(cal, region):
    try:
        grid, unsure, white_bottom, found_region = read_screen(
            cal,
            region,
            samples=READ_SAMPLES
        )

    except BoardError:
        return None

    if unsure or problems_with(grid) or to_fen(grid) != START_FEN:
        return None

    return white_bottom, found_region


def run_live(cal):
    side = ""

    while side not in ("white", "black"):
        side = input("Which side are you playing as? (white/black): ").strip().lower()

        if side not in ("white", "black"):
            print("Please enter white or black.")

    playing_as = chess.WHITE if side == "white" else chess.BLACK

    turn = ""

    while turn not in ("yes", "no"):
        turn = input("Is it your turn right now? (yes/no): ").strip().lower()

        if turn not in ("yes", "no"):
            print("Please enter yes or no.")

    current_turn = playing_as if turn == "yes" else not playing_as

    cal["white_bottom"] = np.bool_(playing_as == chess.WHITE)

    engine = start_engine()
    overlay = None

    try:
        overlay = ChessOverlay()

        board = None

        print("Stockfish initialized.")
        print(f"Your side: {side.capitalize()}")
        print(f"Current turn: {'Your turn' if turn == 'yes' else 'Opponent'}")
        print("Reading the current board position...")
        print("Press Ctrl+C to stop.\n")

        region = None
        last_position = None
        last_error = None

        while True:
            try:
                grid, unsure, white_bottom, region = read_screen(
                    cal,
                    region,
                    samples=READ_SAMPLES
                )

                detected_fen = to_fen(grid)
                issues = problems_with(grid)
                invalid = bool(unsure or issues)

                if invalid:
                    overlay.clear()
                    time.sleep(POLL_INTERVAL)
                    continue

                if board is None:
                    turn_char = "w" if current_turn == chess.WHITE else "b"

                    board = chess.Board(
                        f"{detected_fen} {turn_char} - - 0 1"
                    )

                    if not board.is_valid():
                        overlay.clear()
                        print(
                            "The detected position is not valid. "
                            "Waiting for a clearer board..."
                        )
                        board = None
                        time.sleep(POLL_INTERVAL)
                        continue

                    last_position = None

                    print("\nCurrent board position loaded successfully.")

                else:
                    applied = sync_position(board, detected_fen)

                    if applied is False:
                        overlay.clear()

                        message = (
                            "Board position doesn't match the tracked game. "
                            "Waiting for the next recognizable position..."
                        )

                        if message != last_error:
                            print(message)

                        last_error = message
                        time.sleep(POLL_INTERVAL)
                        continue

                    if applied:
                        print(
                            "Move(s) detected: "
                            + " ".join(m.uci() for m in applied)
                        )

                position_key = board.fen()

                if position_key != last_position:
                    last_position = position_key

                    if board.turn != playing_as:
                        overlay.clear()
                        print("Opponent to move — waiting for your turn.")

                    else:
                        move, score = analyze_position(engine, board)

                        if move is None:
                            overlay.clear()
                            print("Game over or no legal move available.")

                        else:
                            overlay.draw_move(move, region, white_bottom)

                            print(
                                f"Your best move: {move.uci()} "
                                f"({board.san(move)}) | "
                                f"Evaluation: {format_score(score)}"
                            )

                last_error = None

            except BoardError as exc:
                overlay.clear()
                region = None

                if str(exc) != last_error:
                    print(f"[{time.strftime('%H:%M:%S')}] {exc}")

                last_error = str(exc)

            time.sleep(POLL_INTERVAL)

    finally:
        if overlay is not None:
            overlay.close()

        engine.quit()


def main():
    parser = argparse.ArgumentParser(
        description="Chessboard reader with Stockfish move overlay",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "command",
        nargs="?",
        choices=["calibrate"],
        help="calibrate on the starting position"
    )

    parser.add_argument(
        "--live",
        action="store_true",
        help="continuously read the board and show Stockfish's best move"
    )

    parser.add_argument(
        "--image",
        help="read a screenshot file instead of the screen"
    )

    args = parser.parse_args()
    make_dpi_aware()

    try:
        if args.command == "calibrate":
            cal, region = calibrate(screenshot(args.image))
            save_calibration(cal)

            print(
                f"Calibrated: board at x={region[0]} y={region[1]}, "
                f"{region[2]}px, "
                f"{'white' if cal['white_bottom'] else 'black'} at the bottom."
            )

            return

        cal = load_calibration()

        if args.live:
            if args.image:
                raise BoardError(
                    "--image is for single screenshot reads, not --live mode."
                )

            run_live(cal)
            return

        grid, unsure, white_bottom, _ = read_screen(
            cal,
            image_path=args.image
        )

        print(report(grid, unsure, white_bottom))

    except BoardError as exc:
        sys.exit(str(exc))

    except KeyboardInterrupt:
        print("Stopped.")


if __name__ == "__main__":
    main()
