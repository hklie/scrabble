"""analyze_board.py — Board Image Analyzer & Best Move Finder for Spanish Scrabble.

Reads a board image, optionally a rack (image or text), identifies tiles,
computes remaining tiles, and finds the highest-scoring legal plays.

Usage:
    python analyze_board.py boards/Board1.jpg --rack "AGUEIDA"
    python analyze_board.py boards/Board1.jpg --rack boards/rack1.jpg --debug
"""

import argparse
import os
import sys
from dataclasses import dataclass, field
from collections import Counter

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    LEXICON_FISE2, SCRABBLE_TILES, SCRABBLE_POINTS, DIGRAPHS, DIGRAPH_MAP,
    BOARDS_PATH, TOTAL_BLANKS, TOTAL_TILES, ALL_TILES, INTERNAL_POINTS,
    PREMIUM_SQUARES,
)
from preprocessing import tokenize_word, detokenize_word

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_internal(tile):
    """Convert display tile ('ch','ll','rr') to internal ('1','2','3')."""
    return DIGRAPH_MAP.get(tile, tile)


def to_display(tile):
    """Convert internal tile ('1','2','3') to display ('ch','ll','rr')."""
    return DIGRAPHS.get(tile, tile)


def tile_points(tile):
    """Point value for an internal-coded tile."""
    return INTERNAL_POINTS.get(tile, 0)


def row_letter(row):
    return chr(ord('A') + row)


def col_number(col):
    return col + 1


def pos_notation(row, col):
    return f"{row_letter(row)}{col_number(col)}"


# ---------------------------------------------------------------------------
# Part A — Board Image Analysis
# ---------------------------------------------------------------------------

def find_board_bounds(img):
    """Return (x, y, w, h) of the 15×15 playing area inside *img* (BGR)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for cnt in contours[:5]:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            aspect = w / h if h else 0
            # Board should be roughly square and large
            if 0.7 < aspect < 1.4 and w > img.shape[1] * 0.4:
                return (x, y, w, h)

    # Fallback: assume entire image is the board
    return (0, 0, img.shape[1], img.shape[0])


def extract_cells(img, bounds):
    """Return a 15×15 list-of-lists of cell images (BGR numpy arrays)."""
    x, y, w, h = bounds
    board_img = img[y:y+h, x:x+w]
    cell_h = h / 15
    cell_w = w / 15
    cells = []
    for r in range(15):
        row = []
        for c in range(15):
            y1 = int(r * cell_h)
            y2 = int((r + 1) * cell_h)
            x1 = int(c * cell_w)
            x2 = int((c + 1) * cell_w)
            row.append(board_img[y1:y2, x1:x2])
        cells.append(row)
    return cells


def _get_subscript_region(cell_img):
    """Extract the subscript zone: bottom 30% height, right 40% width."""
    ch, cw = cell_img.shape[:2]
    y_start = int(ch * 0.70)
    x_start = int(cw * 0.60)
    sub_img = cell_img[y_start:, x_start:]
    sub_gray = cv2.cvtColor(sub_img, cv2.COLOR_BGR2GRAY)
    return sub_img, sub_gray


def _has_subscript(cell_img):
    """Check if the cell has a subscript digit (present on every real tile).

    Real subscript digits produce 3-15 dark pixels (~2-9% of the zone).
    Premium squares / dark empty cells fill 85%+ of the zone — reject those.
    """
    _, sub_gray = _get_subscript_region(cell_img)
    dark_count = np.count_nonzero(sub_gray < 100)
    total = max(sub_gray.size, 1)
    return dark_count >= 3 and dark_count / total < 0.50


def _get_point_class(cell_img):
    """Classify tile by subscript digit size: '1pt', 'multi', '10pt', or 'unknown'.

    1-pt tiles (A,E,I,L,N,O,R,S,T,U) have a tiny '1' subscript (~4 dark px, span=1).
    Multi-pt tiles (2-8 pts) have single-digit subscripts (8-13 dark px, span=5).
    10-pt tile (Z only) has two-digit '10' subscript (span >= 7).
    """
    _, sub_gray = _get_subscript_region(cell_img)
    dark_mask = sub_gray < 100
    dark_count = np.count_nonzero(dark_mask)

    # Measure ink width and column span (first to last dark column)
    cols_with_ink = np.any(dark_mask, axis=0)
    ink_width = np.count_nonzero(cols_with_ink)
    dark_cols = np.where(cols_with_ink)[0]
    span = (dark_cols[-1] - dark_cols[0] + 1) if len(dark_cols) > 0 else 0

    if span >= 7:
        return '10pt'
    if dark_count >= 7 and ink_width >= 3:
        return 'multi'
    if 2 <= dark_count <= 5:
        return '1pt'
    return 'unknown'


def cell_has_tile(cell_img):
    """Detect whether a cell contains a tile using grayscale + subscript confirmation.

    Two detection paths:
    1. Blank tiles: detected by red ink in center (color check)
    2. Normal tiles: dark letter ink (>10% dark pixels) + subscript digit
    """
    ch, cw = cell_img.shape[:2]
    margin_r, margin_c = ch // 4, cw // 4
    center = cell_img[margin_r:ch-margin_r, margin_c:cw-margin_c]
    total = max(center.shape[0] * center.shape[1], 1)

    # Path 1: Blank tile — red ink makes grayscale look uniformly dark,
    # so detect via color. Red letter ink on white/cream tile = 5-50% red.
    # Uniformly red cells (>50%) are TW premium squares, not tiles.
    b, g, r = cv2.split(center)
    red_mask = (r > 150) & (g.astype(int) < 100) & (b.astype(int) < 100)
    red_ratio = np.count_nonzero(red_mask) / total
    if 0.05 < red_ratio < 0.50:
        return True

    # Path 2: Normal tile — needs dark letter ink on lighter tile background
    gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
    dark_ratio = np.count_nonzero(gray < 80) / max(gray.size, 1)
    if dark_ratio <= 0.10:
        return False

    # Subscript confirmation: real tiles have a point-value subscript digit
    return _has_subscript(cell_img)



def is_blank_tile(cell_img):
    """Detect whether a tile has red text (= blank tile, worth 0 pts).

    Samples center 50% to avoid grid line artifacts. Uses ratio threshold.
    """
    ch, cw = cell_img.shape[:2]
    margin_r, margin_c = ch // 4, cw // 4
    center = cell_img[margin_r:ch-margin_r, margin_c:cw-margin_c]

    b, g, r = cv2.split(center)
    red_mask = (r > 150) & (g.astype(int) < 100) & (b.astype(int) < 100)
    total = max(center.shape[0] * center.shape[1], 1)
    return np.count_nonzero(red_mask) / total > 0.05



def _ocr_mask(mask, reader, allowlist, low_confidence=False):
    """Run easyocr on a binary mask (upscaled 4×). Return uppercase text."""
    mask_big = cv2.resize(mask, (mask.shape[1] * 4, mask.shape[0] * 4),
                          interpolation=cv2.INTER_NEAREST)
    kwargs = dict(allowlist=allowlist, paragraph=False, detail=1)
    if low_confidence:
        kwargs['text_threshold'] = 0.3
        kwargs['low_text'] = 0.3
    results = reader.readtext(mask_big, **kwargs)
    if results:
        best = max(results, key=lambda r: r[2])
        return best[1].upper().strip()
    return ''


def recognize_letter(cell_img, blank, reader):
    """OCR a single tile cell, returning a lowercase tile token.

    Uses multi-pass OCR followed by subscript point-class disambiguation
    for N/Ñ, R/RR, L/LL confusions.
    """
    ch, cw = cell_img.shape[:2]

    # Crop top 75% to exclude the subscript point value
    crop = cell_img[:int(ch * 0.75), :]

    # Build binary mask: isolate the letter pixels
    if blank:
        b, g, r = cv2.split(crop)
        mask = ((r > 130) & (g.astype(int) < 120) & (b.astype(int) < 120)).astype(np.uint8) * 255
    else:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    # Ink dimensions on the clean mask
    cols_with_ink = np.any(mask > 0, axis=0)
    ink_width = np.count_nonzero(cols_with_ink)
    rows_with_ink = np.any(mask > 0, axis=1)
    ink_height = np.count_nonzero(rows_with_ink)
    likely_digraph = ink_width > cw * 0.55

    # Higher-threshold mask for thin-stroke recovery (non-blank only)
    mask_hi = None
    if not blank:
        _, mask_hi = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY_INV)

    # --- Pass 1: OCR with conservative threshold ---
    allowlist = 'ABCDEFGHIJLMNOPQRSTUVXYZ'
    text = _ocr_mask(mask, reader, allowlist)

    # --- Pass 2: if pass 1 failed, retry with higher threshold ---
    if not text or len(text) > 2:
        if not blank:
            text2 = _ocr_mask(mask_hi, reader, allowlist)
            if text2 and len(text2) <= 2:
                text = text2
                cols2 = np.any(mask_hi > 0, axis=0)
                ink_width = max(ink_width, np.count_nonzero(cols2))
                likely_digraph = ink_width > cw * 0.55

    # --- Pass 3: low-confidence OCR with full alphabet ---
    if not text or len(text) > 2:
        if not blank:
            text3 = _ocr_mask(mask_hi, reader, allowlist, low_confidence=True)
            if text3 and len(text3) <= 2:
                text = text3

    # --- Digraph handling (explicit two-char OCR results) ---
    if text in ('CH', 'C H'):
        return 'ch'
    if text in ('LL', 'L L'):
        if likely_digraph:
            return 'll'
        # Narrow "LL" — fall through to point-class check for L
        text = 'L'
    if text in ('RR', 'R R'):
        if likely_digraph:
            return 'rr'
        # Narrow "RR" is likely Ñ (tilde misread as second character)
        return 'ñ'

    # Width-based digraph fallback
    if likely_digraph and len(text) <= 1:
        if text in ('R', ''):
            return 'rr'
        if text in ('L', ''):
            return 'll'

    # --- Point-class disambiguation for N/Ñ, R/RR, L/LL ---
    # Skip for blank tiles — their subscript is always "0" (0 pts), not the tile value
    point_class = _get_point_class(cell_img) if not blank else 'unknown'

    if text == 'N':
        if point_class == 'multi':
            return 'ñ'      # Ñ is 8 pts, N is 1 pt
        return 'n'

    if text == 'R' and not likely_digraph:
        if point_class == 'multi':
            return 'ñ'      # Narrow multi-pt 'R' is likely Ñ (8 pts); real RR would be wide
        return 'r'

    if text == 'L' and not likely_digraph:
        if point_class == 'multi':
            return 'll'     # LL is 8 pts, L is 1 pt
        return 'l'

    if len(text) == 1 and text.isalpha():
        return text.lower()

    # --- Subscript-based identification for Z (only 10-pt tile) ---
    if point_class == '10pt':
        return 'z'

    # --- Shape-based fallbacks for thin / tricky characters ---
    # Use higher-threshold mask for shape analysis (captures thin strokes better)
    shape_mask = mask_hi if (not blank and mask_hi is not None) else mask
    s_cols = np.any(shape_mask > 0, axis=0)
    s_rows = np.any(shape_mask > 0, axis=1)
    s_ink_w = np.count_nonzero(s_cols)
    s_ink_h = np.count_nonzero(s_rows)

    if s_ink_h > ch * 0.25 and s_ink_w > 0:
        # Very narrow → I
        if s_ink_w < cw * 0.25:
            return 'i'

        # Use ink bounding box for Y detection (letters don't start at row 0)
        ink_rows = np.where(s_rows)[0]
        y_top, y_bot = ink_rows[0], ink_rows[-1]
        ink_region = shape_mask[y_top:y_bot+1, :]
        ink_h = ink_region.shape[0]

        if ink_h >= 4:
            quarter = max(ink_h // 4, 1)
            top_q = np.count_nonzero(np.any(ink_region[:quarter, :] > 0, axis=0))
            bot_q = np.count_nonzero(np.any(ink_region[-quarter:, :] > 0, axis=0))

            # Y shape: top quarter wider than bottom quarter (fork arms → stem)
            if top_q > bot_q * 1.15 and bot_q > 0 and s_ink_w > cw * 0.25:
                return 'y'

        # Z shape: ink clusters at top, middle-diagonal, and bottom
        third = shape_mask.shape[0] // 3
        top_ink = np.count_nonzero(shape_mask[:third, :])
        mid_ink = np.count_nonzero(shape_mask[third:2*third, :])
        bot_ink = np.count_nonzero(shape_mask[2*third:, :])
        if top_ink > 0 and bot_ink > 0 and mid_ink > 0:
            top_w2 = np.count_nonzero(np.any(shape_mask[:third, :] > 0, axis=0))
            bot_w2 = np.count_nonzero(np.any(shape_mask[2*third:, :] > 0, axis=0))
            if top_w2 > cw * 0.25 and bot_w2 > cw * 0.25:
                return 'z'

    return '?'


def validate_board_tiles(board, debug=False):
    """Post-OCR correction: fix tiles that exceed bag limits.

    Converts excess Ñ→N, RR→R, LL→L (the most common OCR confusions).
    Logs warnings for other over-counts.
    """
    # Count non-blank tiles on the board
    counts = Counter()
    positions = {}  # tile → list of (r, c)
    for r in range(15):
        for c in range(15):
            if board[r][c] is not None:
                tile, is_blank = board[r][c]
                if not is_blank:
                    display = to_display(tile)
                    counts[display] += 1
                    positions.setdefault(display, []).append((r, c))

    # Correction map: over-counted tile → replacement tile
    corrections = {'ñ': 'n', 'rr': 'r', 'll': 'l'}

    for tile_display, max_count in SCRABBLE_TILES.items():
        if counts[tile_display] > max_count:
            excess = counts[tile_display] - max_count
            if tile_display in corrections:
                replacement = corrections[tile_display]
                repl_internal = to_internal(replacement)
                # Convert the last N excess occurrences
                for r, c in positions[tile_display][-excess:]:
                    board[r][c] = (repl_internal, False)
                    if debug:
                        print(f"  Validation: {tile_display.upper()} at {pos_notation(r, c)}"
                              f" → {replacement.upper()} (bag limit {max_count})")
            else:
                if debug:
                    print(f"  Warning: {tile_display.upper()} count {counts[tile_display]}"
                          f" exceeds bag limit {max_count}")


def read_board_image(path, reader, debug=False):
    """Read a board image and return a 15×15 board + metadata.

    Returns:
        board: 15×15 list, each cell is None or (internal_tile, is_blank)
        blanks_info: list of (row, col, letter_display) for blank tiles
    """
    img = cv2.imread(path)
    if img is None:
        print(f"Error: cannot read image {path}")
        sys.exit(1)

    bounds = find_board_bounds(img)
    cells = extract_cells(img, bounds)

    board = [[None] * 15 for _ in range(15)]
    blanks_info = []

    for r in range(15):
        for c in range(15):
            cell_img = cells[r][c]
            if not cell_has_tile(cell_img):
                continue

            blank = is_blank_tile(cell_img)
            letter = recognize_letter(cell_img, blank, reader)

            if letter == '?':
                if debug:
                    print(f"  Warning: unrecognized tile at {pos_notation(r, c)}")
                continue

            internal = to_internal(letter)
            board[r][c] = (internal, blank)

            if blank:
                blanks_info.append((r, c, to_display(internal)))

            if debug:
                pt_class = _get_point_class(cell_img)
                marker = '*' if blank else ''
                print(f"  {pos_notation(r, c)}: {to_display(internal).upper()}{marker}"
                      f" [{pt_class}]")

    validate_board_tiles(board, debug=debug)
    return board, blanks_info


def read_rack_image(path, reader):
    """OCR a rack image — returns a list of internal tile tokens.

    Strategy: crop top portion (exclude subscript point values), run OCR on
    the full strip without upscaling, sort detections left-to-right.
    Tries multiple crop heights and picks the best result (closest to 7 tiles,
    highest confidence).
    """
    img = cv2.imread(path)
    if img is None:
        print(f"Error: cannot read rack image {path}")
        sys.exit(1)

    h, w = img.shape[:2]
    allowlist = 'ABCDEFGHIJLMNOPQRSTUVXYZ'
    candidates = []

    for crop_pct in [0.65, 0.70, 0.75]:
        crop = img[:int(h * crop_pct), :]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

        results = reader.readtext(mask, allowlist=allowlist,
                                  paragraph=False, detail=1)
        if not results:
            continue

        # Sort detections left-to-right, concatenate
        results.sort(key=lambda r: r[0][0][0])
        text = ''.join(r[1].upper() for r in results)
        min_conf = min(r[2] for r in results)

        # Filter low-confidence detections (subscript digit artifacts)
        filtered = [r for r in results if r[2] > 0.5]
        filtered.sort(key=lambda r: r[0][0][0])
        text_f = ''.join(r[1].upper() for r in filtered)
        min_conf_f = min((r[2] for r in filtered), default=0)

        # Prefer the version closest to 7 characters with best confidence
        for txt, mc in [(text, min_conf), (text_f, min_conf_f)]:
            if txt:
                tokens = tokenize_word(txt.lower())
                candidates.append((tokens, mc, abs(len(tokens) - 7)))

    if candidates:
        # Sort by: fewest excess/missing tiles, then highest min confidence
        candidates.sort(key=lambda c: (c[2], -c[1]))
        best_tokens = candidates[0][0][:7]  # Hard cap at 7 tiles
        return [to_internal(t) for t in best_tokens]

    return []


def parse_rack_text(text):
    """Parse rack text like 'AGUEIDA' or 'A,G,U,E,I,D,A' or with '?' for blanks."""
    text = text.strip()
    if ',' in text:
        tokens = [t.strip().lower() for t in text.split(',')]
    else:
        tokens = []
        t = text.lower()
        i = 0
        while i < len(t):
            if t[i] == '?':
                tokens.append('?')
                i += 1
            elif i + 1 < len(t) and t[i:i+2] in DIGRAPH_MAP:
                tokens.append(t[i:i+2])
                i += 2
            else:
                tokens.append(t[i])
                i += 1

    return [to_internal(t) if t != '?' else '?' for t in tokens]


def compute_remaining_tiles(board, rack):
    """Compute remaining unseen tiles.

    Returns:
        remaining: Counter of internal tile tokens
        blanks_remaining: int
        warnings: list of warning strings
    """
    full = Counter()
    for tile, count in SCRABBLE_TILES.items():
        full[to_internal(tile)] = count
    blanks_total = TOTAL_BLANKS

    # Subtract board tiles
    blanks_on_board = 0
    for r in range(15):
        for c in range(15):
            if board[r][c] is not None:
                tile, is_blank = board[r][c]
                if is_blank:
                    blanks_on_board += 1
                else:
                    full[tile] -= 1

    blanks_remaining = blanks_total - blanks_on_board

    # Subtract rack tiles
    for t in rack:
        if t == '?':
            blanks_remaining -= 1
        else:
            full[t] -= 1

    warnings = []
    for tile, count in full.items():
        if count < 0:
            warnings.append(f"  Tile '{to_display(tile)}' count went to {count} — possible OCR error")
            full[tile] = 0
    if blanks_remaining < 0:
        warnings.append(f"  Blanks count went to {blanks_remaining} — possible OCR error")
        blanks_remaining = 0

    return full, blanks_remaining, warnings


# ---------------------------------------------------------------------------
# Part B — Trie & Move Generation
# ---------------------------------------------------------------------------

TRIE_TERMINAL = ''   # sentinel key: present in dict iff node is a word end


def _build_trie_from_file(lexicon_path):
    """Build a trie as nested dicts. Presence of TRIE_TERMINAL key marks word end."""
    root = {}
    with open(lexicon_path, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            if not word:
                continue
            node = root
            for ch in word:
                if ch not in node:
                    node[ch] = {}
                node = node[ch]
            node[TRIE_TERMINAL] = True
    return root


def build_trie(lexicon_path):
    """Load trie from pickle cache if fresh, otherwise build and cache."""
    import pickle
    cache_path = lexicon_path + '.trie.pkl'
    lexicon_mtime = os.path.getmtime(lexicon_path)

    if os.path.exists(cache_path):
        if os.path.getmtime(cache_path) >= lexicon_mtime:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

    root = _build_trie_from_file(lexicon_path)
    with open(cache_path, 'wb') as f:
        pickle.dump(root, f, protocol=pickle.HIGHEST_PROTOCOL)
    return root


def _word_in_trie(trie, tokens):
    """Validate that a word (list of internal tokens) exists in the trie."""
    node = trie
    for t in tokens:
        if t in node:
            node = node[t]
        else:
            return False
    return TRIE_TERMINAL in node


def transpose_board(board):
    return [[board[c][r] for c in range(15)] for r in range(15)]


def compute_cross_checks(board, trie):
    """For each empty cell, compute which tiles can go there based on
    perpendicular (vertical) neighbours.  Returns 15×15 grid of sets."""
    all_set = set(ALL_TILES)
    checks = [[None] * 15 for _ in range(15)]

    for row in range(15):
        for col in range(15):
            if board[row][col] is not None:
                checks[row][col] = set()          # occupied
                continue

            # Collect tiles above
            above = []
            r = row - 1
            while r >= 0 and board[r][col] is not None:
                above.append(board[r][col][0])
                r -= 1
            above.reverse()

            # Collect tiles below
            below = []
            r = row + 1
            while r < 15 and board[r][col] is not None:
                below.append(board[r][col][0])
                r += 1

            if not above and not below:
                checks[row][col] = all_set        # no constraints
                continue

            # Only tiles that form a valid perpendicular word
            valid = set()
            for tile in ALL_TILES:
                word = above + [tile] + below
                node = trie
                ok = True
                for t in word:
                    if t in node:
                        node = node[t]
                    else:
                        ok = False
                        break
                if ok and TRIE_TERMINAL in node:
                    valid.add(tile)
            checks[row][col] = valid

    return checks


# ---- Move generation (Appel–Jacobson algorithm) ----

@dataclass
class Move:
    row: int                              # 0-indexed start row (original board)
    col: int                              # 0-indexed start col (original board)
    direction: str                        # 'horizontal' | 'vertical'
    word_tokens: list = field(default_factory=list)   # all tiles in the word
    new_tile_positions: list = field(default_factory=list)  # [(offset, tile, is_blank)]
    score: int = 0

    @property
    def word_display(self):
        return ''.join(to_display(t) for t in self.word_tokens).upper()

    @property
    def pos_str(self):
        """Position string: 'B14' for horizontal, '14B' for vertical."""
        if self.direction == 'horizontal':
            return f"{row_letter(self.row)}{col_number(self.col)}"
        else:
            return f"{col_number(self.col)}{row_letter(self.row)}"

    @property
    def dir_arrow(self):
        return '→' if self.direction == 'horizontal' else '↓'

    @property
    def notation(self):
        return f"{self.word_display:16s} {self.pos_str:5s} {self.dir_arrow}"


def _find_row_anchors(row_idx, board):
    """Return set of column indices that are anchors in this row."""
    board_empty = all(board[r][c] is None for r in range(15) for c in range(15))
    if board_empty:
        return {7} if row_idx == 7 else set()

    anchors = set()
    for col in range(15):
        if board[row_idx][col] is not None:
            continue
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row_idx + dr, col + dc
            if 0 <= nr < 15 and 0 <= nc < 15 and board[nr][nc] is not None:
                anchors.add(col)
                break
    return anchors


def _record_move(word_tokens, row, start_col, new_tile_info, moves):
    new_tile_positions = [(c - start_col, tile, blank)
                          for c, tile, blank in new_tile_info]
    moves.append(Move(
        row=row,
        col=start_col,
        direction='horizontal',
        word_tokens=list(word_tokens),
        new_tile_positions=new_tile_positions,
    ))


def _extend_right(partial, node, col, row, start_col, rack,
                   board_row, cross_checks_row, moves, new_info,
                   anchor_col):
    """Extend word rightward from *col*.  Only record moves once we have
    gone past *anchor_col* (ensuring the word connects to existing tiles)."""
    if col >= 15:
        if TRIE_TERMINAL in node and new_info and col > anchor_col:
            _record_move(partial, row, start_col, new_info, moves)
        return

    cell = board_row[col]
    if cell is not None:
        # Existing tile — must follow it
        tile = cell[0]
        if tile in node:
            _extend_right(partial + [tile], node[tile], col + 1,
                          row, start_col, rack, board_row,
                          cross_checks_row, moves, new_info, anchor_col)
    else:
        # Empty cell — can terminate (only if past anchor) or extend
        if TRIE_TERMINAL in node and new_info and col > anchor_col:
            _record_move(partial, row, start_col, new_info, moves)

        valid = cross_checks_row[col]
        tried = set()
        for i, rack_tile in enumerate(rack):
            if rack_tile in tried:
                continue
            tried.add(rack_tile)
            new_rack = rack[:i] + rack[i+1:]

            if rack_tile == '?':
                # Blank — try every letter
                for tile in ALL_TILES:
                    if tile in valid and tile in node:
                        _extend_right(
                            partial + [tile], node[tile], col + 1,
                            row, start_col, new_rack, board_row,
                            cross_checks_row, moves,
                            new_info + [(col, tile, True)], anchor_col)
            else:
                if rack_tile in valid and rack_tile in node:
                    _extend_right(
                        partial + [rack_tile], node[rack_tile], col + 1,
                        row, start_col, new_rack, board_row,
                        cross_checks_row, moves,
                        new_info + [(col, rack_tile, False)], anchor_col)


def _left_part(partial, node, limit, anchor_col, row, rack,
               board_row, cross_checks_row, moves, new_info):
    # Try extending right from what we have so far
    start_col = anchor_col - len(partial)
    _extend_right(partial, node, anchor_col, row, start_col, rack,
                  board_row, cross_checks_row, moves, new_info, anchor_col)

    if limit > 0:
        col = anchor_col - len(partial) - 1
        if col < 0:
            return
        valid = cross_checks_row[col]
        tried = set()
        for i, rack_tile in enumerate(rack):
            if rack_tile in tried:
                continue
            tried.add(rack_tile)
            new_rack = rack[:i] + rack[i+1:]

            if rack_tile == '?':
                for tile in ALL_TILES:
                    if tile in valid and tile in node:
                        _left_part(
                            [tile] + partial, node[tile], limit - 1,
                            anchor_col, row, new_rack, board_row,
                            cross_checks_row, moves,
                            [(col, tile, True)] + new_info)
            else:
                if rack_tile in valid and rack_tile in node:
                    _left_part(
                        [rack_tile] + partial, node[rack_tile], limit - 1,
                        anchor_col, row, new_rack, board_row,
                        cross_checks_row, moves,
                        [(col, rack_tile, False)] + new_info)


def _generate_horizontal_moves(board, rack, trie_root, cross_checks):
    """Generate all legal horizontal moves on *board*."""
    moves = []
    for row in range(15):
        anchors = _find_row_anchors(row, board)
        if not anchors:
            continue
        board_row = board[row]
        cc_row = cross_checks[row]

        for anchor_col in sorted(anchors):
            if anchor_col > 0 and board_row[anchor_col - 1] is not None:
                # Left part is already on the board — read it
                left_tiles = []
                c = anchor_col - 1
                while c >= 0 and board_row[c] is not None:
                    left_tiles.append(board_row[c][0])
                    c -= 1
                left_tiles.reverse()

                node = trie_root
                ok = True
                for t in left_tiles:
                    if t in node:
                        node = node[t]
                    else:
                        ok = False
                        break
                if ok:
                    start_col = anchor_col - len(left_tiles)
                    _extend_right(left_tiles, node, anchor_col, row, start_col,
                                  rack, board_row, cc_row, moves, [],
                                  anchor_col)
            else:
                # Determine how many empty non-anchor cells to the left
                left_limit = 0
                c = anchor_col - 1
                while c >= 0 and board_row[c] is None and c not in anchors:
                    left_limit += 1
                    c -= 1
                left_limit = min(left_limit, len(rack) - 1) if rack else 0

                _left_part([], trie_root, left_limit, anchor_col, row,
                           rack, board_row, cc_row, moves, [])
    return moves


# ---------------------------------------------------------------------------
# Part C — Scoring
# ---------------------------------------------------------------------------

def score_move(move, board):
    """Compute the score of a move on the original board (before placement)."""
    new_map = {}
    for offset, tile, blank in move.new_tile_positions:
        new_map[offset] = (tile, blank)

    # --- Main word ---
    main_score = 0
    word_mult = 1
    for i, token in enumerate(move.word_tokens):
        if move.direction == 'horizontal':
            r, c = move.row, move.col + i
        else:
            r, c = move.row + i, move.col

        if i in new_map:
            _, is_blank = new_map[i]
            lv = 0 if is_blank else tile_points(token)
            sq = PREMIUM_SQUARES.get((r, c))
            if sq == 'DL':
                lv *= 2
            elif sq == 'TL':
                lv *= 3
            elif sq == 'DW':
                word_mult *= 2
            elif sq == 'TW':
                word_mult *= 3
            main_score += lv
        else:
            # Existing tile — face value, no multiplier
            main_score += tile_points(token)

    main_score *= word_mult

    # --- Cross-word scores ---
    cross_total = 0
    for offset, tile, is_blank in move.new_tile_positions:
        if move.direction == 'horizontal':
            r, c = move.row, move.col + offset
            dr, dc = 1, 0          # perpendicular = vertical
        else:
            r, c = move.row + offset, move.col
            dr, dc = 0, 1          # perpendicular = horizontal

        perp = []
        # Backward
        nr, nc = r - dr, c - dc
        while 0 <= nr < 15 and 0 <= nc < 15 and board[nr][nc] is not None:
            perp.append((nr, nc, board[nr][nc][0], board[nr][nc][1]))
            nr -= dr
            nc -= dc
        perp.reverse()

        # The new tile itself
        perp.append((r, c, tile, is_blank))

        # Forward
        nr, nc = r + dr, c + dc
        while 0 <= nr < 15 and 0 <= nc < 15 and board[nr][nc] is not None:
            perp.append((nr, nc, board[nr][nc][0], board[nr][nc][1]))
            nr += dr
            nc += dc

        if len(perp) <= 1:
            continue

        cw_score = 0
        cw_mult = 1
        for pr, pc, pt, pb in perp:
            if pr == r and pc == c:
                lv = 0 if is_blank else tile_points(pt)
                sq = PREMIUM_SQUARES.get((pr, pc))
                if sq == 'DL':
                    lv *= 2
                elif sq == 'TL':
                    lv *= 3
                elif sq == 'DW':
                    cw_mult *= 2
                elif sq == 'TW':
                    cw_mult *= 3
                cw_score += lv
            else:
                lv = 0 if pb else tile_points(pt)
                cw_score += lv
        cross_total += cw_score * cw_mult

    total = main_score + cross_total

    # Bingo bonus (all 7 rack tiles used)
    if len(move.new_tile_positions) == 7:
        total += 50

    return total


def find_best_moves(board, rack, trie, top_n=10):
    """Return the top-N scoring legal moves."""
    # Cross-checks for horizontal moves
    cross_h = compute_cross_checks(board, trie)
    h_moves = _generate_horizontal_moves(board, rack, trie, cross_h)

    # Vertical moves via transpose
    board_t = transpose_board(board)
    cross_v = compute_cross_checks(board_t, trie)
    v_moves_t = _generate_horizontal_moves(board_t, rack, trie, cross_v)

    # Convert transposed moves back to original coordinates
    v_moves = []
    for m in v_moves_t:
        v_moves.append(Move(
            row=m.col,
            col=m.row,
            direction='vertical',
            word_tokens=m.word_tokens,
            new_tile_positions=m.new_tile_positions,
        ))

    all_moves = h_moves + v_moves

    # Validate all moves: every word must exist in the trie
    valid_moves = []
    for m in all_moves:
        if _word_in_trie(trie, m.word_tokens):
            valid_moves.append(m)

    # Score all valid moves
    for m in valid_moves:
        m.score = score_move(m, board)

    # Deduplicate (same word, same position, same direction)
    seen = set()
    unique = []
    for m in valid_moves:
        key = (tuple(m.word_tokens), m.row, m.col, m.direction)
        if key not in seen:
            seen.add(key)
            unique.append(m)

    unique.sort(key=lambda m: m.score, reverse=True)
    return unique[:top_n]


# ---------------------------------------------------------------------------
# Part D — CLI & Output
# ---------------------------------------------------------------------------

def print_board_summary(board, blanks_info):
    regular = sum(1 for r in range(15) for c in range(15)
                  if board[r][c] is not None and not board[r][c][1])
    blank_count = len(blanks_info)
    total = regular + blank_count
    print(f"\n=== Board Analysis ===")
    print(f"Tiles on board: {regular} regular + {blank_count} blanks ({total} total)")
    if blanks_info:
        parts = [f"{letter.upper()} ({pos_notation(r, c)})"
                 for r, c, letter in blanks_info]
        print(f"Blanks used as: {', '.join(parts)}")


def print_rack(rack):
    display = [to_display(t).upper() if t != '?' else '?' for t in rack]
    print(f"\n=== Rack ===")
    print(f"{'  '.join(display)}  ({len(rack)} tiles)")


def print_remaining(remaining, blanks_remaining, warnings):
    total = sum(remaining.values()) + blanks_remaining
    print(f"\n=== Remaining Tiles ({total}) ===")
    if warnings:
        for w in warnings:
            print(w)

    # Group by point value
    by_pts = {}
    for tile, count in remaining.items():
        if count <= 0:
            continue
        pts = tile_points(tile)
        by_pts.setdefault(pts, []).append((tile, count))

    for pts in sorted(by_pts):
        items = sorted(by_pts[pts], key=lambda x: to_display(x[0]))
        parts = [f"{to_display(t)}×{cnt}" for t, cnt in items]
        print(f"  {pts}-pt: {'  '.join(parts)}")
    print(f"  Blanks: {blanks_remaining}")
    print(f"  Total unseen: {total}")


def print_moves(moves, top_n):
    n = min(len(moves), top_n)
    print(f"\n=== Top {n} Moves ===")
    if not moves:
        print("  No legal moves found.")
        return
    for i, m in enumerate(moves[:n], 1):
        print(f"  {i:2d}. {m.notation} — {m.score} pts")


def main():
    parser = argparse.ArgumentParser(
        description='Spanish Scrabble Board Analyzer & Best Move Finder')
    parser.add_argument('board', help='Path to board image')
    parser.add_argument('--rack', '-r',
                        help='Rack image path or letters (e.g. AGUEIDA)')
    parser.add_argument('--top', '-n', type=int, default=10,
                        help='Show top N moves (default 10)')
    parser.add_argument('--debug', '-d', action='store_true',
                        help='Print per-cell OCR debug info')
    parser.add_argument('--corrections', '-c',
                        help='Manual OCR corrections, e.g. "D6=Y,B14=Z"')
    args = parser.parse_args()

    # Init easyocr (uses GPU if available)
    import easyocr
    print("Loading OCR model...")
    reader = easyocr.Reader(['es'], gpu=True, verbose=False)

    # --- Read board ---
    print("Analyzing board image...")
    board, blanks_info = read_board_image(args.board, reader, debug=args.debug)

    # Apply manual corrections  (e.g. --corrections "D6=Y,B14=Z")
    if args.corrections:
        for fix in args.corrections.split(','):
            fix = fix.strip()
            if '=' not in fix:
                continue
            pos, letter = fix.split('=', 1)
            pos = pos.strip()
            letter = letter.strip().lower()
            # Parse position: letter+number (e.g. D6)
            r_ch = pos[0].upper()
            c_num = int(pos[1:])
            r_idx = ord(r_ch) - ord('A')
            c_idx = c_num - 1
            if 0 <= r_idx < 15 and 0 <= c_idx < 15:
                internal = to_internal(letter)
                is_blank = letter.startswith('*')
                if is_blank:
                    internal = to_internal(letter[1:])
                board[r_idx][c_idx] = (internal, is_blank)
                print(f"  Correction applied: {pos} = {to_display(internal).upper()}"
                      f"{'*' if is_blank else ''}")

    print_board_summary(board, blanks_info)

    # --- Read rack ---
    rack = []
    if args.rack:
        # Decide if it is a file path or text
        if os.path.isfile(args.rack):
            print("Reading rack image...")
            rack = read_rack_image(args.rack, reader)
        else:
            rack = parse_rack_text(args.rack)
        print_rack(rack)

    # --- Remaining tiles ---
    remaining, blanks_remaining, warnings = compute_remaining_tiles(board, rack)
    print_remaining(remaining, blanks_remaining, warnings)

    # --- Find best moves ---
    if not rack:
        print("\nNo rack provided — skipping move generation.")
        return

    print("\nLoading lexicon trie...")
    trie = build_trie(LEXICON_FISE2)
    print(f"Finding best moves...")
    moves = find_best_moves(board, rack, trie, args.top)
    print_moves(moves, args.top)


if __name__ == '__main__':
    main()
