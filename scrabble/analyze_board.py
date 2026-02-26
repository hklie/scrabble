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


def cell_has_tile(cell_img):
    """Detect whether a cell contains a tile (yellow/beige region)."""
    hsv = cv2.cvtColor(cell_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Sample the center 50% of the cell to avoid grid lines
    ch, cw = cell_img.shape[:2]
    margin_r, margin_c = ch // 4, cw // 4
    center_h = h[margin_r:ch-margin_r, margin_c:cw-margin_c]
    center_s = s[margin_r:ch-margin_r, margin_c:cw-margin_c]
    center_v = v[margin_r:ch-margin_r, margin_c:cw-margin_c]

    # Yellow/beige tile: H∈[15,35], S>40, V>160
    yellow_mask = (center_h >= 15) & (center_h <= 35) & (center_s > 40) & (center_v > 160)
    yellow_ratio = np.count_nonzero(yellow_mask) / max(yellow_mask.size, 1)

    # White tile (blank): S<50, V>200
    white_mask = (center_s < 50) & (center_v > 200)
    white_ratio = np.count_nonzero(white_mask) / max(white_mask.size, 1)

    return yellow_ratio > 0.25 or white_ratio > 0.40


def _cell_has_letter_content(cell_img):
    """Secondary check: verify the cell actually has dark/red letter pixels.

    Filters out false positives from board markings (e.g. the center star).
    """
    ch, cw = cell_img.shape[:2]
    # Look in center region
    margin_r, margin_c = ch // 4, cw // 4
    center = cell_img[margin_r:ch-margin_r, margin_c:cw-margin_c]

    gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
    # Dark pixels (letter ink on normal tile)
    dark_count = np.count_nonzero(gray < 80)

    # Red pixels (letter ink on blank tile)
    b, g, r = cv2.split(center)
    red_count = np.count_nonzero((r > 130) & (g.astype(int) < 120) & (b.astype(int) < 120))

    # Need a meaningful amount of letter pixels
    total_pixels = max(center.shape[0] * center.shape[1], 1)
    return (dark_count + red_count) / total_pixels > 0.02


def is_blank_tile(cell_img):
    """Detect whether a tile has red text (= blank tile, worth 0 pts)."""
    b, g, r = cv2.split(cell_img)
    red_mask = (r > 150) & (g.astype(int) < 100) & (b.astype(int) < 100)
    red_count = np.count_nonzero(red_mask)
    return red_count > 40


def _has_tilde(mask):
    """Check if there's a tilde (~) above the main letter body via connected components."""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if num_labels <= 2:
        return False  # background + 1 component = no tilde

    # Find the largest foreground component
    areas = stats[1:, cv2.CC_STAT_AREA]
    main_idx = np.argmax(areas) + 1
    main_top = stats[main_idx, cv2.CC_STAT_TOP]

    # Any smaller component whose bottom edge is above the main body = tilde
    for i in range(1, num_labels):
        if i == main_idx:
            continue
        comp_bottom = stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]
        if comp_bottom < main_top + 2 and stats[i, cv2.CC_STAT_AREA] > 3:
            return True
    return False


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
    """OCR a single tile cell, returning a lowercase tile token."""
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

    # --- Ñ detection: check tilde FIRST (before OCR) on both thresholds ---
    if not blank:
        if _has_tilde(mask):
            return 'ñ'
        if _has_tilde(mask_hi):
            return 'ñ'

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

    # --- Digraph handling ---
    if text in ('CH', 'C H'):
        return 'ch'
    if text in ('LL', 'L L'):
        return 'll'
    if text in ('RR', 'R R'):
        return 'rr'

    # Width-based digraph fallback
    if likely_digraph and len(text) <= 1:
        if text in ('R', ''):
            return 'rr'
        if text in ('L', ''):
            return 'll'

    # N returned by OCR without tilde detected above → plain n
    if text == 'N':
        return 'n'

    if len(text) == 1 and text.isalpha():
        return text.lower()

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

        # Y shape: wider top half, narrow bottom half
        half = shape_mask.shape[0] // 2
        top_w = np.count_nonzero(np.any(shape_mask[:half, :] > 0, axis=0))
        bot_w = np.count_nonzero(np.any(shape_mask[half:, :] > 0, axis=0))
        if top_w > bot_w * 1.4 and bot_w > 0 and s_ink_w > cw * 0.25:
            return 'y'

        # Z shape: ink clusters at top, middle-diagonal, and bottom
        third = shape_mask.shape[0] // 3
        top_ink = np.count_nonzero(shape_mask[:third, :])
        mid_ink = np.count_nonzero(shape_mask[third:2*third, :])
        bot_ink = np.count_nonzero(shape_mask[2*third:, :])
        if top_ink > 0 and bot_ink > 0 and mid_ink > 0:
            # Z has horizontal strokes at top/bottom → wide, and diagonal in middle
            top_w2 = np.count_nonzero(np.any(shape_mask[:third, :] > 0, axis=0))
            bot_w2 = np.count_nonzero(np.any(shape_mask[2*third:, :] > 0, axis=0))
            if top_w2 > cw * 0.35 and bot_w2 > cw * 0.35 and text == '':
                return 'z'

    return '?'


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
            # Secondary filter: must have actual letter pixels
            if not _cell_has_letter_content(cell_img):
                if debug:
                    print(f"  Skipped non-tile at {pos_notation(r, c)} (no letter content)")
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
                marker = '*' if blank else ''
                print(f"  {pos_notation(r, c)}: {to_display(internal).upper()}{marker}")

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
        best_tokens = candidates[0][0]
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
