"""autoplay_scrabble.py — Automated solitaire Spanish Scrabble game.

Draws tiles from the bag, finds the highest-scoring play each round using the
Appel-Jacobson engine from analyze_board.py, and outputs a CSV summary.

Usage:
    python scrabble/autoplay_scrabble.py --seed 42
    python scrabble/autoplay_scrabble.py --seed 123 -o Data/game_123.csv
"""

import argparse
import csv
import os
import random
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from analyze_board import (
    build_trie, find_best_moves, Move, to_internal, to_display,
    tile_points, row_letter, col_number,
)
from config import (
    LEXICON_FISE2, SCRABBLE_TILES, TOTAL_BLANKS, ALL_TILES, BASE_PATH, DIGRAPHS,
    PREMIUM_SQUARES, INTERNAL_POINTS,
)

VOWELS = frozenset('aeiou')
RACK_SIZE = 7
MIN_VOWELS = 2
MIN_CONSONANTS = 2


def create_bag():
    """Build the full 100-tile bag in internal encoding, shuffled."""
    bag = []
    for tile_display, count in SCRABBLE_TILES.items():
        internal = to_internal(tile_display)
        bag.extend([internal] * count)
    bag.extend(['?'] * TOTAL_BLANKS)
    random.shuffle(bag)
    return bag


def create_empty_board():
    """Return 15x15 grid of None."""
    return [[None] * 15 for _ in range(15)]


def count_rack_composition(rack):
    """Count vowels, consonants, and blanks in rack."""
    v = sum(1 for t in rack if t in VOWELS)
    b = sum(1 for t in rack if t == '?')
    c = len(rack) - v - b
    return v, c, b


def draw_tiles(bag, rack):
    """Draw tiles from bag to fill rack to RACK_SIZE with vowel/consonant constraint.

    Ensures the final rack has at least MIN_VOWELS vowels and MIN_CONSONANTS
    consonants (blanks count toward either deficit). If the bag can't satisfy
    the constraint, draws whatever is available.
    """
    need = RACK_SIZE - len(rack)
    if need <= 0 or not bag:
        return rack

    rack_vowels, rack_consonants, rack_blanks = count_rack_composition(rack)

    # Compute deficit, allocating existing blanks to reduce it
    vowel_deficit = max(0, MIN_VOWELS - rack_vowels)
    consonant_deficit = max(0, MIN_CONSONANTS - rack_consonants)
    blanks_for_vowels = min(rack_blanks, vowel_deficit)
    remaining_blanks = rack_blanks - blanks_for_vowels
    blanks_for_consonants = min(remaining_blanks, consonant_deficit)
    vowel_deficit -= blanks_for_vowels
    consonant_deficit -= blanks_for_consonants

    # Separate bag into pools by index
    vowel_idx = [i for i, t in enumerate(bag) if t in VOWELS]
    consonant_idx = [i for i, t in enumerate(bag) if t not in VOWELS and t != '?']
    random.shuffle(vowel_idx)
    random.shuffle(consonant_idx)

    drawn = set()

    # Draw required vowels first
    to_draw = min(vowel_deficit, len(vowel_idx), need)
    for i in range(to_draw):
        drawn.add(vowel_idx[i])
    need -= to_draw

    # Draw required consonants
    to_draw = min(consonant_deficit, len(consonant_idx), need)
    for i in range(to_draw):
        drawn.add(consonant_idx[i])
    need -= to_draw

    # Fill remaining slots randomly from whatever is left in bag
    remaining = [i for i in range(len(bag)) if i not in drawn]
    random.shuffle(remaining)
    for i in range(min(need, len(remaining))):
        drawn.add(remaining[i])

    # Collect drawn tiles, remove from bag (reverse order preserves indices)
    drawn_tiles = [bag[i] for i in drawn]
    for i in sorted(drawn, reverse=True):
        bag.pop(i)

    return rack + drawn_tiles


def should_stop(bag, rack):
    """Check if the game should end.

    Stops when: bag+rack are empty, OR bag has no vowels/blanks left and
    rack doesn't have enough effective vowels (vowels + blanks >= MIN_VOWELS).
    """
    if not rack and not bag:
        return True
    rack_vowels, _, rack_blanks = count_rack_composition(rack)
    effective_vowels = rack_vowels + rack_blanks
    bag_vowels = sum(1 for t in bag if t in VOWELS)
    bag_blanks = sum(1 for t in bag if t == '?')
    if bag_vowels == 0 and bag_blanks == 0 and effective_vowels < MIN_VOWELS:
        return True
    return False


def apply_move(board, move):
    """Place move's tiles on the board. Return list of tiles consumed from rack."""
    tiles_used = []
    for offset, tile, is_blank in move.new_tile_positions:
        if move.direction == 'horizontal':
            r, c = move.row, move.col + offset
        else:
            r, c = move.row + offset, move.col
        board[r][c] = (tile, is_blank)
        tiles_used.append('?' if is_blank else tile)
    return tiles_used


def remove_from_rack(rack, tiles_used):
    """Remove each used tile from rack (one occurrence per tile)."""
    rack = list(rack)
    for tile in tiles_used:
        rack.remove(tile)
    return rack


def print_board_text(board):
    """ASCII 15x15 grid with row letters (A-O) and column numbers (1-15)."""
    print('    ' + ''.join(f'{i+1:>3d}' for i in range(15)))
    print('   ' + '-' * 46)
    for r in range(15):
        cells = []
        for c in range(15):
            if board[r][c] is None:
                cells.append('.')
            else:
                tile, is_blank = board[r][c]
                display = to_display(tile)
                cells.append(display.lower() if is_blank else display.upper())
        print(f' {row_letter(r)} |' + ''.join(f'{cell:>3s}' for cell in cells))


def play_game(seed=None):
    """Main game loop. Returns (records, final_score)."""
    if seed is not None:
        random.seed(seed)

    print("Loading lexicon trie...")
    trie = build_trie(LEXICON_FISE2)

    board = create_empty_board()
    bag = create_bag()
    rack = []
    records = []
    total_score = 0
    round_num = 0

    while True:
        rack = draw_tiles(bag, rack)

        if should_stop(bag, rack):
            print(f"\nGame over: stopping condition met (bag: {len(bag)}, rack: {len(rack)})")
            break

        if not rack:
            print("\nGame over: no tiles available")
            break

        round_num += 1
        rack_display = ' '.join(to_display(t).upper() if t != '?' else '?' for t in rack)
        print(f"\n{'='*60}")
        print(f"Round {round_num} | Rack: {rack_display} | Bag: {len(bag)} tiles")
        print(f"{'='*60}")

        moves = find_best_moves(board, rack, trie, top_n=10)

        if not moves:
            print("No valid moves found. Game over.")
            break

        for i, m in enumerate(moves, 1):
            print(f"  {i:2d}. {m.notation} — {m.score} pts")

        best = moves[0]
        tiles_used = apply_move(board, best)
        rack = remove_from_rack(rack, tiles_used)
        total_score += best.score

        print(f"\n  Played: {best.word_display} at {best.pos_str} {best.dir_arrow}"
              f" for {best.score} pts")
        print(f"  Total score: {total_score}")
        print()
        print_board_text(board)

        for i, m in enumerate(moves, 1):
            records.append({
                'round': round_num,
                'rack': rack_display,
                'rank': i,
                'position': m.pos_str,
                'direction': m.dir_arrow,
                'word': m.word_display,
                'score': m.score,
                'accumulated_score': total_score,
            })

    print(f"\n{'='*60}")
    print(f"FINAL SCORE: {total_score} points in {round_num} rounds")
    print(f"{'='*60}")

    return records, total_score, board


def render_board_image(board, filepath, score=0, rounds=0, seed=None):
    """Render the board as a PNG image."""
    from PIL import Image, ImageDraw, ImageFont

    CELL = 52
    MARGIN = 36
    LABEL = 28
    BOARD_PX = 15 * CELL
    W = LABEL + MARGIN + BOARD_PX + MARGIN
    H = LABEL + MARGIN + BOARD_PX + MARGIN + 40  # extra for footer

    # Colors
    BG = (1, 100, 56)            # dark green felt
    GRID = (0, 70, 40)           # grid lines
    EMPTY = (15, 120, 68)        # empty cell
    TW_CLR = (200, 40, 40)       # triple word — red
    DW_CLR = (210, 130, 160)     # double word — pink
    TL_CLR = (40, 90, 180)       # triple letter — blue
    DL_CLR = (110, 175, 220)     # double letter — light blue
    TILE_CLR = (240, 225, 190)   # tile background — cream
    BLANK_CLR = (220, 200, 165)  # blank tile — darker cream
    STAR_CLR = (210, 130, 160)   # center star
    TEXT = (30, 30, 30)
    LABEL_CLR = (200, 210, 200)
    PREMIUM_TEXT = (255, 255, 255, 160)

    PREM_COLORS = {'TW': TW_CLR, 'DW': DW_CLR, 'TL': TL_CLR, 'DL': DL_CLR}

    img = Image.new('RGB', (W, H), BG)
    draw = ImageDraw.Draw(img)

    try:
        font_tile = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 22)
        font_pts = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 10)
        font_label = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
        font_prem = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 9)
        font_footer = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 16)
    except OSError:
        font_tile = ImageFont.load_default()
        font_pts = font_label = font_prem = font_footer = font_tile

    ox = LABEL + MARGIN  # board origin x
    oy = LABEL + MARGIN  # board origin y

    # Column labels
    for c in range(15):
        cx = ox + c * CELL + CELL // 2
        txt = str(c + 1)
        bb = draw.textbbox((0, 0), txt, font=font_label)
        tw = bb[2] - bb[0]
        draw.text((cx - tw // 2, oy - 22), txt, fill=LABEL_CLR, font=font_label)

    # Row labels
    for r in range(15):
        cy = oy + r * CELL + CELL // 2
        txt = chr(ord('A') + r)
        bb = draw.textbbox((0, 0), txt, font=font_label)
        th = bb[3] - bb[1]
        draw.text((ox - 20, cy - th // 2), txt, fill=LABEL_CLR, font=font_label)

    # Draw cells
    for r in range(15):
        for c in range(15):
            x1 = ox + c * CELL
            y1 = oy + r * CELL
            x2 = x1 + CELL - 1
            y2 = y1 + CELL - 1

            if board[r][c] is not None:
                # Tile
                tile, is_blank = board[r][c]
                bg = BLANK_CLR if is_blank else TILE_CLR
                draw.rounded_rectangle([x1 + 1, y1 + 1, x2 - 1, y2 - 1],
                                       radius=4, fill=bg)
                # Letter
                display = to_display(tile).upper()
                if is_blank:
                    display = to_display(tile).lower()
                bb = draw.textbbox((0, 0), display, font=font_tile)
                tw_t = bb[2] - bb[0]
                th_t = bb[3] - bb[1]
                lx = x1 + (CELL - tw_t) // 2
                ly = y1 + (CELL - th_t) // 2 - 4
                draw.text((lx, ly), display, fill=TEXT, font=font_tile)
                # Point value subscript
                pts = 0 if is_blank else INTERNAL_POINTS.get(tile, 0)
                pts_txt = str(pts)
                draw.text((x2 - 13, y2 - 14), pts_txt, fill=(100, 80, 60),
                          font=font_pts)
            else:
                # Empty cell — premium or plain
                prem = PREMIUM_SQUARES.get((r, c))
                if prem:
                    fill = PREM_COLORS[prem]
                elif r == 7 and c == 7:
                    fill = STAR_CLR
                else:
                    fill = EMPTY
                draw.rounded_rectangle([x1 + 1, y1 + 1, x2 - 1, y2 - 1],
                                       radius=3, fill=fill)
                # Premium label or star
                if prem:
                    bb = draw.textbbox((0, 0), prem, font=font_prem)
                    tw_p = bb[2] - bb[0]
                    th_p = bb[3] - bb[1]
                    draw.text((x1 + (CELL - tw_p) // 2,
                               y1 + (CELL - th_p) // 2),
                              prem, fill=(255, 255, 255), font=font_prem)
                elif r == 7 and c == 7:
                    star = '\u2605'
                    bb = draw.textbbox((0, 0), star, font=font_tile)
                    tw_s = bb[2] - bb[0]
                    th_s = bb[3] - bb[1]
                    draw.text((x1 + (CELL - tw_s) // 2,
                               y1 + (CELL - th_s) // 2 - 2),
                              star, fill=(255, 255, 255), font=font_tile)

    # Grid lines
    for i in range(16):
        x = ox + i * CELL
        draw.line([(x, oy), (x, oy + BOARD_PX)], fill=GRID, width=1)
        y = oy + i * CELL
        draw.line([(ox, y), (ox + BOARD_PX, y)], fill=GRID, width=1)

    # Outer border
    draw.rectangle([ox - 1, oy - 1, ox + BOARD_PX, oy + BOARD_PX],
                   outline=(0, 50, 30), width=2)

    # Footer
    parts = []
    if seed is not None:
        parts.append(f"Seed: {seed}")
    parts.append(f"{rounds} rounds")
    parts.append(f"{score} pts")
    footer = '  |  '.join(parts)
    bb = draw.textbbox((0, 0), footer, font=font_footer)
    fw = bb[2] - bb[0]
    draw.text(((W - fw) // 2, oy + BOARD_PX + 12), footer,
              fill=LABEL_CLR, font=font_footer)

    img.save(filepath)
    return filepath


def write_csv(records, filepath):
    """Write game records to CSV."""
    dirname = os.path.dirname(filepath)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'round', 'rack', 'rank', 'position', 'direction',
            'word', 'score', 'accumulated_score',
        ])
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(
        description='Automated solitaire Spanish Scrabble')
    parser.add_argument('--seed', '-s', type=int,
                        help='Random seed for reproducibility')
    parser.add_argument('--output', '-o',
                        help='CSV output path (default: Data/autoplay_TIMESTAMP.csv)')
    args = parser.parse_args()

    if args.output:
        csv_path = args.output
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = os.path.join(BASE_PATH, f'autoplay_{timestamp}.csv')

    records, final_score, board = play_game(seed=args.seed)

    if records:
        write_csv(records, csv_path)
        print(f"\nCSV written to: {csv_path}")

        img_path = csv_path.rsplit('.', 1)[0] + '.png'
        round_count = records[-1]['round']
        render_board_image(board, img_path, score=final_score,
                           rounds=round_count, seed=args.seed)
        print(f"Board image saved to: {img_path}")
    else:
        print("\nNo moves were made, no CSV written.")


if __name__ == '__main__':
    main()
