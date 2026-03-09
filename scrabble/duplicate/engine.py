"""engine.py — Core game engine for Duplicate Scrabble."""

import os
import sys
import random

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from analyze_board import (
    build_trie, find_best_moves, Move, to_internal, to_display,
    row_letter, col_number,
)
from autoplay_scrabble import (
    create_bag, create_empty_board, apply_move, remove_from_rack,
    count_rack_composition, VOWELS, RACK_SIZE,
)
from config import LEXICON_FISE2


def draw_tiles_constrained(bag, rack, min_vowels, min_consonants):
    """Draw tiles from bag to fill rack to RACK_SIZE with parameterized constraints.

    Same algorithm as autoplay_scrabble.draw_tiles() but with configurable
    min_vowels and min_consonants instead of hardcoded constants.
    """
    need = RACK_SIZE - len(rack)
    if need <= 0 or not bag:
        return rack

    rack_vowels, rack_consonants, rack_blanks = count_rack_composition(rack)

    # Compute deficit, allocating existing blanks to reduce it
    vowel_deficit = max(0, min_vowels - rack_vowels)
    consonant_deficit = max(0, min_consonants - rack_consonants)
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

    final_rack = rack + drawn_tiles
    # Verify constraint is met (blanks count toward either)
    v, c, b = count_rack_composition(final_rack)
    if v + b < min_vowels or c + b < min_consonants:
        print(f"  WARNING: rack constraint not met! rack={final_rack} "
              f"v={v} c={c} b={b} min_v={min_vowels} min_c={min_consonants}")

    return final_rack


def generate_all_moves(board, rack, trie):
    """Return all valid moves sorted by score descending."""
    return find_best_moves(board, rack, trie, top_n=99999)


def moves_to_dataframe(moves):
    """Convert list of Move objects to a pandas DataFrame.

    Columns: rank, word, position, direction, dir_arrow, score,
             _word_tokens, _row, _col
    """
    rows = []
    for i, m in enumerate(moves, 1):
        rows.append({
            'rank': i,
            'word': m.word_display,
            'position': m.pos_str,
            'direction': m.direction,
            'dir_arrow': m.dir_arrow,
            'score': m.score,
            '_word_tokens': m.word_tokens,
            '_row': m.row,
            '_col': m.col,
            '_new_tile_positions': m.new_tile_positions,
        })
    return pd.DataFrame(rows)


def _get_blank_offsets_from_input(play_word, word_display):
    """Return set of character offsets where the player used lowercase (blank).

    play_word preserves case (e.g. 'CORTEs'), word_display is all-upper (e.g. 'CORTES').
    We walk both strings in sync, handling digraphs in word_display (CH, LL, RR).
    Returns a set of offsets into word_display, or None if letters don't match.
    """
    digraphs = {'CH', 'LL', 'RR'}
    blank_offsets = set()
    pi = 0  # index into play_word
    di = 0  # index into word_display

    while di < len(word_display) and pi < len(play_word):
        # Check if word_display has a digraph at this position
        dg = word_display[di:di+2]
        if dg in digraphs:
            # Player should have typed 2 chars for this digraph
            if pi + 1 >= len(play_word):
                return None
            p_chunk = play_word[pi:pi+2]
            if p_chunk.upper() != dg:
                return None
            # Both chars of digraph must be same case (both lower = blank)
            if p_chunk[0].islower() and p_chunk[1].islower():
                blank_offsets.add(di)
            elif p_chunk[0].isupper() and p_chunk[1].isupper():
                pass  # regular tile
            else:
                return None  # mixed case within digraph is invalid
            di += 2
            pi += 2
        else:
            p_ch = play_word[pi]
            if p_ch.upper() != word_display[di]:
                return None
            if p_ch.islower():
                blank_offsets.add(di)
            di += 1
            pi += 1

    if di != len(word_display) or pi != len(play_word):
        return None
    return blank_offsets


def _get_blank_offsets_from_move(new_tile_positions, word_tokens, direction):
    """Return set of character-offsets (into word_display) where blanks are placed."""
    blank_offsets_in_tokens = set()
    for offset, tile, is_blank in new_tile_positions:
        if is_blank:
            blank_offsets_in_tokens.add(offset)

    # Convert token offsets to display-string offsets
    display_offset = 0
    blank_display_offsets = set()
    for i, token in enumerate(word_tokens):
        display_char = to_display(token).upper()
        if i in blank_offsets_in_tokens:
            blank_display_offsets.add(display_offset)
        display_offset += len(display_char)

    return blank_display_offsets


def validate_play(play_word, play_position, moves_df):
    """Validate a player's play against the moves DataFrame.

    play_word preserves case: uppercase = regular tile, lowercase = blank.
    E.g. 'CORTEs' means S is played with a blank.
    Returns (True, score) or (False, 0).
    """
    if moves_df.empty:
        return False, 0

    word_upper = play_word.upper()
    pos_upper = play_position.upper()

    # Find all moves matching word + position (ignoring case)
    candidates = moves_df[
        (moves_df['word'].str.upper() == word_upper) &
        (moves_df['position'].str.upper() == pos_upper)
    ]

    if candidates.empty:
        return False, 0

    # Check if player used any blanks
    has_lowercase = any(c.islower() for c in play_word if c.isalpha())

    if not has_lowercase:
        # No blanks specified — accept if any non-blank variant exists
        for _, row in candidates.iterrows():
            move_blanks = _get_blank_offsets_from_move(
                row['_new_tile_positions'], row['_word_tokens'], row['direction'])
            if not move_blanks:
                return True, int(row['score'])
        # All candidates use blanks but player didn't specify — accept best score
        return True, int(candidates.iloc[0]['score'])

    # Player specified blanks — find matching move
    player_blanks = _get_blank_offsets_from_input(play_word, word_upper)
    if player_blanks is None:
        return False, 0

    for _, row in candidates.iterrows():
        move_blanks = _get_blank_offsets_from_move(
            row['_new_tile_positions'], row['_word_tokens'], row['direction'])
        if player_blanks == move_blanks:
            return True, int(row['score'])

    return False, 0


class GameState:
    """Manages the state of a Duplicate Scrabble game."""

    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)

        print("Loading lexicon trie...")
        self.trie = build_trie(LEXICON_FISE2)

        self.board = create_empty_board()
        self.bag = create_bag()
        self.round_num = 0
        self.rack = []
        self.moves_df = pd.DataFrame()
        self.best_move = None
        self.master_scores = []
        self._all_moves = []

    def start_round(self, min_v, min_c):
        """Start a new round: draw rack, generate all moves.

        In Duplicate, rack starts empty each round (no carry-over).
        Returns False if game should end (bag has fewer than RACK_SIZE tiles).
        """
        if len(self.bag) < RACK_SIZE:
            return False

        self.round_num += 1
        self.rack = []
        self.rack = draw_tiles_constrained(self.bag, self.rack, min_v, min_c)

        if not self.rack:
            return False

        self._all_moves = generate_all_moves(self.board, self.rack, self.trie)
        self.moves_df = moves_to_dataframe(self._all_moves)
        self.best_move = self._all_moves[0] if self._all_moves else None

        return True

    def apply_master_move(self):
        """Place the best move on the board and track the master score.

        In Duplicate, only the tiles used by the master move are consumed.
        Unused rack tiles are returned to the bag.
        """
        if self.best_move is None:
            self.master_scores.append(0)
            # No move — return all rack tiles to bag
            self.bag.extend(self.rack)
            self.rack = []
            return

        tiles_used = apply_move(self.board, self.best_move)
        self.master_scores.append(self.best_move.score)

        # Return unused rack tiles to bag
        leftover = remove_from_rack(self.rack, tiles_used)
        self.bag.extend(leftover)
        random.shuffle(self.bag)
        self.rack = []
