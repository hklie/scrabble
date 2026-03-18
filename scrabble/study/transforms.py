"""transforms.py — Word transformation tools for Spanish Scrabble.

One-letter change, insert-letter, and remove-letter operations against the
FISE2 lexicon trie. Each function returns a list of result dicts.

Usage as standalone CLI:
    python -m study.transforms PALABRA
    python -m study.transforms --insert PALABRA
    python -m study.transforms --remove PALABRA
    python -m study.transforms --all PALABRA
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import ALL_TILES, INTERNAL_POINTS, DIGRAPHS
from preprocessing import tokenize_word, detokenize_word
from lexicon import load_lexicon_trie, _word_in_trie


def one_letter_changes(word, trie=None):
    """Return all valid words formed by changing one token in `word`.

    Each result: {'word': str, 'position': int, 'original': str,
                  'replacement': str, 'value': int}
    """
    if trie is None:
        trie = load_lexicon_trie()
    tokens = tokenize_word(word)
    results = []
    for pos in range(len(tokens)):
        original_token = tokens[pos]
        for tile in ALL_TILES:
            if tile == original_token:
                continue
            candidate = tokens[:pos] + [tile] + tokens[pos + 1:]
            if _word_in_trie(trie, candidate):
                new_word = detokenize_word(candidate)
                value = sum(INTERNAL_POINTS.get(t, 0) for t in candidate)
                results.append({
                    'word': new_word,
                    'position': pos,
                    'original': DIGRAPHS.get(original_token, original_token),
                    'replacement': DIGRAPHS.get(tile, tile),
                    'value': value,
                })
    return results


def insert_letter(word, trie=None):
    """Return all valid words formed by inserting one token anywhere in `word`.

    Each result: {'word': str, 'position': int, 'inserted': str, 'value': int}
    """
    if trie is None:
        trie = load_lexicon_trie()
    tokens = tokenize_word(word)
    results = []
    for pos in range(len(tokens) + 1):
        for tile in ALL_TILES:
            candidate = tokens[:pos] + [tile] + tokens[pos:]
            if _word_in_trie(trie, candidate):
                new_word = detokenize_word(candidate)
                value = sum(INTERNAL_POINTS.get(t, 0) for t in candidate)
                results.append({
                    'word': new_word,
                    'position': pos,
                    'inserted': DIGRAPHS.get(tile, tile),
                    'value': value,
                })
    return results


def remove_letter(word, trie=None):
    """Return all valid words formed by removing one token from `word`.

    Each result: {'word': str, 'position': int, 'removed': str, 'value': int}
    """
    if trie is None:
        trie = load_lexicon_trie()
    tokens = tokenize_word(word)
    if len(tokens) < 2:
        return []
    results = []
    for pos in range(len(tokens)):
        removed_token = tokens[pos]
        candidate = tokens[:pos] + tokens[pos + 1:]
        if _word_in_trie(trie, candidate):
            new_word = detokenize_word(candidate)
            value = sum(INTERNAL_POINTS.get(t, 0) for t in candidate)
            results.append({
                'word': new_word,
                'position': pos,
                'removed': DIGRAPHS.get(removed_token, removed_token),
                'value': value,
            })
    return results


def _display_token(token):
    """Convert internal token to display form (uppercase)."""
    return DIGRAPHS.get(token, token).upper()


def _print_changes(word, results):
    """Pretty-print one-letter change results grouped by position."""
    tokens = tokenize_word(word)
    if not results:
        print(f"  No one-letter changes found for {word.upper()}.")
        return
    print(f"\n  One-letter changes for {word.upper()} ({len(results)} results):\n")
    by_pos = {}
    for r in results:
        by_pos.setdefault(r['position'], []).append(r)
    for pos in sorted(by_pos):
        orig = _display_token(tokens[pos])
        items = by_pos[pos]
        print(f"  Position {pos + 1} ({orig}):")
        for r in sorted(items, key=lambda x: x['word']):
            print(f"    {r['replacement'].upper():4s} → {r['word'].upper():20s}  ({r['value']} pts)")
        print()


def _print_insertions(word, results):
    """Pretty-print insertion results grouped by position."""
    if not results:
        print(f"  No insertions found for {word.upper()}.")
        return
    print(f"\n  Insertions for {word.upper()} ({len(results)} results):\n")
    by_pos = {}
    for r in results:
        by_pos.setdefault(r['position'], []).append(r)
    tokens = tokenize_word(word)
    for pos in sorted(by_pos):
        if pos == 0:
            label = "Before first letter"
        elif pos == len(tokens):
            label = "After last letter"
        else:
            label = f"Between pos {pos} and {pos + 1}"
        items = by_pos[pos]
        print(f"  {label}:")
        for r in sorted(items, key=lambda x: x['word']):
            print(f"    +{r['inserted'].upper():4s} → {r['word'].upper():20s}  ({r['value']} pts)")
        print()


def _print_removals(word, results):
    """Pretty-print removal results."""
    tokens = tokenize_word(word)
    if not results:
        print(f"  No removals found for {word.upper()}.")
        return
    print(f"\n  Removals for {word.upper()} ({len(results)} results):\n")
    for r in sorted(results, key=lambda x: x['position']):
        removed = _display_token(tokens[r['position']])
        print(f"    -{removed:4s} (pos {r['position'] + 1}) → {r['word'].upper():20s}  ({r['value']} pts)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Word transformation tools for Spanish Scrabble")
    parser.add_argument("word", help="Word to transform")
    parser.add_argument("--insert", action="store_true",
                        help="Show insertions (add one letter)")
    parser.add_argument("--remove", action="store_true",
                        help="Show removals (remove one letter)")
    parser.add_argument("--all", action="store_true",
                        help="Show all transformations")
    args = parser.parse_args()

    print("  Loading lexicon...", end=" ", flush=True)
    trie = load_lexicon_trie()
    print("done.")

    word = args.word.strip().lower()
    tokens = tokenize_word(word)
    value = sum(INTERNAL_POINTS.get(t, 0) for t in tokens)
    print(f"\n  {word.upper()}  ({len(tokens)} letters, {value} pts)")

    show_all = args.all or (not args.insert and not args.remove)

    if show_all or (not args.insert and not args.remove):
        changes = one_letter_changes(word, trie)
        _print_changes(word, changes)

    if args.insert or args.all:
        insertions = insert_letter(word, trie)
        _print_insertions(word, insertions)

    if args.remove or args.all:
        removals = remove_letter(word, trie)
        _print_removals(word, removals)


if __name__ == "__main__":
    main()
