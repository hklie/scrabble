"""vowel_patterns.py — Find 7-token words with specific vowel/consonant patterns.

Categories (union of all):
  1. Only 2 vowels (5 consonants)
  2. Only 2 consonants (5 vowels)
  3. Exactly 3 consonants with at least 2 duplicated
  4. 3 or more i's
  5. 3 or more u's
  6. 3 or more o's
  7. All vowels drawn only from {i, o, u} (no a, no e)

Usage:
    python scrabble/study/vowel_patterns.py
"""

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import CLEAN_NO_VERBS_FILE
from preprocessing import tokenize_word, detokenize_word

VOWELS = {"a", "e", "i", "o", "u", "á", "é", "í", "ó", "ú", "ü"}
TARGET_LEN = 7


DIGRAPH_CODES = {"1", "2", "3", "4"}  # ch, ll, rr, ñ — all consonants


def is_vowel(token):
    return token in VOWELS


def is_consonant(token):
    return token in DIGRAPH_CODES or (token.isalpha() and token not in VOWELS)


def classify(tokens):
    """Return set of category labels the word belongs to."""
    vowels = [t for t in tokens if is_vowel(t)]
    consonants = [t for t in tokens if is_consonant(t)]
    n_v = len(vowels)
    n_c = len(consonants)
    cats = set()

    if n_v == 2:
        cats.add("2_vowels")
    if n_c == 2:
        cats.add("2_consonants")
    if n_c == 3:
        c_counts = Counter(consonants)
        if any(v >= 2 for v in c_counts.values()):
            cats.add("3_cons_dup")
    if sum(1 for v in vowels if v in {"i", "í"}) >= 3:
        cats.add("3_i")
    if sum(1 for v in vowels if v in {"u", "ú", "ü"}) >= 3:
        cats.add("3_u")
    if sum(1 for v in vowels if v in {"o", "ó"}) >= 3:
        cats.add("3_o")
    if n_v >= 1 and all(v in {"i", "í", "o", "ó", "u", "ú", "ü"} for v in vowels):
        cats.add("only_iou")

    return cats


def main():
    with open(CLEAN_NO_VERBS_FILE, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]

    by_cat = {
        "2_vowels": [],
        "2_consonants": [],
        "3_cons_dup": [],
        "3_i": [],
        "3_u": [],
        "3_o": [],
        "only_iou": [],
    }
    all_matches = set()

    for word in words:
        tokens = tokenize_word(word)
        if len(tokens) != TARGET_LEN:
            continue
        cats = classify(tokens)
        if cats:
            display = detokenize_word(tokens)
            all_matches.add(display)
            for c in cats:
                by_cat[c].append(display)

    base_dir = os.path.dirname(CLEAN_NO_VERBS_FILE)

    # Per-category files
    for label, matched in by_cat.items():
        matched_sorted = sorted(set(matched))
        path = os.path.join(base_dir, f"vowel_pattern_{label}_{TARGET_LEN}.txt")
        with open(path, "w", encoding="utf-8") as f:
            for w in matched_sorted:
                f.write(w + "\n")
        print(f"{label}: {len(matched_sorted)} words -> {path}")

    # Combined file
    combined_path = os.path.join(base_dir, f"vowel_pattern_all_{TARGET_LEN}.txt")
    with open(combined_path, "w", encoding="utf-8") as f:
        for w in sorted(all_matches):
            f.write(w + "\n")
    print(f"\nTotal unique: {len(all_matches)} words -> {combined_path}")


if __name__ == "__main__":
    main()
