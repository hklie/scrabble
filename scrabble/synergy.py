"""synergy.py - Rack Leave Synergy Values

Computes synergy scores for Scrabble rack leaves: the letter combinations
remaining on a player's rack after playing a word. High-synergy leaves mean
the remaining letters co-occur frequently in valid words.

Output: Data/synergy.csv with columns: combination, length, synergy (0-100)
"""

import csv
import os
from collections import Counter
from itertools import combinations_with_replacement

from config import SCRABBLE_TILES, CLEAN_NO_VERBS_FILE, BASE_PATH, DIGRAPH_MAP
from preprocessing import tokenize_word

# Map display token names (like 'ch') to internal codes (like '1')
DISPLAY_TO_INTERNAL = {k: DIGRAPH_MAP.get(k, k) for k in SCRABBLE_TILES}
INTERNAL_TO_DISPLAY = {v: k for k, v in DISPLAY_TO_INTERNAL.items()}
INTERNAL_TOKENS = sorted(DISPLAY_TO_INTERNAL.values())
TILES_INTERNAL = {DISPLAY_TO_INTERNAL[k]: v for k, v in SCRABBLE_TILES.items()}

OUTPUT_FILE = os.path.join(BASE_PATH, "synergy.csv")


def load_corpus():
    """Load the corpus and return a list of Counters (using internal token codes)."""
    counters = []
    with open(CLEAN_NO_VERBS_FILE, encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            if word:
                counters.append(Counter(tokenize_word(word)))
    return counters


def build_inverted_index(corpus_counters):
    """Build inverted index: (token, min_count) -> set of word indices.

    For each word, if token T appears N times, the word index is added
    to keys (T,1), (T,2), ..., (T,N). This allows efficient subset
    matching via set intersection.
    """
    index = {}
    for i, counter in enumerate(corpus_counters):
        for token, count in counter.items():
            for c in range(1, count + 1):
                key = (token, c)
                if key not in index:
                    index[key] = set()
                index[key].add(i)
    return index


def count_subset_matches(combo_counter, inv_index):
    """Count corpus words containing all tokens in combo_counter as a submultiset."""
    sets = []
    for token, count in combo_counter.items():
        key = (token, count)
        if key not in inv_index:
            return 0
        sets.append(inv_index[key])
    if not sets:
        return 0
    # Intersect from smallest to largest for efficiency
    sets.sort(key=len)
    result = sets[0]
    for s in sets[1:]:
        result = result & s
        if not result:
            return 0
    return len(result)


def is_valid_combo(combo_counter):
    """Check if a combination respects SCRABBLE_TILES limits."""
    for token, count in combo_counter.items():
        if token not in TILES_INTERNAL or count > TILES_INTERNAL[token]:
            return False
    return True


def compute_synergies_for_size(token_pool, size, inv_index):
    """Generate all multiset combinations of given size from token_pool, score valid ones."""
    results = []
    for combo in combinations_with_replacement(token_pool, size):
        combo_counter = Counter(combo)
        if not is_valid_combo(combo_counter):
            continue
        score = count_subset_matches(combo_counter, inv_index)
        results.append((combo, score))
    return results


def percentile_normalize(results):
    """Map raw scores to 0-100 integers by percentile rank within the group.

    Lowest count -> 0, highest -> 100, linear interpolation in between.
    """
    if not results:
        return []
    scores = sorted(set(r[1] for r in results))
    if len(scores) == 1:
        return [(combo, 100) for combo, _ in results]
    rank_map = {s: i for i, s in enumerate(scores)}
    max_rank = len(scores) - 1
    return [(combo, round(rank_map[score] / max_rank * 100))
            for combo, score in results]


def format_combo(combo_tokens):
    """Format internal token codes as display string (e.g., 'a+ch+s')."""
    display = sorted(INTERNAL_TO_DISPLAY.get(t, t) for t in combo_tokens)
    return '+'.join(display)


def write_csv(all_scored):
    """Write results to synergy.csv."""
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['combination', 'length', 'synergy'])
        for combo, synergy in sorted(all_scored, key=lambda x: (-x[1], format_combo(x[0]))):
            writer.writerow([format_combo(combo), len(combo), synergy])


def compute_synergy():
    """Main orchestrator for synergy computation."""
    print("Loading corpus...")
    corpus_counters = load_corpus()
    print(f"  {len(corpus_counters)} words loaded.")

    print("Building inverted index...")
    inv_index = build_inverted_index(corpus_counters)
    print(f"  {len(inv_index)} index entries.")

    all_scored = []

    # Size 1: exhaustive over all 28 tokens, also determines top-10 pool
    print("Computing size-1 synergies...")
    size1_raw = compute_synergies_for_size(INTERNAL_TOKENS, 1, inv_index)
    top_tokens = [combo[0] for combo, _ in sorted(size1_raw, key=lambda x: -x[1])[:10]]
    print(f"  Top 10 tokens: {[INTERNAL_TO_DISPLAY.get(t, t) for t in top_tokens]}")
    all_scored.extend(percentile_normalize(size1_raw))
    print(f"  {len(size1_raw)} combinations scored.")

    # Size 2: exhaustive over all 28 tokens
    print("Computing size-2 synergies...")
    size2_raw = compute_synergies_for_size(INTERNAL_TOKENS, 2, inv_index)
    all_scored.extend(percentile_normalize(size2_raw))
    print(f"  {len(size2_raw)} combinations scored.")

    # Sizes 3-5: from top-10 token pool
    for size in (3, 4, 5):
        print(f"Computing size-{size} synergies (top-10 pool)...")
        raw = compute_synergies_for_size(top_tokens, size, inv_index)
        all_scored.extend(percentile_normalize(raw))
        print(f"  {len(raw)} combinations scored.")

    print(f"\nWriting {len(all_scored)} entries to {OUTPUT_FILE}...")
    write_csv(all_scored)
    print("Done.")


if __name__ == "__main__":
    compute_synergy()
