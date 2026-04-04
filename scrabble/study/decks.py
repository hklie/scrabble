"""decks.py — Deck generation from word_analysis.csv.

Loads the CSV, parses rows into card dicts, and provides filtering/grouping
to produce study decks for the quiz system.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (BASE_PATH, DIGRAPH_MAP, TIER_1, TIER_2, TIER_3, TIER_4,
                    SCRABBLE_POINTS, INTERNAL_POINTS)
from preprocessing import tokenize_word, detokenize_word

WORD_ANALYSIS_CSV = os.path.join(BASE_PATH, "word_analysis.csv")
VERBS_CSV = os.path.join(BASE_PATH, "verbs.csv")

PERCENTILE_RANK = {"Top10": 6, "P90": 5, "P75": 4, "P50": 3, "P25": 2, "P10": 1}

# All 28 tile types in display form (used as hook column names in CSV)
HOOK_TILES = list("abcdefghijlmnopqrstuvxyz") + ["ch", "ll", "rr", "ñ"]

# Tier letter sets converted to internal token form
_TIER_TOKENS = {}
for _tier_num, _tier_set in [(1, TIER_1), (2, TIER_2), (3, TIER_3), (4, TIER_4)]:
    tokens = set()
    for letter in _tier_set:
        if letter in DIGRAPH_MAP:
            tokens.add(DIGRAPH_MAP[letter])
        else:
            tokens.add(letter)
    _TIER_TOKENS[_tier_num] = tokens

# Cache
_ALL_CARDS = None
_ALL_VERBS = None


def load_word_analysis(path=WORD_ANALYSIS_CSV):
    """Load CSV into list of card dicts. Cached after first call."""
    global _ALL_CARDS
    if _ALL_CARDS is not None:
        return _ALL_CARDS

    cards = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = row["word"]
            tokens = tokenize_word(word)
            front_hooks, back_hooks = _parse_hooks(row)
            cards.append({
                "word": word,
                "tokens": tokens,
                "length": int(row.get("length", len(tokens))),
                "prefix": row.get("prefix", ""),
                "suffix": row.get("suffix", ""),
                "probability": float(row.get("probabilityx1000", 0)),
                "percentile": row.get("percentile", "P10"),
                "pattern": row.get("pattern", ""),
                "ending": row.get("ending", ""),
                "value": int(row.get("value", 0)),
                "anagrams": int(row.get("anagrams", 0)),
                "category": row.get("category", ""),
                "front_hooks": front_hooks,
                "back_hooks": back_hooks,
            })

    _ALL_CARDS = cards
    return _ALL_CARDS


def _parse_hooks(row):
    """Extract front_hooks and back_hooks lists from CSV row."""
    front = [t for t in HOOK_TILES if row.get(f"{t}-hook", "") == "yes"]
    back = [t for t in HOOK_TILES if row.get(f"hook-{t}", "") == "yes"]
    return front, back


# ── Verb loading ──

def load_verbs(path=VERBS_CSV):
    """Load verbs.csv into card dicts with computed fields."""
    global _ALL_VERBS
    if _ALL_VERBS is not None:
        return _ALL_VERBS

    cards = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = row["word"]
            tokens = tokenize_word(word)
            value = sum(INTERNAL_POINTS.get(t, 0) for t in tokens)
            cards.append({
                "word": word,
                "tokens": tokens,
                "length": int(row.get("length", len(tokens))),
                "verb_type": row.get("type", ""),
                "value": value,
                "ending": word[-1] if word else "",
                # Fields kept for compatibility with quiz modes
                "prefix": "",
                "suffix": "",
                "probability": 0.0,
                "percentile": "",
                "pattern": "",
                "anagrams": 0,
                "category": "verb",
                "front_hooks": [],
                "back_hooks": [],
            })

    _ALL_VERBS = cards
    return _ALL_VERBS


def filter_verbs(verbs, *, min_length=2, max_length=15,
                 beginning=None, verb_type=None, tier=None):
    """Filter verbs by length, beginning string, type, or consonant tier."""
    result = []
    for c in verbs:
        if not (min_length <= c["length"] <= max_length):
            continue
        if beginning and not c["word"].startswith(beginning):
            continue
        if verb_type and c["verb_type"] != verb_type:
            continue
        if tier is not None:
            tier_letters = _TIER_TOKENS.get(tier, set())
            if not any(t in tier_letters for t in c["tokens"]):
                continue
        result.append(c)
    result.sort(key=lambda c: c["word"])
    return result


def group_verbs_by_beginning(verbs, prefix_len=3, min_group=3):
    """Group verbs by their first N characters."""
    from collections import defaultdict
    groups = defaultdict(list)
    for c in verbs:
        if len(c["word"]) >= prefix_len:
            groups[c["word"][:prefix_len]].append(c)
    result = {k: v for k, v in groups.items() if len(v) >= min_group}
    return dict(sorted(result.items(), key=lambda x: -len(x[1])))


def group_verbs_by_type(verbs):
    """Group verbs by type (transitivo, intransitivo, etc.)."""
    from collections import defaultdict
    groups = defaultdict(list)
    for c in verbs:
        if c["verb_type"]:
            groups[c["verb_type"]].append(c)
    return dict(sorted(groups.items(), key=lambda x: -len(x[1])))


def filter_cards(cards, *, min_length=2, max_length=15,
                 min_percentile="P10", max_percentile="Top10",
                 endings=None, prefix=None, suffix=None,
                 pattern=None, tier=None,
                 min_anagrams=0, min_hooks=0,
                 min_value=0):
    """Apply filters, return matching cards sorted by probability desc."""
    min_p = PERCENTILE_RANK.get(min_percentile, 1)
    max_p = PERCENTILE_RANK.get(max_percentile, 6)

    result = []
    for c in cards:
        if not (min_length <= c["length"] <= max_length):
            continue
        cp = PERCENTILE_RANK.get(c["percentile"], 0)
        if not (min_p <= cp <= max_p):
            continue
        if endings and c["ending"] not in endings:
            continue
        if prefix and not c["prefix"].startswith(prefix):
            continue
        if suffix and not c["suffix"].endswith(suffix):
            continue
        if pattern and c["pattern"] != pattern:
            continue
        if tier is not None:
            tier_letters = _TIER_TOKENS.get(tier, set())
            if not any(t in tier_letters for t in c["tokens"]):
                continue
        if c["anagrams"] < min_anagrams:
            continue
        total_hooks = len(c["front_hooks"]) + len(c["back_hooks"])
        if total_hooks < min_hooks:
            continue
        if c["value"] < min_value:
            continue
        result.append(c)

    result.sort(key=lambda c: c["probability"], reverse=True)
    return result


# ── Preset decks ──

# Words (non-verbs) presets
WORD_PRESETS = {
    # By length
    "words-2":       "Palabras de 2 letras",
    "words-3":       "Palabras de 3 letras",
    "words-4":       "Palabras de 4 letras",
    "words-5":       "Palabras de 5 letras",
    # Vowel patterns (7-letter)
    "7L-2vowels":    "7 letras con solo 2 vocales",
    "7L-2cons":      "7 letras con solo 2 consonantes",
    # High probability
    "high-prob":     "Alta probabilidad (Top10, 4-8 letras)",
    # High scoring letters (tier 3+4)
    "scoring-5":     "5 letras con fichas de alto valor",
    "scoring-6":     "6 letras con fichas de alto valor",
    # 5-letter by ending
    "5L-end-d":      "5 letras terminadas en D",
    "5L-end-l":      "5 letras terminadas en L",
    "5L-end-n":      "5 letras terminadas en N",
    "5L-end-r":      "5 letras terminadas en R",
    "5L-end-z":      "5 letras terminadas en Z",
}

# Verb presets
VERB_PRESETS = {
    "verbs-3":  "Verbos de 3 letras",
    "verbs-4":  "Verbos de 4 letras",
    "verbs-5":  "Verbos de 5 letras",
    "verbs-6":  "Verbos de 6 letras",
    "verbs-7":  "Verbos de 7 letras",
    "verbs-8":  "Verbos de 8 letras",
}

_WORD_PRESET_FILTERS = {
    "words-2":      dict(min_length=2, max_length=2),
    "words-3":      dict(min_length=3, max_length=3),
    "words-4":      dict(min_length=4, max_length=4),
    "words-5":      dict(min_length=5, max_length=5),
    "7L-2vowels":   dict(min_length=7, max_length=7),  # post-filtered
    "7L-2cons":     dict(min_length=7, max_length=7),   # post-filtered
    "high-prob":    dict(min_length=4, max_length=8, min_percentile="Top10"),
    "scoring-5":    dict(min_length=5, max_length=5, min_value=12),
    "scoring-6":    dict(min_length=6, max_length=6, min_value=14),
    "5L-end-d":     dict(min_length=5, max_length=5, endings={"d"}),
    "5L-end-l":     dict(min_length=5, max_length=5, endings={"l"}),
    "5L-end-n":     dict(min_length=5, max_length=5, endings={"n"}),
    "5L-end-r":     dict(min_length=5, max_length=5, endings={"r"}),
    "5L-end-z":     dict(min_length=5, max_length=5, endings={"z"}),
}

VOWELS = {"a", "e", "i", "o", "u"}


def _count_vowels(tokens):
    return sum(1 for t in tokens if t in VOWELS)


def _count_consonants(tokens):
    return sum(1 for t in tokens if t not in VOWELS)


def apply_preset(cards, name):
    """Apply a named preset filter. Returns filtered card list."""
    kwargs = _WORD_PRESET_FILTERS.get(name, {})
    result = filter_cards(cards, **kwargs)

    # Post-filter for vowel/consonant count patterns
    if name == "7L-2vowels":
        result = [c for c in result if _count_vowels(c["tokens"]) == 2]
    elif name == "7L-2cons":
        result = [c for c in result if _count_consonants(c["tokens"]) == 2]

    return result


def apply_verb_preset(verbs, name):
    """Apply a verb preset filter by length."""
    length_map = {f"verbs-{n}": n for n in range(2, 10)}
    ln = length_map.get(name)
    if ln is not None:
        return filter_verbs(verbs, min_length=ln, max_length=ln)
    return verbs


def available_decks():
    """Return dict of all preset name → description."""
    result = {}
    result.update(WORD_PRESETS)
    result.update(VERB_PRESETS)
    return result



# ── Group-by helpers ──

def group_by_prefix(cards, min_group=5):
    """Group cards by prefix. Only groups with >= min_group words are kept.
    Returns dict of prefix → list of cards, sorted by group size desc."""
    from collections import defaultdict
    groups = defaultdict(list)
    for c in cards:
        if c["prefix"]:
            groups[c["prefix"]].append(c)
    # Filter small groups, sort by size descending
    result = {k: v for k, v in groups.items() if len(v) >= min_group}
    return dict(sorted(result.items(), key=lambda x: -len(x[1])))


def group_by_suffix(cards, min_group=5):
    """Group cards by suffix. Only groups with >= min_group words are kept."""
    from collections import defaultdict
    groups = defaultdict(list)
    for c in cards:
        if c["suffix"]:
            groups[c["suffix"]].append(c)
    result = {k: v for k, v in groups.items() if len(v) >= min_group}
    return dict(sorted(result.items(), key=lambda x: -len(x[1])))


def group_by_ending(cards, min_group=5):
    """Group cards by ending letter/digraph."""
    from collections import defaultdict
    groups = defaultdict(list)
    for c in cards:
        if c["ending"]:
            groups[c["ending"]].append(c)
    result = {k: v for k, v in groups.items() if len(v) >= min_group}
    return dict(sorted(result.items(), key=lambda x: -len(x[1])))
