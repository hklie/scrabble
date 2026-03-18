"""lexicon.py — Shared FISE2 trie utilities for Spanish Scrabble.

Provides word validation, trie building, and lookup functions used by
analyze_board.py, quiz modes, and transformation tools.
"""

import os
import pickle

from config import LEXICON_FISE2, ALL_TILES, INTERNAL_POINTS
from preprocessing import tokenize_word, detokenize_word


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


# ── High-level API ──

_TRIE_CACHE = None


def load_lexicon_trie():
    """Load or build the FISE2 trie (pickle-cached). Cached in memory after first call."""
    global _TRIE_CACHE
    if _TRIE_CACHE is not None:
        return _TRIE_CACHE
    _TRIE_CACHE = build_trie(LEXICON_FISE2)
    return _TRIE_CACHE


def is_valid_word(word, trie=None):
    """Check if a word exists in the FISE2 lexicon. Handles digraphs transparently.

    Args:
        word: A Spanish word (e.g. "churro", "año").
        trie: Optional pre-loaded trie. If None, loads the default FISE2 trie.

    Returns:
        True if the word is valid in the lexicon.
    """
    if trie is None:
        trie = load_lexicon_trie()
    tokens = tokenize_word(word)
    return _word_in_trie(trie, tokens)


def word_value(word):
    """Compute the point value of a word."""
    tokens = tokenize_word(word)
    return sum(INTERNAL_POINTS.get(t, 0) for t in tokens)
