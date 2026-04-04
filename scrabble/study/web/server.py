#!/usr/bin/env python3
"""Web server for the Spanish Scrabble study tool.

Usage:
    cd scrabble
    python -m study.web.server [--port 8080]
"""

import argparse
import json
import os
import random
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime

# Fix imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from lexicon import load_lexicon_trie, is_valid_word, word_value, _word_in_trie
from preprocessing import tokenize_word, detokenize_word
from config import (SCRABBLE_POINTS, DIGRAPHS, INTERNAL_POINTS, ALL_TILES,
                    EXTENSIVE_PREFIXES, EXTENSIVE_SUFIXES)
from study.decks import (load_word_analysis, load_verbs, apply_preset,
                         apply_verb_preset, available_decks,
                         WORD_PRESETS, VERB_PRESETS, filter_cards)
from study.srs import (load_progress, save_progress, update_card,
                        build_session, get_due_cards, CardState)
from study.transforms import one_letter_changes, insert_letter, remove_letter
from study.quiz import (_scramble, _blank_tokens, _parse_hook_input,
                         _hook_score, _quality_from_score, display_token)

# ── App setup ──

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

app = FastAPI(title="Lexicable — Aprende navegando en el universo de las palabras")

# ── Global state (loaded at startup) ──

trie = None
all_cards = []
all_verbs = []
card_lookup = {}
progress = {}
sessions = {}

# Session history: persisted to Data/history.json
HISTORY_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                            "Data", "history.json")
session_history = []


def _load_history():
    global session_history
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            session_history = json.load(f)
    else:
        session_history = []


def _save_history():
    tmp = HISTORY_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(session_history, f, ensure_ascii=False, indent=1)
    os.replace(tmp, HISTORY_FILE)


# Custom word lists: persisted to Data/custom_lists.json
CUSTOM_LISTS_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                  "Data", "custom_lists.json")
custom_lists = {}  # id -> {name, words: [str]}


def _load_custom_lists():
    global custom_lists
    if os.path.exists(CUSTOM_LISTS_FILE):
        with open(CUSTOM_LISTS_FILE, "r", encoding="utf-8") as f:
            custom_lists = json.load(f)
    else:
        custom_lists = {}


def _save_custom_lists():
    tmp = CUSTOM_LISTS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(custom_lists, f, ensure_ascii=False, indent=1)
    os.replace(tmp, CUSTOM_LISTS_FILE)


@app.on_event("startup")
def startup():
    global trie, all_cards, all_verbs, card_lookup, progress
    print("Loading lexicon trie...", end=" ", flush=True)
    trie = load_lexicon_trie()
    print("done.")
    print("Loading word database...", end=" ", flush=True)
    all_cards = load_word_analysis()
    all_verbs = load_verbs()
    card_lookup = {c["word"]: c for c in all_cards}
    for v in all_verbs:
        if v["word"] not in card_lookup:
            card_lookup[v["word"]] = v
    print(f"{len(all_cards)} words, {len(all_verbs)} verbs loaded.")
    progress = load_progress()
    print(f"Progress: {len(progress)} words tracked.")
    print("Building anagram index...", end=" ", flush=True)
    _build_anagram_index()
    print(f"{len(_anagram_index)} anagram groups.")
    print("Building extension index...", end=" ", flush=True)
    _build_extension_index()
    print(f"{len(_extension_index)} words with derivations.")
    _load_history()
    print(f"Session history: {len(session_history)} sessions.")
    _load_custom_lists()
    print(f"Custom lists: {len(custom_lists)} lists.")


# ── Pydantic models ──

class QuizStartRequest(BaseModel):
    mode: str
    deck: str = ""
    size: int = 20
    min_length: int | None = None
    max_length: int | None = None


class AnswerRequest(BaseModel):
    answer: str


class RateRequest(BaseModel):
    quality: int


class CustomListRequest(BaseModel):
    name: str
    words: str  # newline or comma separated


# ── Session state ──

@dataclass
class QuizSession:
    session_id: str
    mode: str
    cards: list
    current_index: int = 0
    results: list = field(default_factory=list)
    card_states: dict = field(default_factory=dict)  # index -> mode-specific data
    attempt: int = 0


# ── Helpers ──

# Accent normalization: strip accents except ü (preserved in lexicon)
_ACCENT_MAP = str.maketrans("áéíóú", "aeiou")


def _normalize_word(word):
    """Normalize accents for lexicon lookup. Preserves ü."""
    return word.lower().strip().translate(_ACCENT_MAP)

def _resolve_suffix(word, pattern):
    """Resolve a suffix pattern like 'enV' to the actual suffix 'ena' for a word."""
    if not pattern or not word:
        return pattern
    result = []
    wi = len(word) - 1  # walk backwards through the word
    for pi in range(len(pattern) - 1, -1, -1):
        ch = pattern[pi]
        if ch == "V":
            result.append(word[wi] if wi >= 0 else "?")
            wi -= 1
        elif ch == "C":
            result.append(word[wi] if wi >= 0 else "?")
            wi -= 1
        else:
            result.append(ch)
            wi -= 1
    return "".join(reversed(result))


def _resolve_prefix(word, pattern):
    """Resolve a prefix pattern like 'reC' to the actual prefix for a word."""
    if not pattern or not word:
        return pattern
    result = []
    for i, ch in enumerate(pattern):
        if ch in ("V", "C"):
            result.append(word[i] if i < len(word) else "?")
        else:
            result.append(ch)
    return "".join(result)


# Anagram index: sorted_tokens_key -> list of words
_anagram_index = {}


def _build_anagram_index():
    """Build index for fast anagram lookup."""
    global _anagram_index
    for c in all_cards:
        key = tuple(sorted(c["tokens"]))
        _anagram_index.setdefault(key, []).append(c["word"])
    for v in all_verbs:
        key = tuple(sorted(v["tokens"]))
        _anagram_index.setdefault(key, []).append(v["word"])


# Word extension index: word -> list of longer words that start with it
_extension_index = {}


def _build_extension_index():
    """Build index for word family/derivation lookup using sorted list + bisect."""
    global _extension_index
    import bisect
    all_words_set = set(c["word"] for c in all_cards)
    all_words_set.update(v["word"] for v in all_verbs)
    sorted_words = sorted(all_words_set)

    for word in sorted_words:
        # Use bisect to find the range of words starting with this word
        lo = bisect.bisect_left(sorted_words, word)
        # Upper bound: word with last char incremented
        hi_key = word[:-1] + chr(ord(word[-1]) + 1)
        hi = bisect.bisect_left(sorted_words, hi_key)
        exts = [sorted_words[i] for i in range(lo, hi)
                if sorted_words[i] != word
                and len(sorted_words[i]) <= len(word) + 6]
        if exts:
            _extension_index[word] = exts


# ── Word explorer endpoints ──

@app.get("/api/validar/{word}")
def validar_word(word: str):
    word = _normalize_word(word)
    tokens = tokenize_word(word)
    valid = _word_in_trie(trie, tokens)
    value = sum(INTERNAL_POINTS.get(t, 0) for t in tokens)

    result = {
        "word": word,
        "valid": valid,
        "length": len(tokens),
        "value": value,
    }

    # Anagram list
    anagram_key = tuple(sorted(tokens))
    anagram_list = [w for w in _anagram_index.get(anagram_key, []) if w != word]

    card = card_lookup.get(word)
    if card:
        raw_suffix = card.get("suffix", "")
        raw_prefix = card.get("prefix", "")
        result.update({
            "percentile": card.get("percentile", ""),
            "prefix": _resolve_prefix(word, raw_prefix),
            "suffix": _resolve_suffix(word, raw_suffix),
            "ending": card.get("ending", ""),
            "front_hooks": card.get("front_hooks", []),
            "back_hooks": card.get("back_hooks", []),
            "anagrams": len(anagram_list),
            "anagram_list": anagram_list,
            "verb_type": card.get("verb_type", ""),
        })
    else:
        result.update({
            "percentile": "", "prefix": "", "suffix": "",
            "ending": word[-1] if word else "",
            "front_hooks": [], "back_hooks": [],
            "anagrams": len(anagram_list),
            "anagram_list": anagram_list,
            "verb_type": "",
        })

    # SRS history for this word
    srs = progress.get(word)
    if srs:
        result["srs"] = {
            "easiness": round(srs.easiness, 2),
            "interval": srs.interval,
            "repetitions": srs.repetitions,
            "next_review": srs.next_review,
            "last_review": srs.last_review,
            "total_reviews": srs.total_reviews,
            "lapses": srs.lapses,
        }

    return result


@app.get("/api/transformar/{word}")
def transformar_word(word: str):
    word = _normalize_word(word)
    results = one_letter_changes(word, trie)
    # Group by position
    by_pos = {}
    for r in results:
        by_pos.setdefault(r["position"], []).append(r)
    return {"word": word, "count": len(results), "by_position": by_pos}


@app.get("/api/extender/{word}")
def extender_word(word: str):
    word = _normalize_word(word)
    results = insert_letter(word, trie)
    by_pos = {}
    for r in results:
        by_pos.setdefault(r["position"], []).append(r)
    return {"word": word, "count": len(results), "by_position": by_pos}


@app.get("/api/reducir/{word}")
def reducir_word(word: str):
    word = _normalize_word(word)
    results = remove_letter(word, trie)
    return {"word": word, "count": len(results), "results": results}


# ── Deck listing ──

@app.get("/api/mazos")
def listar_mazos():
    categories = []

    # Words by length
    cat = {"name": "Palabras por longitud (sin verbos conjugados)", "decks": []}
    for name in ["words-2", "words-3", "words-4", "words-5"]:
        cat["decks"].append({
            "id": name, "label": WORD_PRESETS[name],
            "count": len(apply_preset(all_cards, name))
        })
    categories.append(cat)

    # Vowel patterns
    cat = {"name": "Patrones vocálicos (7 letras)", "decks": []}
    for name in ["7L-2vowels", "7L-2cons"]:
        cat["decks"].append({
            "id": name, "label": WORD_PRESETS[name],
            "count": len(apply_preset(all_cards, name))
        })
    categories.append(cat)

    # Probability & scoring
    cat = {"name": "Alta probabilidad y puntuación", "decks": []}
    for name in ["high-prob", "scoring-5", "scoring-6"]:
        cat["decks"].append({
            "id": name, "label": WORD_PRESETS[name],
            "count": len(apply_preset(all_cards, name))
        })
    categories.append(cat)

    # 5L by ending
    cat = {"name": "5 letras por terminación", "decks": []}
    for name in ["5L-end-d", "5L-end-l", "5L-end-n", "5L-end-r", "5L-end-z"]:
        cat["decks"].append({
            "id": name, "label": WORD_PRESETS[name],
            "count": len(apply_preset(all_cards, name))
        })
    categories.append(cat)

    # Verbs
    cat = {"name": "Verbos por longitud", "decks": []}
    for name, desc in VERB_PRESETS.items():
        cat["decks"].append({
            "id": name, "label": desc,
            "count": len(apply_verb_preset(all_verbs, name))
        })
    categories.append(cat)

    # Custom lists
    if custom_lists:
        cat = {"name": "Mis listas", "decks": []}
        for lid, lst in custom_lists.items():
            cat["decks"].append({
                "id": f"custom:{lid}", "label": lst["name"],
                "count": len(lst["words"])
            })
        categories.append(cat)

    return {"categories": categories}


# ── Deck content ──

@app.get("/api/mazo/{deck_id}/palabras")
def get_deck_words(deck_id: str):
    """Return all words in a deck with metadata."""
    if deck_id in VERB_PRESETS:
        cards = apply_verb_preset(all_verbs, deck_id)
    elif deck_id in WORD_PRESETS:
        cards = apply_preset(all_cards, deck_id)
    else:
        return JSONResponse({"error": "Mazo no encontrado."}, 404)

    all_presets = {}
    all_presets.update(WORD_PRESETS)
    all_presets.update(VERB_PRESETS)
    label = all_presets.get(deck_id, deck_id)

    words = []
    for c in cards:
        words.append({
            "word": c["word"],
            "value": c["value"],
            "length": c["length"],
            "prefix": _resolve_prefix(c["word"], c.get("prefix", "")),
            "suffix": _resolve_suffix(c["word"], c.get("suffix", "")),
            "front_hooks": c.get("front_hooks", []),
            "back_hooks": c.get("back_hooks", []),
        })
    return {"deck_id": deck_id, "label": label, "count": len(words),
            "words": words}


@app.get("/api/mazo/{deck_id}/csv")
def export_deck_csv(deck_id: str):
    """Export deck words as CSV download."""
    from fastapi.responses import Response
    data = get_deck_words(deck_id)
    if isinstance(data, JSONResponse):
        return data

    lines = ["Palabra,Puntos,Longitud,Prefijo,Sufijo,Ganchos Delanteros,Ganchos Traseros"]
    for w in data["words"]:
        fh = " ".join(w["front_hooks"])
        bh = " ".join(w["back_hooks"])
        lines.append(f'{w["word"]},{w["value"]},{w["length"]},'
                     f'{w["prefix"]},{w["suffix"]},"{fh}","{bh}"')
    csv_content = "\n".join(lines)
    filename = f'mazo_{deck_id}.csv'
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


# ── Dynamic deck word listing ──

class DynamicDeckRequest(BaseModel):
    deck: str


@app.post("/api/mazo-dinamico/palabras")
def get_dynamic_deck_words(req: DynamicDeckRequest):
    """Return words for any deck ID (preset, custom, prefix, suffix, ending)."""
    cards = _resolve_deck(req.deck)
    words = []
    for c in cards[:5000]:  # cap at 5000 for performance
        words.append({
            "word": c["word"],
            "value": c["value"],
            "length": c["length"],
        })
    return {"deck": req.deck, "count": len(cards), "words": words}


# ── Custom lists ──

@app.get("/api/listas")
def get_listas():
    """Return all custom lists."""
    result = []
    for lid, lst in custom_lists.items():
        result.append({"id": lid, "name": lst["name"],
                       "count": len(lst["words"])})
    return {"lists": result}


@app.post("/api/listas")
def create_lista(req: CustomListRequest):
    """Create a custom list. Validates words against FISE2."""
    raw_words = [w.strip().lower() for w in
                 req.words.replace(",", "\n").split("\n") if w.strip()]
    # Normalize accents
    raw_words = [_normalize_word(w) for w in raw_words]
    valid = [w for w in raw_words if _word_in_trie(trie, tokenize_word(w))]
    invalid = [w for w in raw_words if w not in set(valid)]

    lid = str(uuid.uuid4())[:8]
    custom_lists[lid] = {"name": req.name, "words": valid}
    _save_custom_lists()

    return {"id": lid, "name": req.name, "valid": len(valid),
            "invalid": len(invalid), "invalid_words": invalid[:20]}


@app.delete("/api/listas/{list_id}")
def delete_lista(list_id: str):
    if list_id not in custom_lists:
        return JSONResponse({"error": "Lista no encontrada."}, 404)
    name = custom_lists[list_id]["name"]
    del custom_lists[list_id]
    _save_custom_lists()
    return {"deleted": list_id, "name": name}


@app.put("/api/listas/{list_id}")
def rename_lista(list_id: str, req: CustomListRequest):
    if list_id not in custom_lists:
        return JSONResponse({"error": "Lista no encontrada."}, 404)
    custom_lists[list_id]["name"] = req.name
    if req.words.strip():
        raw_words = [_normalize_word(w.strip().lower()) for w in
                     req.words.replace(",", "\n").split("\n") if w.strip()]
        valid = [w for w in raw_words if _word_in_trie(trie, tokenize_word(w))]
        custom_lists[list_id]["words"] = valid
    _save_custom_lists()
    return {"id": list_id, "name": custom_lists[list_id]["name"],
            "count": len(custom_lists[list_id]["words"])}


# ── Prefix/Suffix selectors ──

@app.get("/api/prefijos")
def get_prefijos():
    """Return all prefixes with word counts, alphabetically sorted."""
    prefix_counts = {}
    for c in all_cards:
        p = c.get("prefix", "")
        if p:
            resolved = _resolve_prefix(c["word"], p)
            prefix_counts.setdefault(resolved, {"pattern": p, "count": 0})
            prefix_counts[resolved]["count"] += 1
    result = [{"prefix": k, "pattern": v["pattern"], "count": v["count"]}
              for k, v in sorted(prefix_counts.items(),
                                  key=lambda x: x[0])]
    return {"prefixes": result, "total": len(result)}


@app.get("/api/sufijos")
def get_sufijos():
    """Return all suffixes with word counts, alphabetically sorted.
    Excludes abstract patterns CC and VV. Shows vowel variants."""
    EXCLUDE_PATTERNS = {"CC", "VV"}

    suffix_data = {}  # pattern -> {variants: set, count: int}
    for c in all_cards:
        s = c.get("suffix", "")
        if s and s not in EXCLUDE_PATTERNS:
            resolved = _resolve_suffix(c["word"], s)
            if s not in suffix_data:
                suffix_data[s] = {"variants": set(), "count": 0}
            suffix_data[s]["variants"].add(resolved)
            suffix_data[s]["count"] += 1

    result = []
    for pattern, data in suffix_data.items():
        variants = sorted(data["variants"])
        # Build compact label: find common base, list differing endings
        if len(variants) > 1 and "V" in pattern:
            # Extract the base (everything except last char) and list endings
            base = variants[0][:-1]
            endings = [v[-1] for v in variants]
            label = f"-{base},{','.join(endings)}"
        else:
            label = f"-{variants[0]}"
        result.append({
            "suffix": label, "pattern": pattern,
            "count": data["count"], "variants": variants,
        })

    result.sort(key=lambda x: x["variants"][0])
    return {"suffixes": result, "total": len(result)}


@app.get("/api/prefijos/{prefix}/palabras")
def get_prefix_words(prefix: str):
    """Return words matching a prefix."""
    words = [c for c in all_cards if c.get("prefix", "") == prefix]
    # Also try matching resolved prefix
    if not words:
        words = [c for c in all_cards
                 if _resolve_prefix(c["word"], c.get("prefix", "")) == prefix]
    result = [{"word": c["word"], "value": c["value"], "length": c["length"]}
              for c in words]
    return {"prefix": prefix, "count": len(result), "words": result}


@app.get("/api/sufijos/{suffix}/palabras")
def get_suffix_words(suffix: str):
    """Return words matching a suffix pattern."""
    words = [c for c in all_cards if c.get("suffix", "") == suffix]
    result = [{"word": c["word"], "value": c["value"], "length": c["length"]}
              for c in words]
    return {"suffix": suffix, "count": len(result), "words": result}


@app.get("/api/terminaciones")
def get_terminaciones():
    """Return all endings with word counts, filterable by length."""
    ending_counts = {}
    for c in all_cards:
        e = c.get("ending", "")
        if e:
            ending_counts[e] = ending_counts.get(e, 0) + 1
    result = [{"ending": k, "count": v}
              for k, v in sorted(ending_counts.items(),
                                  key=lambda x: -x[1])]
    return {"endings": result, "total": len(result)}


@app.get("/api/terminaciones/{ending}/palabras")
def get_ending_words(ending: str, length: int = 0):
    """Return words with a given ending, optionally filtered by length."""
    words = [c for c in all_cards if c.get("ending", "") == ending]
    if length > 0:
        words = [c for c in words if c["length"] == length]
    result = [{"word": c["word"], "value": c["value"], "length": c["length"]}
              for c in words]
    return {"ending": ending, "length": length, "count": len(result),
            "words": result}


# ── Progress / Stats ──

@app.get("/api/progreso")
def progreso():
    from datetime import date
    today = date.today()
    total = len(progress)
    if total == 0:
        return {"total": 0, "due": 0, "total_reviews": 0,
                "mastered": 0, "learning": 0, "struggling": 0, "avg_ef": 0}
    due = len(get_due_cards(progress, today))
    total_reviews = sum(s.total_reviews for s in progress.values())
    mastered = sum(1 for s in progress.values()
                   if s.easiness >= 2.5 and s.repetitions >= 3)
    learning = sum(1 for s in progress.values() if 0 < s.repetitions < 3)
    struggling = sum(1 for s in progress.values() if s.easiness < 1.8)
    avg_ef = sum(s.easiness for s in progress.values()) / total
    return {
        "total": total, "due": due, "total_reviews": total_reviews,
        "mastered": mastered, "learning": learning,
        "struggling": struggling, "avg_ef": round(avg_ef, 2),
    }


@app.get("/api/progreso/detalle/{category}")
def progreso_detalle(category: str):
    """Return word lists for a progress category."""
    from datetime import date
    today = date.today()

    words = []
    if category == "estudiadas":
        for word, s in sorted(progress.items()):
            words.append({
                "word": word, "easiness": round(s.easiness, 2),
                "repetitions": s.repetitions, "lapses": s.lapses,
                "next_review": s.next_review, "total_reviews": s.total_reviews,
            })
    elif category == "pendientes":
        due = get_due_cards(progress, today)
        for word in due:
            s = progress[word]
            words.append({
                "word": word, "easiness": round(s.easiness, 2),
                "next_review": s.next_review, "lapses": s.lapses,
            })
    elif category == "dominadas":
        for word, s in sorted(progress.items()):
            if s.easiness >= 2.5 and s.repetitions >= 3:
                words.append({
                    "word": word, "easiness": round(s.easiness, 2),
                    "repetitions": s.repetitions,
                })
    elif category == "dificultad":
        for word, s in sorted(progress.items()):
            if s.easiness < 1.8:
                words.append({
                    "word": word, "easiness": round(s.easiness, 2),
                    "lapses": s.lapses, "repetitions": s.repetitions,
                })
    elif category == "aprendizaje":
        for word, s in sorted(progress.items()):
            if 0 < s.repetitions < 3:
                words.append({
                    "word": word, "easiness": round(s.easiness, 2),
                    "repetitions": s.repetitions, "next_review": s.next_review,
                })
    else:
        return JSONResponse({"error": "Categoría no válida."}, 400)

    return {"category": category, "count": len(words), "words": words}


# ── Quiz session management ──

def _resolve_deck(deck_id, min_length=None, max_length=None):
    """Resolve a deck ID to a list of card dicts."""
    if deck_id in VERB_PRESETS:
        return apply_verb_preset(all_verbs, deck_id)
    elif deck_id in WORD_PRESETS:
        return apply_preset(all_cards, deck_id)
    elif deck_id.startswith("custom:"):
        lid = deck_id[7:]
        if lid in custom_lists:
            words = custom_lists[lid]["words"]
            return [card_lookup[w] for w in words if w in card_lookup]
        return []
    elif deck_id.startswith("prefix:"):
        prefix = deck_id[7:]
        return [c for c in all_cards if c.get("prefix", "") == prefix]
    elif deck_id.startswith("suffix:"):
        suffix = deck_id[7:]
        return [c for c in all_cards if c.get("suffix", "") == suffix]
    elif deck_id.startswith("ending:"):
        parts = deck_id[7:].split(":")
        ending = parts[0]
        length = int(parts[1]) if len(parts) > 1 and parts[1] else 0
        result = [c for c in all_cards if c.get("ending", "") == ending]
        if length > 0:
            result = [c for c in result if c["length"] == length]
        return result
    elif deck_id == "" and min_length and max_length:
        return filter_cards(all_cards, min_length=min_length,
                            max_length=max_length)
    else:
        return all_cards


def _generate_card_prompt(session, index):
    """Generate the prompt for a card at the given index."""
    card = session.cards[index]
    mode = session.mode
    tokens = list(card["tokens"])

    base = {
        "index": index,
        "total": len(session.cards),
        "mode": mode,
        "word_display": card["word"].upper(),
        "length": card["length"],
        "value": card["value"],
    }

    if mode == "review":
        base["prompt_type"] = "review"

    elif mode == "anagram":
        scrambled = _scramble(tokens)
        scrambled_display = [display_token(t) for t in scrambled]
        session.card_states[index] = {"scrambled": scrambled}
        base["prompt_type"] = "anagram"
        base["scrambled"] = scrambled_display

    elif mode == "hooks":
        base["prompt_type"] = "hooks"
        base["has_front"] = bool(card.get("front_hooks"))
        base["has_back"] = bool(card.get("back_hooks"))

    elif mode == "pattern":
        blanked, hidden = _blank_tokens(tokens)
        blanked_display = [display_token(t) if t != "_" else "_"
                           for t in blanked]
        session.card_states[index] = {"blanked": blanked, "hidden": hidden}
        base["prompt_type"] = "pattern"
        base["blanked"] = blanked_display

    elif mode == "morphology":
        word = card["word"]
        derivations = _extension_index.get(word, [])
        if derivations:
            # Cap at 15 for display
            session.card_states[index] = {
                "derivations": derivations[:15],
                "all_derivations": set(derivations),
            }
            base["prompt_type"] = "morphology"
            base["derivation_count"] = len(derivations)
        else:
            base["prompt_type"] = "skip"
            base["reason"] = f"'{word.upper()}' no tiene derivaciones en el léxico."

    elif mode == "transformation":
        changes = one_letter_changes(card["word"], trie)
        if changes:
            by_pos = {}
            for c in changes:
                by_pos.setdefault(c["position"], []).append(c)
            pos = random.choice(list(by_pos.keys()))
            pos_changes = by_pos[pos]
            display_parts = []
            for j, t in enumerate(tokens):
                display_parts.append("_" if j == pos else display_token(t))
            session.card_states[index] = {
                "position": pos,
                "changes": pos_changes,
                "actual": set(c["replacement"] for c in pos_changes),
            }
            base["prompt_type"] = "transformation"
            base["display"] = display_parts
            base["position"] = pos
            base["original"] = display_token(tokens[pos])
            base["change_count"] = len(pos_changes)
        else:
            base["prompt_type"] = "skip"
            base["reason"] = "No hay transformaciones para esta palabra."

    elif mode == "extension":
        inserts = insert_letter(card["word"], trie)
        if inserts:
            by_pos = {}
            for ins in inserts:
                by_pos.setdefault(ins["position"], []).append(ins)
            pos = random.choice(list(by_pos.keys()))
            pos_inserts = by_pos[pos]
            display_parts = []
            for j, t in enumerate(tokens):
                if j == pos:
                    display_parts.append("[_]")
                display_parts.append(display_token(t))
            if pos == len(tokens):
                display_parts.append("[_]")
            session.card_states[index] = {
                "position": pos,
                "inserts": pos_inserts,
                "actual": set(ins["inserted"] for ins in pos_inserts),
            }
            base["prompt_type"] = "extension"
            base["display"] = display_parts
            base["position"] = pos
            base["insert_count"] = len(pos_inserts)
        else:
            base["prompt_type"] = "skip"
            base["reason"] = "No hay extensiones para esta palabra."

    elif mode == "reduction":
        removals = remove_letter(card["word"], trie)
        if removals:
            session.card_states[index] = {
                "removals": removals,
                "actual": set(r["removed"] for r in removals),
            }
            base["prompt_type"] = "reduction"
            base["display"] = [display_token(t) for t in tokens]
            base["removal_count"] = len(removals)
        else:
            base["prompt_type"] = "skip"
            base["reason"] = "No hay reducciones para esta palabra."

    return base


def _build_reveal(card):
    """Build reveal data for a card."""
    word = card["word"]
    return {
        "word": word.upper(),
        "length": card["length"],
        "value": card["value"],
        "front_hooks": card.get("front_hooks", []),
        "back_hooks": card.get("back_hooks", []),
        "prefix": _resolve_prefix(word, card.get("prefix", "")),
        "suffix": _resolve_suffix(word, card.get("suffix", "")),
        "ending": card.get("ending", ""),
        "anagrams": card.get("anagrams", 0),
        "verb_type": card.get("verb_type", ""),
    }


@app.post("/api/quiz/iniciar")
def iniciar_quiz(req: QuizStartRequest):
    pool_cards = _resolve_deck(req.deck, req.min_length, req.max_length)

    # Filter pool for mode-specific requirements
    if req.mode == "hooks":
        pool_cards = [c for c in pool_cards
                      if c.get("front_hooks") or c.get("back_hooks")]
    elif req.mode == "morphology":
        pool_cards = [c for c in pool_cards
                      if c["word"] in _extension_index]

    pool = [c["word"] for c in pool_cards]
    session_words = build_session(progress, pool, session_size=req.size)
    pool_lookup = {c["word"]: c for c in pool_cards}
    session_cards = [pool_lookup[w] for w in session_words if w in pool_lookup]

    if not session_cards:
        return JSONResponse({"error": "No hay tarjetas disponibles para este modo y mazo."}, 400)

    sid = str(uuid.uuid4())[:8]
    session = QuizSession(
        session_id=sid, mode=req.mode, cards=session_cards
    )
    session._deck_id = req.deck  # for history logging
    sessions[sid] = session

    # Resolve deck label for display
    all_presets = {}
    all_presets.update(WORD_PRESETS)
    all_presets.update(VERB_PRESETS)
    deck_label = all_presets.get(req.deck, req.deck or "Personalizado")

    # Resolve mode label
    mode_labels = {
        "review": "Repaso", "anagram": "Anagrama", "hooks": "Ganchos",
        "pattern": "Patrón", "morphology": "Morfología",
        "transformation": "Transformación", "extension": "Extensión",
        "reduction": "Reducción",
    }
    mode_label = mode_labels.get(req.mode, req.mode)

    prompt = _generate_card_prompt(session, 0)
    return {
        "session_id": sid, "total": len(session_cards), "card": prompt,
        "mode_label": mode_label, "deck_label": deck_label,
    }


@app.get("/api/quiz/{session_id}/tarjeta")
def get_tarjeta(session_id: str):
    session = sessions.get(session_id)
    if not session:
        return JSONResponse({"error": "Sesión no encontrada."}, 404)
    if session.current_index >= len(session.cards):
        return _session_summary(session)
    prompt = _generate_card_prompt(session, session.current_index)
    return {"card": prompt}


@app.post("/api/quiz/{session_id}/responder")
def responder(session_id: str, req: AnswerRequest):
    session = sessions.get(session_id)
    if not session:
        return JSONResponse({"error": "Sesión no encontrada."}, 404)

    idx = session.current_index
    if idx >= len(session.cards):
        return _session_summary(session)

    card = session.cards[idx]
    mode = session.mode
    answer = req.answer.strip().lower()
    tokens = list(card["tokens"])
    reveal_data = _build_reveal(card)

    if answer == "?":
        quality = 0
        # Enrich reveal with mode-specific data
        cs = session.card_states.get(idx, {})
        if mode == "anagram":
            anagram_key = tuple(sorted(tokens))
            reveal_data["all_anagrams"] = sorted(_anagram_index.get(anagram_key, []))
        elif mode == "morphology":
            reveal_data["all_derivations"] = sorted(cs.get("all_derivations", set()))[:15]
        elif mode in ("transformation", "extension", "reduction"):
            if mode == "transformation":
                reveal_data["changes"] = [
                    {"replacement": c["replacement"].upper(),
                     "word": c["word"].upper(), "value": c["value"]}
                    for c in cs.get("changes", [])
                ]
            elif mode == "extension":
                reveal_data["inserts"] = [
                    {"inserted": i["inserted"].upper(),
                     "word": i["word"].upper(), "value": i["value"]}
                    for i in cs.get("inserts", [])
                ]
            elif mode == "reduction":
                reveal_data["removals"] = [
                    {"removed": r["removed"].upper(),
                     "word": r["word"].upper(), "value": r["value"],
                     "position": r["position"]}
                    for r in cs.get("removals", [])
                ]
        return _apply_and_advance(session, card, quality, reveal_data,
                                  correct=False, skipped=True)

    if mode == "anagram":
        # User submits all anagrams they can find (comma or space separated)
        given_words = set()
        for part in answer.replace(",", " ").split():
            w = part.strip().lower()
            if w:
                given_words.add(w)

        # Get all valid anagrams
        anagram_key = tuple(sorted(tokens))
        all_anagrams = set(_anagram_index.get(anagram_key, []))
        reveal_data["all_anagrams"] = sorted(all_anagrams)

        # Classify each answer
        correct_words = []
        wrong_words = []
        for w in given_words:
            if w in all_anagrams:
                correct_words.append(w)
            else:
                wrong_words.append(w)
        missed_words = sorted(all_anagrams - set(correct_words))

        # Score by fraction found (penalize wrong answers)
        if all_anagrams:
            score = max(0, (len(correct_words) - 0.5 * len(wrong_words))) / len(all_anagrams)
            score = min(1.0, score)
        else:
            score = 1.0

        quality = _quality_from_score(score)
        reveal_data["score"] = round(score * 100)
        reveal_data["correct_words"] = sorted(correct_words)
        reveal_data["wrong_words"] = sorted(wrong_words)
        reveal_data["missed_words"] = missed_words

        return _apply_and_advance(session, card, quality, reveal_data,
                                  correct=quality >= 3)

    elif mode == "hooks":
        # Answer format: "front_letters|back_letters" (pipe-separated)
        parts = answer.split("|")
        front_answer = parts[0].strip() if len(parts) > 0 else ""
        back_answer = parts[1].strip() if len(parts) > 1 else ""

        front_actual = set(card.get("front_hooks", []))
        back_actual = set(card.get("back_hooks", []))

        scores = []
        if front_actual:
            front_given = _parse_hook_input(front_answer) if front_answer else set()
            scores.append(_hook_score(front_given, front_actual))
        if back_actual:
            back_given = _parse_hook_input(back_answer) if back_answer else set()
            scores.append(_hook_score(back_given, back_actual))

        avg_score = sum(scores) / len(scores) if scores else 0
        quality = _quality_from_score(avg_score)
        reveal_data["score"] = round(avg_score * 100)
        return _apply_and_advance(session, card, quality, reveal_data,
                                  correct=quality >= 3)

    elif mode == "pattern":
        answer_tokens = tokenize_word(answer)
        cs = session.card_states.get(idx, {})
        blanked = cs.get("blanked", [])
        hidden = cs.get("hidden", set())

        # Check: same length, visible letters match, valid word, same points
        answer_value = sum(INTERNAL_POINTS.get(t, 0) for t in answer_tokens)
        card_value = card["value"]
        pattern_match = (
            len(answer_tokens) == len(tokens)
            and all(answer_tokens[j] == tokens[j]
                    for j in range(len(tokens)) if j not in hidden)
            and _word_in_trie(trie, answer_tokens)
            and answer_value == card_value
        )
        exact_match = answer_tokens == tokens

        if exact_match or pattern_match:
            quality = 5 if session.attempt == 0 else 3
            if not exact_match:
                # Also show the original word
                reveal_data["note"] = f"Tu respuesta ({answer.upper()}) es válida. La palabra original era {card['word'].upper()}."
            return _apply_and_advance(session, card, quality, reveal_data,
                                      correct=True)
        else:
            session.attempt += 1
            # Feedback
            if len(answer_tokens) != len(tokens):
                hint = f"La palabra debe tener {len(tokens)} fichas."
            elif not _word_in_trie(trie, answer_tokens):
                hint = f"'{answer.upper()}' no es una palabra válida."
            elif answer_value != card_value:
                hint = f"'{answer.upper()}' vale {answer_value} pts, no {card_value}."
            else:
                hint = "Las letras visibles no coinciden con el patrón."
            if session.attempt >= 2:
                return _apply_and_advance(session, card, 1, reveal_data,
                                          correct=False)
            return {"correct": False, "can_retry": True, "hint": hint}

    elif mode == "morphology":
        # User submits derivations (comma or space separated)
        given_words = set()
        for part in answer.replace(",", " ").split():
            w = part.strip().lower()
            if w:
                given_words.add(w)

        cs = session.card_states.get(idx, {})
        all_derivations = cs.get("all_derivations", set())
        capped = cs.get("derivations", [])

        correct_words = sorted(given_words & all_derivations)
        wrong_words = sorted(given_words - all_derivations)
        missed_words = sorted(set(capped) - given_words)

        if all_derivations:
            # Score against the capped list (max 15)
            target = set(capped)
            score = max(0, (len(given_words & target) - 0.5 * len(wrong_words))) / len(target)
            score = min(1.0, score)
        else:
            score = 0

        quality = _quality_from_score(score)
        reveal_data["score"] = round(score * 100)
        reveal_data["correct_words"] = correct_words
        reveal_data["wrong_words"] = wrong_words
        reveal_data["missed_words"] = missed_words
        reveal_data["all_derivations"] = sorted(all_derivations)[:15]
        return _apply_and_advance(session, card, quality, reveal_data,
                                  correct=quality >= 3)

    elif mode in ("transformation", "extension", "reduction"):
        cs = session.card_states.get(idx, {})
        actual = cs.get("actual", set())
        given = _parse_hook_input(answer)
        score = _hook_score(given, actual)
        quality = _quality_from_score(score)
        reveal_data["score"] = round(score * 100)

        if mode == "transformation":
            reveal_data["changes"] = [
                {"replacement": c["replacement"].upper(),
                 "word": c["word"].upper(), "value": c["value"]}
                for c in cs.get("changes", [])
            ]
        elif mode == "extension":
            reveal_data["inserts"] = [
                {"inserted": i["inserted"].upper(),
                 "word": i["word"].upper(), "value": i["value"]}
                for i in cs.get("inserts", [])
            ]
        elif mode == "reduction":
            reveal_data["removals"] = [
                {"removed": r["removed"].upper(),
                 "word": r["word"].upper(), "value": r["value"],
                 "position": r["position"]}
                for r in cs.get("removals", [])
            ]

        return _apply_and_advance(session, card, quality, reveal_data,
                                  correct=quality >= 3)

    elif mode == "review":
        # Review mode: answer is the quality rating (0-5)
        try:
            quality = max(0, min(5, int(answer)))
        except ValueError:
            quality = 0
        return _apply_and_advance(session, card, quality, reveal_data,
                                  correct=quality >= 3)

    return JSONResponse({"error": "Modo no soportado."}, 400)


@app.get("/api/quiz/{session_id}/revelar")
def revelar(session_id: str):
    """Return reveal data for the current card WITHOUT advancing."""
    session = sessions.get(session_id)
    if not session:
        return JSONResponse({"error": "Sesión no encontrada."}, 404)
    idx = session.current_index
    if idx >= len(session.cards):
        return _session_summary(session)
    card = session.cards[idx]
    return {"reveal": _build_reveal(card)}


@app.post("/api/quiz/{session_id}/calificar")
def calificar(session_id: str, req: RateRequest):
    """Rate the current card (review mode) and advance."""
    session = sessions.get(session_id)
    if not session:
        return JSONResponse({"error": "Sesión no encontrada."}, 404)
    idx = session.current_index
    if idx >= len(session.cards):
        return _session_summary(session)
    card = session.cards[idx]
    quality = max(0, min(5, req.quality))
    reveal_data = _build_reveal(card)
    return _apply_and_advance(session, card, quality, reveal_data,
                              correct=quality >= 3)


@app.post("/api/quiz/{session_id}/saltar")
def saltar(session_id: str):
    """Skip the current card (no SRS update) and advance."""
    session = sessions.get(session_id)
    if not session:
        return JSONResponse({"error": "Sesión no encontrada."}, 404)
    session.current_index += 1
    session.attempt = 0
    if session.current_index >= len(session.cards):
        return _session_summary(session)
    prompt = _generate_card_prompt(session, session.current_index)
    return {"card": prompt}


@app.get("/api/quiz/{session_id}/siguiente")
def siguiente(session_id: str):
    """Advance to next card (used after reviewing reveal)."""
    session = sessions.get(session_id)
    if not session:
        return JSONResponse({"error": "Sesión no encontrada."}, 404)
    if session.current_index >= len(session.cards):
        return _session_summary(session)
    prompt = _generate_card_prompt(session, session.current_index)
    return {"card": prompt}


def _apply_and_advance(session, card, quality, reveal_data,
                        correct=False, skipped=False):
    """Apply SRS update and advance to next card."""
    state = progress.get(card["word"], CardState())
    progress[card["word"]] = update_card(state, quality)
    save_progress(progress)

    session.results.append((card["word"], quality))
    session.current_index += 1
    session.attempt = 0

    done = session.current_index >= len(session.cards)

    response = {
        "correct": correct,
        "skipped": skipped,
        "quality": quality,
        "reveal": reveal_data,
        "done": done,
    }

    if done:
        response["summary"] = _build_summary(session)

    return response


def _session_summary(session):
    return {"done": True, "summary": _build_summary(session)}


def _build_summary(session):
    if not session.results:
        return {"reviewed": 0, "avg_quality": 0, "struggling": []}
    avg = sum(q for _, q in session.results) / len(session.results)
    struggling = [w for w, q in session.results if q < 3]
    correct = [w for w, q in session.results if q >= 3]
    summary = {
        "reviewed": len(session.results),
        "avg_quality": round(avg, 1),
        "struggling": struggling,
    }

    # Log to session history
    all_presets = {}
    all_presets.update(WORD_PRESETS)
    all_presets.update(VERB_PRESETS)

    mode_labels = {
        "review": "Repaso", "anagram": "Anagrama", "hooks": "Ganchos",
        "pattern": "Patrón", "morphology": "Derivaciones",
        "transformation": "Transformación", "extension": "Extensión",
        "reduction": "Reducción",
    }

    entry = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "mode": mode_labels.get(session.mode, session.mode),
        "deck": all_presets.get(
            getattr(session, '_deck_id', ''), "Personalizado"),
        "reviewed": len(session.results),
        "avg_quality": round(avg, 1),
        "correct": correct,
        "struggling": struggling,
    }
    session_history.append(entry)
    _save_history()

    return summary


# ── History endpoints ──

@app.get("/api/historial")
def get_historial():
    """Return session history, most recent first."""
    return {"sessions": list(reversed(session_history)),
            "total": len(session_history)}


@app.get("/api/historial/csv")
def export_historial_csv():
    """Export session history as CSV download."""
    from fastapi.responses import Response
    lines = ["Fecha,Modo,Mazo,Revisadas,Calidad Promedio,Correctas,En Dificultad"]
    for s in session_history:
        correct_str = "; ".join(s.get("correct", []))
        struggling_str = "; ".join(s.get("struggling", []))
        lines.append(f'{s["date"]},{s["mode"]},{s["deck"]},'
                     f'{s["reviewed"]},{s["avg_quality"]},'
                     f'"{correct_str}","{struggling_str}"')
    csv_content = "\n".join(lines)
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=historial_lexicable.csv"}
    )


# ── Static files ──

@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ── Main ──

if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser(description="Estudio Scrabble Web Server")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
