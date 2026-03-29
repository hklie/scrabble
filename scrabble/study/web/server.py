#!/usr/bin/env python3
"""Web server for the Spanish Scrabble study tool.

Usage:
    cd scrabble
    python -m study.web.server [--port 8080]
"""

import argparse
import os
import random
import sys
import uuid
from dataclasses import dataclass, field

# Fix imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from lexicon import load_lexicon_trie, is_valid_word, word_value, _word_in_trie
from preprocessing import tokenize_word, detokenize_word
from config import SCRABBLE_POINTS, DIGRAPHS, INTERNAL_POINTS, ALL_TILES
from study.decks import (load_word_analysis, load_verbs, apply_preset,
                         apply_verb_preset, available_decks,
                         WORD_PRESETS, VERB_PRESETS)
from study.srs import (load_progress, save_progress, update_card,
                        build_session, get_due_cards, CardState)
from study.transforms import one_letter_changes, insert_letter, remove_letter
from study.quiz import (_scramble, _blank_tokens, _parse_hook_input,
                         _hook_score, _quality_from_score, display_token)

# ── App setup ──

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

app = FastAPI(title="Estudio Scrabble Español")

# ── Global state (loaded at startup) ──

trie = None
all_cards = []
all_verbs = []
card_lookup = {}
progress = {}
sessions = {}


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


# ── Word explorer endpoints ──

@app.get("/api/validar/{word}")
def validar_word(word: str):
    word = word.strip().lower()
    tokens = tokenize_word(word)
    valid = _word_in_trie(trie, tokens)
    value = sum(INTERNAL_POINTS.get(t, 0) for t in tokens)

    result = {
        "word": word,
        "valid": valid,
        "length": len(tokens),
        "value": value,
    }

    card = card_lookup.get(word)
    if card:
        result.update({
            "percentile": card.get("percentile", ""),
            "prefix": card.get("prefix", ""),
            "suffix": card.get("suffix", ""),
            "ending": card.get("ending", ""),
            "front_hooks": card.get("front_hooks", []),
            "back_hooks": card.get("back_hooks", []),
            "anagrams": card.get("anagrams", 0),
            "verb_type": card.get("verb_type", ""),
        })
    else:
        result.update({
            "percentile": "", "prefix": "", "suffix": "",
            "ending": word[-1] if word else "",
            "front_hooks": [], "back_hooks": [],
            "anagrams": 0, "verb_type": "",
        })

    return result


@app.get("/api/transformar/{word}")
def transformar_word(word: str):
    word = word.strip().lower()
    results = one_letter_changes(word, trie)
    # Group by position
    by_pos = {}
    for r in results:
        by_pos.setdefault(r["position"], []).append(r)
    return {"word": word, "count": len(results), "by_position": by_pos}


@app.get("/api/extender/{word}")
def extender_word(word: str):
    word = word.strip().lower()
    results = insert_letter(word, trie)
    by_pos = {}
    for r in results:
        by_pos.setdefault(r["position"], []).append(r)
    return {"word": word, "count": len(results), "by_position": by_pos}


@app.get("/api/reducir/{word}")
def reducir_word(word: str):
    word = word.strip().lower()
    results = remove_letter(word, trie)
    return {"word": word, "count": len(results), "results": results}


# ── Deck listing ──

@app.get("/api/mazos")
def listar_mazos():
    categories = []

    # Words by length
    cat = {"name": "Palabras por longitud", "decks": []}
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

    return {"categories": categories}


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


# ── Quiz session management ──

def _resolve_deck(deck_id, min_length=None, max_length=None):
    """Resolve a deck ID to a list of card dicts."""
    if deck_id in VERB_PRESETS:
        return apply_verb_preset(all_verbs, deck_id)
    elif deck_id in WORD_PRESETS:
        return apply_preset(all_cards, deck_id)
    elif deck_id == "" and min_length and max_length:
        from study.decks import filter_cards
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
        base["prompt_type"] = "morphology"
        base["has_prefix"] = bool(card.get("prefix"))
        base["has_suffix"] = bool(card.get("suffix"))

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
    return {
        "word": card["word"].upper(),
        "length": card["length"],
        "value": card["value"],
        "front_hooks": card.get("front_hooks", []),
        "back_hooks": card.get("back_hooks", []),
        "prefix": card.get("prefix", ""),
        "suffix": card.get("suffix", ""),
        "ending": card.get("ending", ""),
        "anagrams": card.get("anagrams", 0),
        "verb_type": card.get("verb_type", ""),
    }


@app.post("/api/quiz/iniciar")
def iniciar_quiz(req: QuizStartRequest):
    pool_cards = _resolve_deck(req.deck, req.min_length, req.max_length)
    pool = [c["word"] for c in pool_cards]
    session_words = build_session(progress, pool, session_size=req.size)
    pool_lookup = {c["word"]: c for c in pool_cards}
    session_cards = [pool_lookup[w] for w in session_words if w in pool_lookup]

    if not session_cards:
        return JSONResponse({"error": "No hay tarjetas disponibles."}, 400)

    sid = str(uuid.uuid4())[:8]
    session = QuizSession(
        session_id=sid, mode=req.mode, cards=session_cards
    )
    sessions[sid] = session

    prompt = _generate_card_prompt(session, 0)
    return {"session_id": sid, "total": len(session_cards), "card": prompt}


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
        return _apply_and_advance(session, card, quality, reveal_data,
                                  correct=False, skipped=True)

    if mode == "anagram":
        answer_tokens = tokenize_word(answer)
        correct = (sorted(answer_tokens) == sorted(tokens)
                   and answer_tokens == tokens)
        if correct:
            quality = 5 if session.attempt == 0 else 3
            return _apply_and_advance(session, card, quality, reveal_data,
                                      correct=True)
        else:
            session.attempt += 1
            if session.attempt >= 2:
                return _apply_and_advance(session, card, 1, reveal_data,
                                          correct=False)
            return {"correct": False, "can_retry": True}

    elif mode == "hooks":
        given = _parse_hook_input(answer)
        front = set(card.get("front_hooks", []))
        back = set(card.get("back_hooks", []))
        combined = front | back
        score = _hook_score(given, combined)
        quality = _quality_from_score(score)
        reveal_data["score"] = round(score * 100)
        return _apply_and_advance(session, card, quality, reveal_data,
                                  correct=quality >= 3)

    elif mode == "pattern":
        answer_tokens = tokenize_word(answer)
        correct = answer_tokens == tokens
        if correct:
            quality = 5 if session.attempt == 0 else 3
            return _apply_and_advance(session, card, quality, reveal_data,
                                      correct=True)
        else:
            session.attempt += 1
            if session.attempt >= 2:
                return _apply_and_advance(session, card, 1, reveal_data,
                                          correct=False)
            return {"correct": False, "can_retry": True}

    elif mode == "morphology":
        parts = [p.strip() for p in answer.split(",")]
        prefix_ans = parts[0] if len(parts) > 0 else ""
        suffix_ans = parts[1] if len(parts) > 1 else ""
        scores = []
        if card.get("prefix"):
            scores.append(1.0 if prefix_ans == card["prefix"] else 0.0)
        if card.get("suffix"):
            scores.append(1.0 if suffix_ans == card["suffix"] else 0.0)
        avg = sum(scores) / len(scores) if scores else 0
        quality = _quality_from_score(avg)
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
    return {
        "reviewed": len(session.results),
        "avg_quality": round(avg, 1),
        "struggling": struggling,
    }


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
