"""srs.py — SM-2 Spaced Repetition Engine with JSON persistence.

Pure scheduling logic — no knowledge of Scrabble or word content.
Maps string keys (words) to review state, persists in Data/progress.json.
"""

import json
import os
from dataclasses import dataclass, asdict, field
from datetime import date, timedelta
from typing import Optional

PROGRESS_FILE = os.path.join(os.path.dirname(__file__),
                             '..', '..', 'Data', 'progress.json')


@dataclass
class CardState:
    easiness: float = 2.5
    interval: int = 0
    repetitions: int = 0
    next_review: str = ""       # ISO date; empty = never reviewed
    last_review: str = ""
    total_reviews: int = 0
    lapses: int = 0


def load_progress(path: str = PROGRESS_FILE) -> dict[str, CardState]:
    """Load progress.json → dict of word → CardState. Empty dict if missing."""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cards = {}
    for word, state in data.get("cards", {}).items():
        cards[word] = CardState(**state)
    return cards


def save_progress(cards: dict[str, CardState], path: str = PROGRESS_FILE) -> None:
    """Atomically write progress.json (write .tmp then rename)."""
    data = {
        "version": 1,
        "cards": {word: asdict(state) for word, state in cards.items()},
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def update_card(state: CardState, quality: int) -> CardState:
    """Apply SM-2 algorithm. quality: 0–5. Returns updated CardState.

    0 = complete blackout
    1 = wrong, but recognized on reveal
    2 = wrong, but felt close
    3 = correct with serious difficulty
    4 = correct after hesitation
    5 = perfect recall
    """
    quality = max(0, min(5, quality))
    today = date.today().isoformat()

    if quality < 3:
        state.repetitions = 0
        state.interval = 1
        state.lapses += 1
    else:
        if state.repetitions == 0:
            state.interval = 1
        elif state.repetitions == 1:
            state.interval = 6
        else:
            state.interval = max(1, round(state.interval * state.easiness))
        state.repetitions += 1

    state.easiness += 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)
    state.easiness = max(1.3, state.easiness)

    state.next_review = (date.today() + timedelta(days=state.interval)).isoformat()
    state.last_review = today
    state.total_reviews += 1
    return state


def get_due_cards(cards: dict[str, CardState],
                  today: Optional[date] = None) -> list[str]:
    """Words with next_review <= today, sorted most overdue first."""
    today = today or date.today()
    today_str = today.isoformat()
    due = []
    for word, state in cards.items():
        if state.next_review and state.next_review <= today_str:
            due.append((word, state.next_review))
    due.sort(key=lambda x: x[1])
    return [w for w, _ in due]


def get_new_cards(cards: dict[str, CardState],
                  pool: list[str], limit: int = 10) -> list[str]:
    """Words from pool with no entry in cards, up to limit."""
    new = [w for w in pool if w not in cards]
    return new[:limit]


def build_session(cards: dict[str, CardState], pool: list[str],
                  session_size: int = 20) -> list[str]:
    """Due cards first, then new cards to fill remaining slots."""
    pool_set = set(pool)
    due = [w for w in get_due_cards(cards) if w in pool_set]

    if len(due) >= session_size:
        return due[:session_size]

    remaining = session_size - len(due)
    new_limit = min(10, remaining)
    new = get_new_cards(cards, pool, limit=new_limit)
    return due + new
