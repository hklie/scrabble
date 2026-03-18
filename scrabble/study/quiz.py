#!/usr/bin/env python3
"""Interactive spaced-repetition quiz for Spanish Scrabble word study.

Usage:
    python -m study.quiz                        # interactive menu
    python -m study.quiz --mode anagram         # anagram drill
    python -m study.quiz --mode hooks           # hook quiz
    python -m study.quiz --mode pattern         # pattern fill
    python -m study.quiz --mode morphology      # prefix/suffix/ending quiz
    python -m study.quiz --deck bingo-7         # preset deck
    python -m study.quiz --length 7             # filter by length
    python -m study.quiz --tier 4               # rare letter words
    python -m study.quiz --size 30              # session of 30 cards
    python -m study.quiz --stats                # show progress stats
    python -m study.quiz --list-decks           # list preset decks

Run from the scrabble/ package directory:
    cd /home/hk/Code/BackEnd/scrabble/scrabble && python -m study.quiz
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from study.srs import (load_progress, save_progress, update_card,
                        build_session, get_due_cards, CardState)
from study.decks import (load_word_analysis, filter_cards, apply_preset,
                         available_decks, PERCENTILE_RANK,
                         WORD_PRESETS, VERB_PRESETS, apply_verb_preset,
                         group_by_prefix, group_by_suffix, group_by_ending,
                         load_verbs, filter_verbs,
                         group_verbs_by_beginning, group_verbs_by_type)
from preprocessing import tokenize_word, detokenize_word
from config import SCRABBLE_POINTS, DIGRAPHS, INTERNAL_POINTS, ALL_TILES
from lexicon import load_lexicon_trie, is_valid_word, word_value, _word_in_trie


# ── Terminal helpers ──

def clear_screen():
    os.system("clear" if os.name != "nt" else "cls")


def colored(text, color):
    """Simple ANSI coloring."""
    codes = {"green": "32", "red": "31", "yellow": "33",
             "cyan": "36", "bold": "1", "dim": "2"}
    c = codes.get(color, "0")
    return f"\033[{c}m{text}\033[0m"


def display_token(token):
    """Convert internal token to display form (uppercase)."""
    return DIGRAPHS.get(token, token).upper()


def display_tokens(tokens):
    """Token list → display string with spaces."""
    return "  ".join(display_token(t) for t in tokens)


def input_safe(prompt=""):
    """Input that handles Ctrl-C/D gracefully."""
    try:
        return input(prompt)
    except (KeyboardInterrupt, EOFError):
        print()
        return None


# ── Morphology info (shown on reveal in all modes) ──

def _show_morphology(card):
    """Display prefix, suffix, ending for a card."""
    parts = []
    if card["prefix"]:
        parts.append(f"Prefix: {colored(card['prefix'] + '-', 'cyan')}")
    if card["suffix"]:
        parts.append(f"Suffix: {colored('-' + card['suffix'], 'cyan')}")
    if card["ending"]:
        parts.append(f"Ending: {card['ending']}")
    if parts:
        print("  " + "  |  ".join(parts))


def _show_full_reveal(card):
    """Show hooks + morphology + anagrams + verb type after answering."""
    if card["front_hooks"] or card["back_hooks"]:
        fh = ", ".join(card["front_hooks"]) if card["front_hooks"] else "—"
        bh = ", ".join(card["back_hooks"]) if card["back_hooks"] else "—"
        print(f"  Front hooks: {fh}")
        print(f"  Back hooks:  {bh}")
    _show_morphology(card)
    if card.get("verb_type"):
        print(f"  Type: {colored(card['verb_type'], 'cyan')}")
    if card["anagrams"] > 1:
        print(f"  Anagrams: {card['anagrams']}")


# ── Review mode ──

def run_review(session_cards, progress):
    """Self-assessment flashcards. Show word, reveal info, rate 0–5."""
    results = []
    total = len(session_cards)

    for i, card in enumerate(session_cards, 1):
        clear_screen()
        print(colored(f"  Review {i}/{total}", "bold"))
        print()
        print(f"  {colored(card['word'].upper(), 'cyan')}")
        print(f"  {card['length']} letters  |  {card['value']} pts  |  {card['percentile']}")
        print()
        r = input_safe("  Press Enter to reveal (q to quit)... ")
        if r is None or r.strip().lower() == "q":
            break

        # Reveal
        print()
        _show_full_reveal(card)
        print()

        q = _ask_quality()
        if q is None:
            break

        state = progress.get(card["word"], CardState())
        progress[card["word"]] = update_card(state, q)
        results.append((card["word"], q))

    return results


# ── Anagram mode ──

def run_anagram(session_cards, progress):
    """Show scrambled letters, player types the word."""
    results = []
    total = len(session_cards)

    for i, card in enumerate(session_cards, 1):
        clear_screen()
        tokens = list(card["tokens"])
        scrambled = _scramble(tokens)
        print(colored(f"  Anagram {i}/{total}", "bold"))
        print()
        print(f"  Letters: {display_tokens(scrambled)}")
        print(f"  ({card['length']} letters, {card['value']} pts)")
        print()

        quality = 0
        for attempt in range(2):
            answer = input_safe("  Your answer (? to reveal): ")
            if answer is None:
                return results
            answer = answer.strip().lower()
            if answer == "?":
                print(f"\n  {colored(card['word'].upper(), 'yellow')}")
                quality = 0
                break
            answer_tokens = tokenize_word(answer)
            if sorted(answer_tokens) == sorted(tokens) and _is_valid_word(answer, card):
                print(f"  {colored('Correct!', 'green')}")
                quality = 5 if attempt == 0 else 3
                break
            else:
                if attempt == 0:
                    print(f"  {colored('Try again...', 'red')}")
                else:
                    print(f"\n  Answer: {colored(card['word'].upper(), 'yellow')}")
                    quality = 1

        _show_full_reveal(card)
        input_safe("\n  Enter to continue... ")
        state = progress.get(card["word"], CardState())
        progress[card["word"]] = update_card(state, quality)
        results.append((card["word"], quality))

    return results


def _is_valid_word(answer, card):
    """Check if the answer matches the card's word (tokenized comparison)."""
    return tokenize_word(answer) == card["tokens"]


def _scramble(tokens):
    """Shuffle tokens, ensuring different order from original if possible."""
    if len(tokens) <= 1:
        return list(tokens)
    for _ in range(20):
        s = list(tokens)
        random.shuffle(s)
        if s != tokens:
            return s
    return list(tokens)


# ── Hook quiz mode ──

def run_hooks(session_cards, progress):
    """Show a word, ask what letters can hook before/after."""
    results = []
    total = len(session_cards)

    # Only use cards that actually have hooks
    hookable = [c for c in session_cards
                if c["front_hooks"] or c["back_hooks"]]
    if not hookable:
        print("  No hookable words in this session.")
        return results

    for i, card in enumerate(hookable, 1):
        clear_screen()
        print(colored(f"  Hook Quiz {i}/{len(hookable)}", "bold"))
        print()
        print(f"  Word: {colored(card['word'].upper(), 'cyan')}")
        print(f"  ({card['length']} letters, {card['value']} pts)")
        print()

        score_parts = []

        if card["front_hooks"]:
            answer = input_safe("  Letters that hook BEFORE (? to skip): ")
            if answer is None:
                return results
            if answer.strip() == "?":
                s = 0.0
            else:
                given = _parse_hook_input(answer)
                actual = set(card["front_hooks"])
                s = _hook_score(given, actual)
            score_parts.append(s)
            print(f"  Front hooks: {', '.join(card['front_hooks'])}"
                  f"  {'✓' if s >= 0.8 else ''}")

        if card["back_hooks"]:
            answer = input_safe("  Letters that hook AFTER  (? to skip): ")
            if answer is None:
                return results
            if answer.strip() == "?":
                s = 0.0
            else:
                given = _parse_hook_input(answer)
                actual = set(card["back_hooks"])
                s = _hook_score(given, actual)
            score_parts.append(s)
            print(f"  Back hooks:  {', '.join(card['back_hooks'])}"
                  f"  {'✓' if s >= 0.8 else ''}")

        avg_score = sum(score_parts) / len(score_parts) if score_parts else 0
        quality = _quality_from_score(avg_score)
        print(f"\n  Score: {avg_score:.0%} → quality {quality}")

        input_safe("\n  Enter to continue... ")
        state = progress.get(card["word"], CardState())
        progress[card["word"]] = update_card(state, quality)
        results.append((card["word"], quality))

    return results


def _parse_hook_input(text):
    """Parse hook input: accepts 'S Z T', 's,z,t', 's, z, t', 'szt', 'ch ll' etc."""
    text = text.strip().lower()
    # Check for comma-separated
    if "," in text:
        parts = [p.strip() for p in text.split(",") if p.strip()]
    elif " " in text:
        parts = text.split()
    else:
        # Single string like "szt" — split into individual chars,
        # but first check for digraphs
        parts = []
        i = 0
        while i < len(text):
            if i + 1 < len(text) and text[i:i+2] in ("ch", "ll", "rr"):
                parts.append(text[i:i+2])
                i += 2
            elif text[i] == "ñ":
                parts.append("ñ")
                i += 1
            else:
                parts.append(text[i])
                i += 1
    return set(parts)


def _hook_score(given, actual):
    """Fraction of hooks correctly identified, with penalty for false positives."""
    if not actual:
        return 1.0 if not given else 0.0
    correct = given & actual
    false_pos = given - actual
    score = (len(correct) - 0.5 * len(false_pos)) / len(actual)
    return max(0.0, min(1.0, score))


def _quality_from_score(score):
    if score >= 0.9:
        return 5
    if score >= 0.7:
        return 4
    if score >= 0.5:
        return 3
    if score >= 0.3:
        return 2
    return 1


# ── Pattern fill mode ──

def run_pattern(session_cards, progress):
    """Show word with blanks, player fills them in."""
    results = []
    total = len(session_cards)

    for i, card in enumerate(session_cards, 1):
        clear_screen()
        tokens = list(card["tokens"])
        blanked, hidden_idx = _blank_tokens(tokens)

        print(colored(f"  Pattern {i}/{total}", "bold"))
        print()
        print(f"  {display_tokens(blanked)}")
        print(f"  ({card['length']} letters, {card['value']} pts)")
        print()

        quality = 0
        for attempt in range(2):
            answer = input_safe("  Full word (? to reveal): ")
            if answer is None:
                return results
            answer = answer.strip().lower()
            if answer == "?":
                print(f"\n  {colored(card['word'].upper(), 'yellow')}")
                quality = 0
                break
            if tokenize_word(answer) == tokens:
                print(f"  {colored('Correct!', 'green')}")
                quality = 5 if attempt == 0 else 3
                break
            else:
                if attempt == 0:
                    print(f"  {colored('Try again...', 'red')}")
                else:
                    print(f"\n  Answer: {colored(card['word'].upper(), 'yellow')}")
                    quality = 1

        _show_full_reveal(card)
        input_safe("\n  Enter to continue... ")
        state = progress.get(card["word"], CardState())
        progress[card["word"]] = update_card(state, quality)
        results.append((card["word"], quality))

    return results


def _blank_tokens(tokens):
    """Replace ~40% of tokens with '_', preferring high-value ones."""
    n = len(tokens)
    n_blank = max(1, min(n - 2, round(n * 0.4)))

    # Sort indices by point value (descending), blank the highest-value ones
    def token_value(idx):
        t = tokens[idx]
        return INTERNAL_POINTS.get(t, SCRABBLE_POINTS.get(DIGRAPHS.get(t, t), 0))

    indices = sorted(range(n), key=token_value, reverse=True)
    hidden = set(indices[:n_blank])

    blanked = ["_" if i in hidden else tokens[i] for i in range(n)]
    return blanked, hidden


# ── Morphology quiz mode ──

def run_morphology(session_cards, progress):
    """Given a word, name its prefix, suffix, and ending."""
    # Only quiz cards that have at least a prefix or suffix
    quizzable = [c for c in session_cards if c["prefix"] or c["suffix"]]
    if not quizzable:
        print("  No words with prefix/suffix in this session.")
        return []

    results = []
    total = len(quizzable)

    for i, card in enumerate(quizzable, 1):
        clear_screen()
        print(colored(f"  Morphology {i}/{total}", "bold"))
        print()
        print(f"  Word: {colored(card['word'].upper(), 'cyan')}")
        print(f"  ({card['length']} letters, {card['value']} pts)")
        print()

        score_parts = []

        # Ask prefix
        if card["prefix"]:
            answer = input_safe("  Prefix (? to skip): ")
            if answer is None:
                return results
            if answer.strip() == "?":
                score_parts.append(0.0)
            elif answer.strip().lower() == card["prefix"]:
                print(f"  {colored('Correct!', 'green')}")
                score_parts.append(1.0)
            else:
                print(f"  {colored('Wrong', 'red')}")
                score_parts.append(0.0)
            print(f"  Prefix: {colored(card['prefix'] + '-', 'yellow')}")

        # Ask suffix
        if card["suffix"]:
            answer = input_safe("  Suffix (? to skip): ")
            if answer is None:
                return results
            if answer.strip() == "?":
                score_parts.append(0.0)
            elif answer.strip().lower() == card["suffix"]:
                print(f"  {colored('Correct!', 'green')}")
                score_parts.append(1.0)
            else:
                print(f"  {colored('Wrong', 'red')}")
                score_parts.append(0.0)
            print(f"  Suffix: {colored('-' + card['suffix'], 'yellow')}")

        # Always show ending
        print(f"  Ending: {card['ending']}")

        avg_score = sum(score_parts) / len(score_parts) if score_parts else 0
        quality = _quality_from_score(avg_score)
        print(f"\n  Score: {avg_score:.0%} → quality {quality}")

        _show_full_reveal(card)
        input_safe("\n  Enter to continue... ")
        state = progress.get(card["word"], CardState())
        progress[card["word"]] = update_card(state, quality)
        results.append((card["word"], quality))

    return results


# ── Transformation quiz mode ──

def run_transformation(session_cards, progress):
    """Show a word, player must find valid one-letter changes."""
    from study.transforms import one_letter_changes

    trie = load_lexicon_trie()
    results = []
    total = len(session_cards)

    for i, card in enumerate(session_cards, 1):
        clear_screen()
        print(colored(f"  Transformation {i}/{total}", "bold"))
        print()
        print(f"  Word: {colored(card['word'].upper(), 'cyan')}")
        print(f"  ({card['length']} letters, {card['value']} pts)")
        print()

        all_changes = one_letter_changes(card['word'], trie)
        if not all_changes:
            print("  No one-letter changes exist for this word.")
            input_safe("  Enter to continue... ")
            continue

        # Pick a random position that has changes
        positions_with_changes = {}
        for c in all_changes:
            positions_with_changes.setdefault(c['position'], []).append(c)

        pos = random.choice(list(positions_with_changes.keys()))
        pos_changes = positions_with_changes[pos]
        tokens = list(card['tokens'])
        orig_display = display_token(tokens[pos])

        # Show word with one position highlighted
        display_parts = []
        for j, t in enumerate(tokens):
            if j == pos:
                display_parts.append(colored("_", "yellow"))
            else:
                display_parts.append(display_token(t))
        print(f"  {' '.join(display_parts)}")
        print(f"  Replace {colored(orig_display, 'yellow')} at position {pos + 1} "
              f"to form a new word.")
        print(f"  ({len(pos_changes)} valid replacement(s))")
        print()

        answer = input_safe("  Replacement letters (space-separated, ? to reveal): ")
        if answer is None:
            return results

        if answer.strip() == "?":
            quality = 0
        else:
            given = _parse_hook_input(answer)
            actual = set(c['replacement'] for c in pos_changes)
            quality = _quality_from_score(_hook_score(given, actual))

        # Show all valid changes at this position
        print()
        for c in sorted(pos_changes, key=lambda x: x['word']):
            print(f"    {c['replacement'].upper():4s} → {c['word'].upper()}  ({c['value']} pts)")

        print(f"\n  Quality: {quality}")
        input_safe("\n  Enter to continue... ")
        state = progress.get(card["word"], CardState())
        progress[card["word"]] = update_card(state, quality)
        results.append((card["word"], quality))

    return results


# ── Extension quiz mode ──

def run_extension(session_cards, progress):
    """Show a word with a blank slot, player names valid letters to insert."""
    from study.transforms import insert_letter

    trie = load_lexicon_trie()
    results = []
    total = len(session_cards)

    for i, card in enumerate(session_cards, 1):
        clear_screen()
        print(colored(f"  Extension {i}/{total}", "bold"))
        print()
        print(f"  Word: {colored(card['word'].upper(), 'cyan')}")
        print(f"  ({card['length']} letters, {card['value']} pts)")
        print()

        all_inserts = insert_letter(card['word'], trie)
        if not all_inserts:
            print("  No insertions found for this word.")
            input_safe("  Enter to continue... ")
            continue

        # Pick a random position
        positions_with_inserts = {}
        for ins in all_inserts:
            positions_with_inserts.setdefault(ins['position'], []).append(ins)

        pos = random.choice(list(positions_with_inserts.keys()))
        pos_inserts = positions_with_inserts[pos]
        tokens = list(card['tokens'])

        # Show word with insertion slot
        display_parts = []
        for j, t in enumerate(tokens):
            if j == pos:
                display_parts.append(colored("[_]", "yellow"))
            display_parts.append(display_token(t))
        if pos == len(tokens):
            display_parts.append(colored("[_]", "yellow"))

        if pos == 0:
            pos_label = "before first letter"
        elif pos == len(tokens):
            pos_label = "after last letter"
        else:
            pos_label = f"between positions {pos} and {pos + 1}"

        print(f"  {'  '.join(display_parts)}")
        print(f"  Insert a letter {pos_label}.")
        print(f"  ({len(pos_inserts)} valid insertion(s))")
        print()

        answer = input_safe("  Letters to insert (space-separated, ? to reveal): ")
        if answer is None:
            return results

        if answer.strip() == "?":
            quality = 0
        else:
            given = _parse_hook_input(answer)
            actual = set(ins['inserted'] for ins in pos_inserts)
            quality = _quality_from_score(_hook_score(given, actual))

        # Show all valid insertions at this position
        print()
        for ins in sorted(pos_inserts, key=lambda x: x['word']):
            print(f"    +{ins['inserted'].upper():4s} → {ins['word'].upper()}  ({ins['value']} pts)")

        print(f"\n  Quality: {quality}")
        input_safe("\n  Enter to continue... ")
        state = progress.get(card["word"], CardState())
        progress[card["word"]] = update_card(state, quality)
        results.append((card["word"], quality))

    return results


# ── Reduction quiz mode ──

def run_reduction(session_cards, progress):
    """Show a word, player must identify which letters can be removed."""
    from study.transforms import remove_letter

    trie = load_lexicon_trie()
    results = []
    total = len(session_cards)

    for i, card in enumerate(session_cards, 1):
        clear_screen()
        print(colored(f"  Reduction {i}/{total}", "bold"))
        print()
        print(f"  Word: {colored(card['word'].upper(), 'cyan')}")
        print(f"  ({card['length']} letters, {card['value']} pts)")
        print()

        all_removals = remove_letter(card['word'], trie)
        if not all_removals:
            print("  No valid removals for this word.")
            input_safe("  Enter to continue... ")
            continue

        print(f"  Which letter(s) can be removed to leave a valid word?")
        print(f"  ({len(all_removals)} valid removal(s))")
        print()

        answer = input_safe("  Letters to remove (space-separated, ? to reveal): ")
        if answer is None:
            return results

        if answer.strip() == "?":
            quality = 0
        else:
            given = _parse_hook_input(answer)
            actual = set(r['removed'] for r in all_removals)
            quality = _quality_from_score(_hook_score(given, actual))

        # Show all valid removals
        print()
        tokens = list(card['tokens'])
        for r in sorted(all_removals, key=lambda x: x['position']):
            removed_display = display_token(tokens[r['position']])
            print(f"    -{removed_display:4s} (pos {r['position'] + 1}) → "
                  f"{r['word'].upper()}  ({r['value']} pts)")

        print(f"\n  Quality: {quality}")
        input_safe("\n  Enter to continue... ")
        state = progress.get(card["word"], CardState())
        progress[card["word"]] = update_card(state, quality)
        results.append((card["word"], quality))

    return results


# ── Word lookup (Consultar palabra) ──

def _word_lookup_menu(all_cards):
    """Interactive word validity lookup with additional info."""
    trie = load_lexicon_trie()
    card_lookup = {c["word"]: c for c in all_cards}

    while True:
        clear_screen()
        print(colored("  ── Consultar Palabra ──", "bold"))
        print()
        word = input_safe("  Word (q to go back): ")
        if word is None or word.strip().lower() == "q":
            return
        word = word.strip().lower()
        if not word:
            continue

        tokens = tokenize_word(word)
        valid = _word_in_trie(trie, tokens)
        value = sum(INTERNAL_POINTS.get(t, 0) for t in tokens)

        print()
        if valid:
            print(f"  {colored(word.upper(), 'green')}  ── VÁLIDA")
        else:
            print(f"  {colored(word.upper(), 'red')}  ── NO VÁLIDA")

        print(f"  {len(tokens)} letters  |  {value} pts")

        # Show card info if available
        card = card_lookup.get(word)
        if card:
            print()
            _show_full_reveal(card)

        # Show transforms summary
        if valid:
            from study.transforms import one_letter_changes, insert_letter, remove_letter
            changes = one_letter_changes(word, trie)
            inserts = insert_letter(word, trie)
            removals = remove_letter(word, trie)
            print()
            print(f"  Transformations: {len(changes)} changes, "
                  f"{len(inserts)} insertions, {len(removals)} removals")

        print()
        input_safe("  Enter to continue... ")


# ── Quality input ──

def _ask_quality():
    """Prompt for 0–5 rating."""
    print("  Rate:  0=blackout  1=wrong  2=hard  3=difficult  4=ok  5=easy")
    while True:
        r = input_safe("  Quality (0-5): ")
        if r is None:
            return None
        r = r.strip()
        if r in ("0", "1", "2", "3", "4", "5"):
            return int(r)
        print("  Please enter 0–5.")


# ── Stats ──

def show_stats(progress):
    """Display progress statistics."""
    if not progress:
        print("  No progress recorded yet. Start a quiz session!")
        return

    from datetime import date
    today = date.today()
    total = len(progress)
    due = len(get_due_cards(progress, today))
    total_reviews = sum(s.total_reviews for s in progress.values())
    total_lapses = sum(s.lapses for s in progress.values())

    mastered = sum(1 for s in progress.values() if s.easiness >= 2.5 and s.repetitions >= 3)
    learning = sum(1 for s in progress.values() if 0 < s.repetitions < 3)
    struggling = sum(1 for s in progress.values() if s.easiness < 1.8)

    avg_ef = sum(s.easiness for s in progress.values()) / total

    print()
    print(colored("  ── Progress ──", "bold"))
    print(f"  Total words studied:  {total}")
    print(f"  Due today:            {due}")
    print(f"  Total reviews:        {total_reviews}")
    print(f"  Total lapses:         {total_lapses}")
    print()
    print(f"  Mastered (EF≥2.5, 3+ reps):  {mastered}")
    print(f"  Learning (1-2 reps):          {learning}")
    print(f"  Struggling (EF<1.8):          {struggling}")
    print(f"  Average ease factor:          {avg_ef:.2f}")
    print()


# ── Session summary ──

def show_summary(results):
    """End-of-session summary."""
    if not results:
        print("\n  No cards reviewed.")
        return

    avg_q = sum(q for _, q in results) / len(results)
    struggling = [w for w, q in results if q < 3]

    print()
    print(colored("  ── Session Summary ──", "bold"))
    print(f"  Cards reviewed:   {len(results)}")
    print(f"  Average quality:  {avg_q:.1f}")
    if struggling:
        print(f"  Struggling ({len(struggling)}):")
        for w in struggling:
            print(f"    {w}")
    print()


# ── Interactive menu ──

def interactive_menu(args, all_cards, progress):
    """Show a menu when no mode is specified via args."""
    due_count = len(get_due_cards(progress))

    while True:
        clear_screen()
        total_studied = len(progress)
        print(colored("  ── Scrabble Word Quiz ──", "bold"))
        print()
        print(f"  Words studied: {total_studied}   Due today: {due_count}")
        print()
        print("  Modes:")
        print("    1. Review (self-assessment flashcards)")
        print("    2. Anagram drill")
        print("    3. Hook quiz")
        print("    4. Pattern fill")
        print("    5. Morphology (prefix/suffix/ending)")
        print("    6. Transformation (one-letter change)")
        print("    7. Extension (insert a letter)")
        print("    8. Reduction (remove a letter)")
        print()
        print("  Options:")
        print("    c. Check word (consultar palabra)")
        print("    v. Verb study (by length, beginning, or type)")
        print("    g. Group study (by prefix, suffix, or ending)")
        print("    s. Show stats")
        print("    d. List preset decks")
        print("    q. Quit")
        print()

        choice = input_safe("  Choice: ")
        if choice is None or choice.strip().lower() == "q":
            return

        choice = choice.strip().lower()

        if choice == "s":
            clear_screen()
            show_stats(progress)
            input_safe("  Enter to continue... ")
            continue

        if choice == "d":
            clear_screen()
            print(colored("  ── Preset Decks ──", "bold"))
            print()
            for name, desc in available_decks().items():
                print(f"    {name:16s}  {desc}")
            print()
            input_safe("  Enter to continue... ")
            continue

        if choice == "v":
            _verb_study_menu(args, progress)
            due_count = len(get_due_cards(progress))
            continue

        if choice == "g":
            _group_study_menu(args, all_cards, progress)
            due_count = len(get_due_cards(progress))
            continue

        if choice == "c":
            _word_lookup_menu(all_cards)
            continue

        mode_map = {"1": "review", "2": "anagram", "3": "hooks",
                    "4": "pattern", "5": "morphology",
                    "6": "transformation", "7": "extension",
                    "8": "reduction"}
        mode = mode_map.get(choice)
        if not mode:
            continue

        # Ask for deck
        pool_cards = _deck_selection_menu(all_cards, progress, args, due_count)
        if pool_cards is None:
            continue

        # Build session and run
        pool = [c["word"] for c in pool_cards]
        session_words = build_session(progress, pool, session_size=args.size)
        card_lookup = {c["word"]: c for c in pool_cards}
        session_cards = [card_lookup[w] for w in session_words if w in card_lookup]

        if not session_cards:
            print("\n  No cards available for this selection.")
            input_safe("  Enter to continue... ")
            continue

        print(f"\n  Starting {mode} with {len(session_cards)} cards...")
        input_safe("  Enter to begin... ")

        run_fn = {"review": run_review, "anagram": run_anagram,
                  "hooks": run_hooks, "pattern": run_pattern,
                  "morphology": run_morphology,
                  "transformation": run_transformation,
                  "extension": run_extension,
                  "reduction": run_reduction}[mode]
        results = run_fn(session_cards, progress)
        save_progress(progress)

        clear_screen()
        show_summary(results)
        due_count = len(get_due_cards(progress))
        input_safe("  Enter to continue... ")


def _deck_selection_menu(all_cards, progress, args, due_count):
    """Show organized deck categories. Returns card list or None to cancel."""
    clear_screen()
    print(colored("  ── Deck Selection ──", "bold"))
    print()

    # Build numbered list from categories
    items = []  # (label, preset_name, is_verb)

    print(colored("  Words by length:", "dim"))
    for name in ["words-2", "words-3", "words-4", "words-5"]:
        items.append((WORD_PRESETS[name], name, False))
        print(f"    {len(items):2d}. {WORD_PRESETS[name]}")

    print(colored("  7-letter vowel patterns:", "dim"))
    for name in ["7L-2vowels", "7L-2cons"]:
        items.append((WORD_PRESETS[name], name, False))
        print(f"    {len(items):2d}. {WORD_PRESETS[name]}")

    print(colored("  High probability & scoring:", "dim"))
    for name in ["high-prob", "scoring-5", "scoring-6"]:
        items.append((WORD_PRESETS[name], name, False))
        print(f"    {len(items):2d}. {WORD_PRESETS[name]}")

    print(colored("  5-letter by ending:", "dim"))
    for name in ["5L-end-l", "5L-end-n", "5L-end-r", "5L-end-z"]:
        items.append((WORD_PRESETS[name], name, False))
        print(f"    {len(items):2d}. {WORD_PRESETS[name]}")

    print(colored("  Verbs by length:", "dim"))
    for name, desc in VERB_PRESETS.items():
        items.append((desc, name, True))
        print(f"    {len(items):2d}. {desc}")

    n_presets = len(items)
    print()
    print(f"    {n_presets+1}. Custom filter")
    print(f"    Enter. Review due cards ({due_count} due)")
    print()

    deck_choice = input_safe("  Choice: ")
    if deck_choice is None:
        return None
    deck_choice = deck_choice.strip()

    if deck_choice == "":
        return all_cards
    elif deck_choice.isdigit():
        idx = int(deck_choice) - 1
        if 0 <= idx < n_presets:
            _, preset_name, is_verb = items[idx]
            if is_verb:
                verbs = load_verbs()
                return apply_verb_preset(verbs, preset_name)
            else:
                return apply_preset(all_cards, preset_name)
        elif idx == n_presets:
            return _custom_filter_menu(all_cards)
    return None


def _verb_study_menu(args, progress):
    """Study verbs filtered by length, beginning, or type."""
    print("  Loading verbs...", end=" ", flush=True)
    all_verbs = load_verbs()
    print(f"{len(all_verbs)} verbs loaded.")

    clear_screen()
    print(colored("  ── Verb Study ──", "bold"))
    print()
    print("  Filter by:")
    print("    1. Length")
    print("    2. Beginning (first letters)")
    print("    3. Type (transitivo, intransitivo, ...)")
    print("    4. Browse beginnings (grouped)")
    print("    5. All verbs")
    print()

    choice = input_safe("  Choice (q to go back): ")
    if choice is None or choice.strip().lower() == "q":
        return
    choice = choice.strip()

    if choice == "1":
        ln = input_safe("  Word length (3-8): ")
        if ln is None or not ln.strip().isdigit():
            return
        ln = int(ln.strip())
        filtered = filter_verbs(all_verbs, min_length=ln, max_length=ln)
        label = f"{ln}-letter verbs"

    elif choice == "2":
        beg = input_safe("  Beginning (e.g., des, arr, re): ")
        if beg is None or not beg.strip():
            return
        beg = beg.strip().lower()
        filtered = filter_verbs(all_verbs, beginning=beg)
        label = f"verbs starting with '{beg}'"

    elif choice == "3":
        clear_screen()
        print(colored("  ── Verb Types ──", "bold"))
        print()
        type_groups = group_verbs_by_type(all_verbs)
        type_list = list(type_groups.items())
        for j, (vtype, vlist) in enumerate(type_list, 1):
            print(f"    {j}. {vtype:16s}  ({len(vlist)} verbs)")
        print()
        tc = input_safe("  Choice: ")
        if tc is None or not tc.strip().isdigit():
            return
        idx = int(tc.strip()) - 1
        if not (0 <= idx < len(type_list)):
            return
        vtype, filtered = type_list[idx]
        label = f"{vtype} verbs"

    elif choice == "4":
        clear_screen()
        print(colored("  ── Verb Beginnings ──", "bold"))
        print()
        pl = input_safe("  Prefix length (2-4, default 3): ")
        prefix_len = int(pl.strip()) if pl and pl.strip().isdigit() else 3
        prefix_len = max(2, min(4, prefix_len))

        groups = group_verbs_by_beginning(all_verbs, prefix_len=prefix_len,
                                          min_group=3)
        group_list = list(groups.items())
        if not group_list:
            print("  No groups found.")
            input_safe("  Enter to continue... ")
            return

        # Paginated display
        page = 0
        page_size = 20
        while True:
            clear_screen()
            print(colored(f"  ── Verb Beginnings ({prefix_len} chars) ──", "bold"))
            print()
            start = page * page_size
            end = min(start + page_size, len(group_list))
            for j in range(start, end):
                name, cards = group_list[j]
                print(f"    {j+1:3d}. {name:8s}  ({len(cards)} verbs)")
            print()
            if end < len(group_list):
                print(f"  Showing {start+1}–{end} of {len(group_list)}.  "
                      f"n=next, p=prev")
            nav = input_safe("  Enter group number (q to go back): ")
            if nav is None or nav.strip().lower() == "q":
                return
            if nav.strip().lower() == "n" and end < len(group_list):
                page += 1
                continue
            if nav.strip().lower() == "p" and page > 0:
                page -= 1
                continue
            if nav.strip().isdigit():
                idx = int(nav.strip()) - 1
                if 0 <= idx < len(group_list):
                    name, filtered = group_list[idx]
                    label = f"verbs starting with '{name}'"
                    break
            continue
        else:
            return

    elif choice == "5":
        filtered = list(all_verbs)
        label = "all verbs"

    else:
        return

    if not filtered:
        print(f"\n  No verbs found for {label}.")
        input_safe("  Enter to continue... ")
        return

    print(f"\n  {len(filtered)} {label}")

    # Pick quiz mode
    print()
    print("  Mode:")
    print("    1. Review        2. Anagram")
    print("    3. Pattern")
    print()
    mc = input_safe("  Choice (default=1): ")
    if mc is None:
        return
    mode_map = {"1": "review", "2": "anagram", "3": "pattern",
                "": "review"}
    mode = mode_map.get(mc.strip(), "review")

    pool = [c["word"] for c in filtered]
    session_words = build_session(progress, pool, session_size=args.size)
    card_lookup = {c["word"]: c for c in filtered}
    session_cards = [card_lookup[w] for w in session_words if w in card_lookup]

    if not session_cards:
        print("\n  No cards available.")
        input_safe("  Enter to continue... ")
        return

    # Show verb type in card for display
    print(f"\n  Starting {mode} with {len(session_cards)} verbs ({label})...")
    input_safe("  Enter to begin... ")

    run_fn = {"review": run_review, "anagram": run_anagram,
              "pattern": run_pattern}[mode]
    results = run_fn(session_cards, progress)
    save_progress(progress)

    clear_screen()
    show_summary(results)
    input_safe("  Enter to continue... ")


def _group_study_menu(args, all_cards, progress):
    """Study words grouped by prefix, suffix, or ending."""
    clear_screen()
    print(colored("  ── Group Study ──", "bold"))
    print()
    print("  Group by:")
    print("    1. Prefix")
    print("    2. Suffix")
    print("    3. Ending")
    print()

    choice = input_safe("  Choice (q to go back): ")
    if choice is None or choice.strip().lower() == "q":
        return
    choice = choice.strip()

    group_fn = {"1": group_by_prefix, "2": group_by_suffix,
                "3": group_by_ending}.get(choice)
    group_label = {"1": "prefix", "2": "suffix", "3": "ending"}.get(choice)
    if not group_fn:
        return

    groups = group_fn(all_cards, min_group=3)
    if not groups:
        print("  No groups found.")
        input_safe("  Enter to continue... ")
        return

    # Show available groups
    clear_screen()
    print(colored(f"  ── Groups by {group_label} ──", "bold"))
    print()
    group_list = list(groups.items())
    # Show in pages of 20
    page = 0
    page_size = 20
    while True:
        start = page * page_size
        end = min(start + page_size, len(group_list))
        for j in range(start, end):
            name, cards = group_list[j]
            print(f"    {j+1:3d}. {name:12s}  ({len(cards)} words)")
        print()
        if end < len(group_list):
            print(f"  Showing {start+1}–{end} of {len(group_list)}.  "
                  f"n=next page, p=prev page")
        nav = input_safe("  Enter group number (q to go back): ")
        if nav is None or nav.strip().lower() == "q":
            return
        if nav.strip().lower() == "n" and end < len(group_list):
            page += 1
            clear_screen()
            print(colored(f"  ── Groups by {group_label} ──", "bold"))
            print()
            continue
        if nav.strip().lower() == "p" and page > 0:
            page -= 1
            clear_screen()
            print(colored(f"  ── Groups by {group_label} ──", "bold"))
            print()
            continue
        if nav.strip().isdigit():
            idx = int(nav.strip()) - 1
            if 0 <= idx < len(group_list):
                group_name, group_cards = group_list[idx]
                _run_group_session(args, group_name, group_label,
                                   group_cards, progress)
                return
        # Invalid, re-show
        clear_screen()
        print(colored(f"  ── Groups by {group_label} ──", "bold"))
        print()


def _run_group_session(args, group_name, group_label, group_cards, progress):
    """Run a study session for a specific group."""
    clear_screen()
    print(colored(f"  ── {group_label.title()}: {group_name} "
                  f"({len(group_cards)} words) ──", "bold"))
    print()
    print("  Mode:")
    print("    1. Review        2. Anagram")
    print("    3. Hooks         4. Pattern")
    print("    5. Morphology    6. Transformation")
    print("    7. Extension     8. Reduction")
    print()

    choice = input_safe("  Choice: ")
    if choice is None:
        return
    mode_map = {"1": "review", "2": "anagram", "3": "hooks",
                "4": "pattern", "5": "morphology",
                "6": "transformation", "7": "extension",
                "8": "reduction"}
    mode = mode_map.get(choice.strip(), "review")

    pool = [c["word"] for c in group_cards]
    session_words = build_session(progress, pool, session_size=args.size)
    card_lookup = {c["word"]: c for c in group_cards}
    session_cards = [card_lookup[w] for w in session_words if w in card_lookup]

    if not session_cards:
        print("\n  No cards available.")
        input_safe("  Enter to continue... ")
        return

    print(f"\n  Starting {mode} with {len(session_cards)} cards "
          f"({group_label}: {group_name})...")
    input_safe("  Enter to begin... ")

    run_fn = {"review": run_review, "anagram": run_anagram,
              "hooks": run_hooks, "pattern": run_pattern,
              "morphology": run_morphology,
              "transformation": run_transformation,
              "extension": run_extension,
              "reduction": run_reduction}[mode]
    results = run_fn(session_cards, progress)
    save_progress(progress)

    clear_screen()
    show_summary(results)
    input_safe("  Enter to continue... ")


def _custom_filter_menu(all_cards):
    """Prompt for custom filter parameters."""
    print()
    length = input_safe("  Word length (Enter=any): ")
    if length is None:
        return None
    min_len = int(length) if length.strip().isdigit() else 2
    max_len = int(length) if length.strip().isdigit() else 15

    tier = input_safe("  Consonant tier 1-4 (Enter=any): ")
    if tier is None:
        return None
    tier_val = int(tier) if tier.strip() in ("1", "2", "3", "4") else None

    ending = input_safe("  Ending letter (Enter=any): ")
    if ending is None:
        return None
    endings = {ending.strip().lower()} if ending.strip() else None

    pct = input_safe("  Min percentile P10/P25/P50/P75/P90/Top10 (Enter=P10): ")
    if pct is None:
        return None
    min_pct = pct.strip() if pct.strip() in PERCENTILE_RANK else "P10"

    return filter_cards(all_cards, min_length=min_len, max_length=max_len,
                        tier=tier_val, endings=endings, min_percentile=min_pct)


# ── CLI entry point ──

def parse_args():
    parser = argparse.ArgumentParser(
        description="Scrabble SRS Quiz",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument("--mode", choices=["review", "anagram", "hooks",
                                           "pattern", "morphology",
                                           "transformation", "extension",
                                           "reduction"],
                        default=None)
    parser.add_argument("--deck", type=str, default=None,
                        help="Preset deck name")
    parser.add_argument("--list-decks", action="store_true")
    parser.add_argument("--length", type=int, default=None)
    parser.add_argument("--tier", type=int, choices=[1, 2, 3, 4], default=None)
    parser.add_argument("--size", type=int, default=20, help="Session size")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--min-percentile", default="P25")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_decks:
        for name, desc in available_decks().items():
            print(f"  {name:16s}  {desc}")
        return

    progress = load_progress()

    if args.stats:
        show_stats(progress)
        return

    print("  Loading word database...", end=" ", flush=True)
    all_cards = load_word_analysis()
    print(f"{len(all_cards)} words loaded.")

    # If no mode specified, launch interactive menu
    if args.mode is None and args.deck is None:
        interactive_menu(args, all_cards, progress)
        return

    # Direct mode from CLI args
    if args.deck:
        if args.deck in VERB_PRESETS:
            verbs = load_verbs()
            pool_cards = apply_verb_preset(verbs, args.deck)
        else:
            pool_cards = apply_preset(all_cards, args.deck)
    else:
        kwargs = {"min_percentile": args.min_percentile}
        if args.length:
            kwargs["min_length"] = args.length
            kwargs["max_length"] = args.length
        if args.tier:
            kwargs["tier"] = args.tier
        pool_cards = filter_cards(all_cards, **kwargs)

    pool = [c["word"] for c in pool_cards]
    session_words = build_session(progress, pool, session_size=args.size)
    card_lookup = {c["word"]: c for c in pool_cards}
    session_cards = [card_lookup[w] for w in session_words if w in card_lookup]

    if not session_cards:
        print("  No cards available for this selection.")
        return

    mode = args.mode or "review"
    run_fn = {"review": run_review, "anagram": run_anagram,
              "hooks": run_hooks, "pattern": run_pattern,
              "morphology": run_morphology,
              "transformation": run_transformation,
              "extension": run_extension,
              "reduction": run_reduction}[mode]

    results = run_fn(session_cards, progress)
    save_progress(progress)
    show_summary(results)


if __name__ == "__main__":
    main()
