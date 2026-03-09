"""ui.py — Terminal UI for Duplicate Scrabble."""

import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from analyze_board import to_display
from autoplay_scrabble import print_board_text


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def display_round_header(round_num, total_rounds, bag_size):
    label = f"Round {round_num}"
    if total_rounds > 0:
        label += f" / {total_rounds}"
    print(f"\n{'=' * 60}")
    print(f"  {label}  |  Bag: {bag_size} tiles")
    print(f"{'=' * 60}")


def display_rack(rack):
    display = [to_display(t).upper() if t != '?' else '?' for t in rack]
    print(f"\n  Rack:  {' '.join(display)}")


def display_board(board):
    print()
    print_board_text(board)


def run_countdown_timer(seconds):
    """Run a background countdown timer. Returns (expired_event, stop_event).

    expired is set when time runs out. Set stop to cancel the timer early.
    """
    expired = threading.Event()
    stop = threading.Event()

    def _countdown():
        remaining = seconds
        while remaining > 0 and not stop.is_set():
            mins = remaining // 60
            secs = remaining % 60
            sys.stdout.write(f"\r  Time remaining: {mins}:{secs:02d}  ")
            sys.stdout.flush()
            stop.wait(timeout=1)
            remaining -= 1
        if not stop.is_set():
            sys.stdout.write("\r  TIME'S UP!                    \n")
            sys.stdout.flush()
            expired.set()
        else:
            sys.stdout.write("\r                                \r")
            sys.stdout.flush()

    t = threading.Thread(target=_countdown, daemon=True)
    t.start()
    return expired, stop


def collect_player_plays(player_names, timer_expired, timer_stop):
    """Wait for timer, then collect each player's play from moderator.

    Each play is entered as 'WORD POSITION' (e.g., 'DESIGNA H8').
    Empty input or 'PASS' means no play.

    Returns dict[name, play_str].
    """
    # Wait for the countdown to finish before prompting
    timer_expired.wait()
    timer_stop.set()

    print(f"\n{'─' * 40}")
    print("  Enter each player's play (WORD POSITION) or PASS:")
    print(f"{'─' * 40}")

    plays = {}
    for name in player_names:
        raw = input(f"  {name}: ").strip()
        if not raw or raw.upper() == 'PASS':
            plays[name] = ''
        else:
            plays[name] = raw
    return plays


def parse_play_input(play_str):
    """Split 'WORD POSITION' into (word, position). Returns ('', '') on invalid."""
    parts = play_str.strip().split()
    if len(parts) >= 2:
        return parts[0], parts[1]
    return '', ''


def display_round_results(round_num, master_move, master_score, player_results):
    """Show the master's play and each player's result for the round.

    player_results: list of (name, play_str, score, valid)
    """
    print(f"\n{'─' * 40}")
    print(f"  Round {round_num} Results")
    print(f"{'─' * 40}")

    if master_move:
        print(f"  Master: {master_move.word_display} {master_move.pos_str} "
              f"{master_move.dir_arrow}  —  {master_score} pts")
    else:
        print(f"  Master: no valid move  —  0 pts")

    print()
    # Sort by score descending
    sorted_results = sorted(player_results, key=lambda x: x[2], reverse=True)
    for name, play_str, score, valid in sorted_results:
        if not play_str:
            print(f"  {name}: PASS  —  0 pts")
        elif valid:
            print(f"  {name}: {play_str}  —  {score} pts")
        else:
            print(f"  {name}: {play_str}  —  INVALID (0 pts)")


def display_leaderboard(leaderboard, master_total):
    """Show cumulative standings with anonymous labels and % vs master.

    leaderboard: list of (rank, player)
    """
    print(f"\n{'─' * 40}")
    print(f"  Leaderboard")
    print(f"{'─' * 40}")
    print(f"  {'Master':20s}  {master_total:>5d} pts")
    print(f"  {'─' * 36}")

    for i, (rank, player) in enumerate(leaderboard):
        label = f"Player {i + 1}"
        pct = (player.total_score / master_total * 100) if master_total > 0 else 0
        print(f"  {rank}. {label:18s}  {player.total_score:>5d} pts  ({pct:5.1f}%)")


def display_game_over(leaderboard, master_total, total_rounds):
    """Show final standings."""
    print(f"\n{'=' * 60}")
    print(f"  GAME OVER — {total_rounds} rounds played")
    print(f"{'=' * 60}")
    print(f"\n  Master total: {master_total} pts")
    print()

    for i, (rank, player) in enumerate(leaderboard):
        label = f"Player {i + 1}"
        pct = (player.total_score / master_total * 100) if master_total > 0 else 0
        print(f"  {rank}. {label:18s}  {player.total_score:>5d} pts  ({pct:5.1f}%)")

    print(f"\n{'=' * 60}")
