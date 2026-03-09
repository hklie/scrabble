"""main.py — CLI entry point for Duplicate Scrabble.

Usage:
    python scrabble/duplicate/main.py scrabble/duplicate/dupli_config.txt [--seed 42] [--output-dir Data/duplicate/]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scrabble.duplicate.dupli_config import parse_config, get_constraint_for_round
from scrabble.duplicate.engine import GameState, validate_play
from scrabble.duplicate.players import PlayerRegistry
from scrabble.duplicate.ui import (
    clear_screen, display_round_header, display_rack, display_board,
    run_countdown_timer, collect_player_plays, parse_play_input,
    display_round_results, display_leaderboard, display_game_over,
)
from scrabble.duplicate.export import export_results
from scrabble.config import BASE_PATH


def register_players():
    """Interactively register players."""
    registry = PlayerRegistry()

    while True:
        try:
            n = int(input("\n  How many players? "))
            if n < 1:
                print("  Need at least 1 player.")
                continue
            break
        except ValueError:
            print("  Enter a number.")

    for i in range(n):
        name = input(f"  Player {i + 1} name: ").strip()
        if not name:
            name = f"Player {i + 1}"
        registry.register(name)

    return registry


def run_game(config_path, seed=None, output_dir=None):
    """Main game loop."""
    config = parse_config(config_path)

    if output_dir is None:
        output_dir = os.path.join(BASE_PATH, 'duplicate')

    print(f"\n  Duplicate Scrabble")
    print(f"  Config: {config_path}")
    print(f"  Rounds: {'unlimited' if config.rounds == 0 else config.rounds}")
    print(f"  Time per play: {config.time_seconds // 60}:{config.time_seconds % 60:02d}")
    print(f"  Output: {config.output_format}")

    registry = register_players()
    player_names = [p.name for p in registry.players]

    game = GameState(seed=seed)

    round_num = 0
    max_rounds = config.rounds if config.rounds > 0 else 9999

    while round_num < max_rounds:
        round_num += 1

        # Get constraint for this round
        k = get_constraint_for_round(config, round_num)

        # Start round: draw rack, generate all moves
        if not game.start_round(k, k):
            print("\n  Game over: not enough tiles in the bag.")
            break

        # Display
        clear_screen()
        display_round_header(round_num,
                             config.rounds if config.rounds > 0 else 0,
                             len(game.bag))
        display_board(game.board)
        display_rack(game.rack)

        move_count = len(game.moves_df)
        print(f"  Valid moves: {move_count}")

        if move_count == 0:
            print("\n  No valid moves found. Game over.")
            break

        # Start countdown timer
        timer_expired, timer_stop = run_countdown_timer(config.time_seconds)

        # Wait for timer, then collect plays
        plays = collect_player_plays(player_names, timer_expired, timer_stop)

        # Validate each play
        player_results = []
        for name in player_names:
            play_str = plays.get(name, '')
            player = next(p for p in registry.players if p.name == name)

            if not play_str:
                player.round_scores.append(0)
                player.round_plays.append('PASS')
                player_results.append((name, '', 0, False))
            else:
                word, position = parse_play_input(play_str)
                if word and position:
                    valid, score = validate_play(word, position, game.moves_df)
                    if valid:
                        player.round_scores.append(score)
                        player.round_plays.append(play_str.upper())
                        player_results.append((name, play_str.upper(), score, True))
                    else:
                        player.round_scores.append(0)
                        player.round_plays.append(play_str.upper())
                        player_results.append((name, play_str.upper(), 0, False))
                else:
                    player.round_scores.append(0)
                    player.round_plays.append(play_str.upper())
                    player_results.append((name, play_str.upper(), 0, False))

        # Master plays best move
        master_score = game.best_move.score if game.best_move else 0
        game.apply_master_move()

        # Display results
        display_round_results(round_num, game.best_move, master_score, player_results)

        master_total = sum(game.master_scores)
        leaderboard = registry.get_leaderboard()
        display_leaderboard(leaderboard, master_total)

        input("\n  Press Enter to continue...")

    # Game over
    total_rounds = game.round_num
    master_total = sum(game.master_scores)
    leaderboard = registry.get_leaderboard()
    display_game_over(leaderboard, master_total, total_rounds)

    # Export results
    filepath = export_results(
        registry.players, game.master_scores, total_rounds,
        config.output_format, output_dir,
    )
    print(f"\n  Results exported to: {filepath}")


def main():
    parser = argparse.ArgumentParser(description='Duplicate Scrabble')
    parser.add_argument('config', help='Path to dupli_config.txt')
    parser.add_argument('--seed', '-s', type=int, help='Random seed')
    parser.add_argument('--output-dir', '-o', help='Output directory')
    args = parser.parse_args()

    run_game(args.config, seed=args.seed, output_dir=args.output_dir)


if __name__ == '__main__':
    main()
