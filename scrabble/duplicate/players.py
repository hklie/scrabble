"""players.py — Player registry and scoring for Duplicate Scrabble."""

from dataclasses import dataclass, field


@dataclass
class Player:
    name: str
    round_scores: list = field(default_factory=list)
    round_plays: list = field(default_factory=list)  # play string per round

    @property
    def total_score(self):
        return sum(self.round_scores)


class PlayerRegistry:
    def __init__(self):
        self.players = []

    def register(self, name):
        player = Player(name=name)
        self.players.append(player)
        return player

    def get_leaderboard(self):
        """Return list of (rank, player) sorted by total score descending.

        Display uses anonymous labels: "Player 1", "Player 2", etc.
        Rank is by position (1-indexed), ties get same rank.
        """
        sorted_players = sorted(self.players, key=lambda p: p.total_score, reverse=True)
        leaderboard = []
        prev_score = None
        rank = 0
        for i, player in enumerate(sorted_players):
            if player.total_score != prev_score:
                rank = i + 1
                prev_score = player.total_score
            leaderboard.append((rank, player))
        return leaderboard
