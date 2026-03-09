"""dupli_config.py — Configuration parser for Duplicate Scrabble."""

from dataclasses import dataclass, field


@dataclass
class ConstraintRule:
    min_count: int
    until_round: int


@dataclass
class DupliConfig:
    rounds: int
    constraints: list = field(default_factory=list)  # list[ConstraintRule]
    time_seconds: int = 180
    output_format: str = 'csv'
    title: str = ''


def parse_config(filepath):
    """Parse a dupli_config.txt file into a DupliConfig.

    Format:
        rounds = 0
        constraints = (2,15),(1,30)
        time = 3:00
        output = csv
    """
    config = DupliConfig(rounds=0)

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip().lower()
            value = value.strip()

            if key == 'rounds':
                config.rounds = int(value)

            elif key == 'constraints':
                rules = []
                # Parse (k,until_round) pairs
                import re
                pairs = re.findall(r'\((\d+)\s*,\s*(\d+)\)', value)
                for min_count, until_round in pairs:
                    rules.append(ConstraintRule(
                        min_count=int(min_count),
                        until_round=int(until_round),
                    ))
                # Sort by until_round ascending
                rules.sort(key=lambda r: r.until_round)
                config.constraints = rules

            elif key == 'time':
                # Parse M:SS format
                parts = value.split(':')
                if len(parts) == 2:
                    config.time_seconds = int(parts[0]) * 60 + int(parts[1])
                else:
                    config.time_seconds = int(parts[0])

            elif key == 'output':
                config.output_format = value.lower()

            elif key == 'title':
                config.title = value

    return config


def get_constraint_for_round(config, round_num):
    """Return the minimum vowel/consonant count for a given round.

    Walks through constraint rules in order. After the last rule's
    until_round, constraint drops to 0.
    """
    for rule in config.constraints:
        if round_num <= rule.until_round:
            return rule.min_count
    return 0
