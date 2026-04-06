# Duplicate Scrabble

Multiplayer Duplicate Scrabble game where all players see the same board and rack each round. Players compete to find the best play. Supports both terminal (CLI) and web-based (FastAPI + WebSocket) modes.

## How Duplicate Scrabble Works

Unlike standard Scrabble, in Duplicate mode:
- All players receive the **same rack** and see the **same board** each round.
- Players independently submit their best play within a time limit.
- Only the **best possible move** (the "Master" move) is placed on the board.
- Players are ranked by how close their plays are to the Master's score.

## Quick Start

### Web Server Mode (recommended)

Players join via phone/tablet by scanning a QR code or entering the server URL.

```bash
# Basic usage
python scrabble/duplicate/server.py scrabble/duplicate/dupli_config.txt

# With seed for reproducibility and custom title
python scrabble/duplicate/server.py scrabble/duplicate/dupli_config.txt \
    --seed 42 --title "Club Championship - Game 1"

# Custom port
python scrabble/duplicate/server.py scrabble/duplicate/dupli_config.txt --port 9000
```

1. Host opens `http://localhost:8000/host` on a laptop/projector.
2. Players open `http://<host-ip>:8000/play` on their phones.
3. Players enter the 4-digit room code and their name.
4. Host clicks "Iniciar Ronda" to start each round.

### CLI Mode

Terminal-based game where a moderator enters each player's play.

```bash
python scrabble/duplicate/main.py scrabble/duplicate/dupli_config.txt [--seed 42]
```

## Configuration

Edit `dupli_config.txt` to customize the game:

```
# Game title (optional)
title = CAMPEONATO MUNDIAL DE DUPLICADA - Partida 1

# Number of rounds (0 = unlimited, play until bag depleted)
rounds = 15

# Rack constraints: (min_vowels_and_consonants, until_round)
# Rounds 1-15: at least 2 vowels AND 2 consonants in rack
# Rounds 16-30: at least 1 of each
# After round 30: no constraint
constraints = (2,15),(1,30)

# Time per round (M:SS or seconds)
time = 3:00

# Export format: csv, excel, html, graphical
output = csv
```

| Option | Description |
|--------|-------------|
| `rounds` | Number of rounds. `0` = play until the tile bag is depleted |
| `constraints` | `(min, until_round)` pairs. Ensures racks have enough vowels and consonants |
| `time` | Countdown per round. Format: `M:SS` or raw seconds |
| `output` | Result export format: `csv`, `excel` (.xlsx), `html`, `graphical` (PNG chart) |
| `title` | Game title displayed on the host screen |

## Play Format

Players submit moves as `WORD POSITION`:
- `CORTES H8` — play CORTES starting at H8
- Position notation: `H8` (row-col) = horizontal, `8H` (col-row) = vertical
- **Blank tiles**: use lowercase for the blank letter. `CORTEs H8` means the S is a blank tile.

## Web Interface

### Host View (`/host`)
- 15x15 board with premium squares (TP, DP, TL, DL), green felt style
- Rack with point values on each tile
- Countdown timer (3x size), audible 3-beep warning at 30 seconds
- Real-time player count and submission tracking
- Round results: Master move + cumulative classification (top 20)
- Player plays are hidden to prevent strategy leaking
- End-of-game summary: master plays per round + top 10 final ranking
- QR code for easy player join

### Player View (`/play`)
- Join screen: room code + name
- Fullscreen enforcement (anti-cheat)
- Rack with point values and shuffle button
- Countdown timer, play input with blank confirmation overlay
- Round results: your score vs Master, leaderboard position
- Auto-reconnect on disconnection

### Anti-Cheat Features
- Tab-leave detection: only flags switches lasting 5+ seconds (ignores notifications, status bar)
- Grace period for fullscreen transitions to avoid false positives
- Fullscreen enforcement on mobile
- Wake Lock API keeps screen active
- Aggregate tab-violation count shown on host screen

## Module Structure

| File | Description |
|------|-------------|
| `server.py` | FastAPI web server with WebSocket endpoints for host and players |
| `engine.py` | Core game engine: board state, rack drawing, move generation, validation |
| `main.py` | CLI entry point for terminal-based play |
| `players.py` | Player registry and score tracking |
| `ui.py` | Terminal display utilities (board, rack, timer, results) |
| `export.py` | Multi-format result export (CSV, Excel, HTML, PNG chart) |
| `dupli_config.py` | Configuration file parser |
| `dupli_config.txt` | Example configuration |
| `static/host.html` | Host/moderator web interface |
| `static/player.html` | Player web interface (mobile-friendly, PWA-capable) |

## Game Engine Details

### Rack Drawing
`draw_tiles_constrained()` enforces minimum vowel/consonant counts per round. Blank tiles (`?`) can count toward either requirement. If constraints can't be met (bag too small), a warning is printed and the best possible rack is drawn.

### Move Generation
Uses the Appel-Jacobson algorithm with a trie built from the FISE2 lexicon (~639K words). All legal moves are generated and ranked by score. The highest-scoring move becomes the Master move.

### Move Validation
`validate_play(word, position, moves_df)` checks the player's submission against the full move list. Blank position matching is case-sensitive: uppercase = regular tile, lowercase = blank.

### Scoring
Standard Scrabble scoring with premium squares (Double/Triple Word, Double/Triple Letter) and a 50-point bingo bonus for using all 7 rack tiles.

## Export

Results are saved to `duplicate/resultados/` with timestamp:
- **CSV**: UTF-8 with BOM, player rows sorted by total score
- **Excel**: Formatted .xlsx with bold headers, colored Master row, auto-width columns
- **HTML**: Styled dark-theme table
- **Graphical**: PNG line chart showing cumulative score progression per player

## Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` | HTTP + WebSocket server |
| `uvicorn` | ASGI server |
| `websockets` | WebSocket protocol |
| `pandas` | Move DataFrames, result export |
| `openpyxl` | Excel export |
| `matplotlib` | Graphical export (PNG charts) |
