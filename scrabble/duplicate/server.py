"""server.py — FastAPI + WebSocket server for Duplicate Scrabble.

Usage:
    python scrabble/duplicate/server.py scrabble/duplicate/dupli_config.txt [--seed 42] [--port 8000]

Players join via http://<host-ip>:<port>/play and enter the room code.
Host view at http://<host-ip>:<port>/host
"""

import argparse
import asyncio
import json
import os
import random
import string
import sys
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..'))           # scrabble/
sys.path.insert(0, os.path.join(_here, '..', '..'))     # project root

from scrabble.duplicate.dupli_config import parse_config, get_constraint_for_round
from scrabble.duplicate.engine import GameState, validate_play
from scrabble.analyze_board import to_display

STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ── Global game state ────────────────────────────────────────────────────────

game_lock = asyncio.Lock()

# Filled in at startup
config = None
game: GameState = None
room_code: str = ""
game_title: str = ""

# Connected clients
host_ws: WebSocket = None
player_connections: dict[str, WebSocket] = {}   # name -> ws

# Per-round state
round_active = False
round_timer_task = None
round_submissions: dict[str, dict] = {}         # name -> {play, timestamp}
round_tab_flags: dict[str, list] = {}           # name -> list of leave timestamps


def generate_room_code():
    return ''.join(random.choices(string.digits, k=4))


def board_to_json(board):
    """Convert 15x15 board to serializable list of lists."""
    result = []
    for r in range(15):
        row = []
        for c in range(15):
            if board[r][c] is None:
                row.append("")
            else:
                tile, is_blank = board[r][c]
                display = to_display(tile)
                row.append(display.lower() if is_blank else display.upper())
        result.append(row)
    return result


def rack_to_json(rack):
    return [to_display(t).upper() if t != '?' else '?' for t in rack]


async def broadcast_to_players(msg: dict):
    """Send a message to all connected players."""
    text = json.dumps(msg)
    disconnected = []
    for name, ws in player_connections.items():
        try:
            await ws.send_text(text)
        except Exception:
            disconnected.append(name)
    for name in disconnected:
        player_connections.pop(name, None)


async def send_to_host(msg: dict):
    global host_ws
    if host_ws:
        try:
            await host_ws.send_text(json.dumps(msg))
        except Exception:
            host_ws = None


async def update_host_players():
    """Send current player list to host."""
    names = list(player_connections.keys())
    await send_to_host({"type": "players", "names": names})


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return HTMLResponse("<h2>Scrabble Duplicado</h2>"
                        "<p><a href='/host'>Moderador</a> | "
                        "<a href='/play'>Jugador</a></p>")


@app.get("/manifest.json")
async def manifest():
    from fastapi.responses import JSONResponse
    title = game_title or "Scrabble Duplicado"
    return JSONResponse({
        "name": title,
        "short_name": "Duplicado",
        "start_url": "/play",
        "display": "standalone",
        "background_color": "#1a1a2e",
        "theme_color": "#1a1a2e",
    })


@app.get("/host")
async def host_page():
    return FileResponse(os.path.join(STATIC_DIR, "host.html"))


@app.get("/play")
async def player_page():
    return FileResponse(os.path.join(STATIC_DIR, "player.html"))


# ── Host WebSocket ────────────────────────────────────────────────────────────

@app.websocket("/ws/host")
async def ws_host(ws: WebSocket):
    global host_ws
    await ws.accept()
    host_ws = ws

    # Send initial state
    await ws.send_text(json.dumps({
        "type": "init",
        "room_code": room_code,
        "title": game_title,
        "config": {
            "rounds": config.rounds,
            "time_seconds": config.time_seconds,
        },
    }))
    await update_host_players()

    try:
        while True:
            data = json.loads(await ws.receive_text())
            cmd = data.get("cmd")

            if cmd == "start_round":
                await start_round()
    except WebSocketDisconnect:
        host_ws = None


# ── Player WebSocket ─────────────────────────────────────────────────────────

@app.websocket("/ws/player")
async def ws_player(ws: WebSocket):
    await ws.accept()
    name = None

    try:
        while True:
            data = json.loads(await ws.receive_text())
            cmd = data.get("cmd")

            if cmd == "join":
                code = data.get("code", "")
                proposed_name = data.get("name", "").strip()

                if code != room_code:
                    await ws.send_text(json.dumps({
                        "type": "error", "msg": "Invalid room code."
                    }))
                    continue

                if not proposed_name:
                    await ws.send_text(json.dumps({
                        "type": "error", "msg": "Name cannot be empty."
                    }))
                    continue

                if proposed_name in player_connections:
                    await ws.send_text(json.dumps({
                        "type": "error", "msg": "Name already taken."
                    }))
                    continue

                name = proposed_name
                player_connections[name] = ws
                await ws.send_text(json.dumps({
                    "type": "joined", "name": name, "title": game_title
                }))
                await update_host_players()

            elif cmd == "submit":
                if not name or not round_active:
                    continue
                if name in round_submissions:
                    await ws.send_text(json.dumps({
                        "type": "error", "msg": "Already submitted."
                    }))
                    continue

                play = data.get("play", "").strip()
                round_submissions[name] = {
                    "play": play,
                    "timestamp": time.time(),
                }
                await ws.send_text(json.dumps({
                    "type": "submitted"
                }))
                # Notify host
                await send_to_host({
                    "type": "player_submitted",
                    "name": name,
                    "count": len(round_submissions),
                    "total": len(player_connections),
                })

            elif cmd == "tab_leave":
                if name and round_active:
                    round_tab_flags.setdefault(name, []).append(time.time())
                    await send_to_host({
                        "type": "tab_warning",
                        "name": name,
                        "count": len(round_tab_flags.get(name, [])),
                    })

    except WebSocketDisconnect:
        if name:
            player_connections.pop(name, None)
            await update_host_players()


# ── Game logic ────────────────────────────────────────────────────────────────

async def start_round():
    global round_active, round_timer_task

    async with game_lock:
        round_num = game.round_num + 1
        k = get_constraint_for_round(config, round_num)

        if not game.start_round(k, k):
            await send_to_host({"type": "game_over", "reason": "Not enough tiles."})
            await broadcast_to_players({"type": "game_over"})
            return

        round_submissions.clear()
        round_tab_flags.clear()
        round_active = True

        board = board_to_json(game.board)
        rack = rack_to_json(game.rack)
        move_count = len(game.moves_df)

        round_data = {
            "type": "round_start",
            "round": game.round_num,
            "total_rounds": config.rounds if config.rounds > 0 else 0,
            "bag_size": len(game.bag),
            "board": board,
            "rack": rack,
            "move_count": move_count,
            "time_seconds": config.time_seconds,
        }

        await send_to_host(round_data)
        await broadcast_to_players(round_data)

        if move_count == 0:
            round_active = False
            await send_to_host({"type": "game_over", "reason": "No valid moves."})
            await broadcast_to_players({"type": "game_over"})
            return

    # Start timer
    round_timer_task = asyncio.create_task(
        _run_timer(config.time_seconds, game.round_num)
    )


async def _run_timer(seconds, round_num):
    """Countdown timer. When it expires, evaluate the round."""
    await asyncio.sleep(seconds)
    # Only evaluate if we're still on the same round
    if game.round_num == round_num and round_active:
        await evaluate_round()


async def evaluate_round():
    global round_active

    async with game_lock:
        round_active = False

        player_results = []
        for name, ws in player_connections.items():
            sub = round_submissions.get(name)
            play_str = sub["play"] if sub else ""
            tab_leaves = len(round_tab_flags.get(name, []))

            if not play_str or play_str.upper() == "PASS":
                player_results.append({
                    "name": name, "play": "PASS", "score": 0,
                    "valid": True, "tab_leaves": tab_leaves,
                })
            else:
                parts = play_str.strip().split()
                if len(parts) >= 2:
                    word, position = parts[0], parts[1]
                    valid, score = validate_play(word, position, game.moves_df)
                    player_results.append({
                        "name": name, "play": play_str.upper(),
                        "score": score, "valid": valid,
                        "tab_leaves": tab_leaves,
                    })
                else:
                    player_results.append({
                        "name": name, "play": play_str.upper(),
                        "score": 0, "valid": False,
                        "tab_leaves": tab_leaves,
                    })

        # Master move
        master_play = ""
        master_score = 0
        if game.best_move:
            master_play = (f"{game.best_move.word_display} "
                           f"{game.best_move.pos_str} {game.best_move.dir_arrow}")
            master_score = game.best_move.score
        game.apply_master_move()

        # Build leaderboard (cumulative). We track scores in a simple dict.
        # We'll use the round_submissions history approach — store on game object.
        if not hasattr(game, '_player_scores'):
            game._player_scores = {}
        for pr in player_results:
            game._player_scores.setdefault(pr["name"], 0)
            game._player_scores[pr["name"]] += pr["score"]

        master_total = sum(game.master_scores)
        leaderboard = sorted(
            [{"name": n, "total": s,
              "pct": round(s / master_total * 100, 1) if master_total > 0 else 0}
             for n, s in game._player_scores.items()],
            key=lambda x: x["total"], reverse=True,
        )

        results = {
            "type": "round_results",
            "round": game.round_num,
            "master_play": master_play,
            "master_score": master_score,
            "master_total": master_total,
            "player_results": player_results,
            "leaderboard": leaderboard,
            "board": board_to_json(game.board),
        }

        await send_to_host(results)
        # Send each player their own result
        for pr in player_results:
            pname = pr["name"]
            ws = player_connections.get(pname)
            if ws:
                personal = dict(results)
                personal["your_result"] = pr
                try:
                    await ws.send_text(json.dumps(personal))
                except Exception:
                    pass


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global config, game, room_code, game_title

    parser = argparse.ArgumentParser(description='Duplicate Scrabble Web Server')
    parser.add_argument('config', help='Path to dupli_config.txt')
    parser.add_argument('--seed', '-s', type=int, help='Random seed')
    parser.add_argument('--port', '-p', type=int, default=8000, help='Server port')
    parser.add_argument('--title', '-t',
                        help='Custom game title (text or path to .txt file)')
    args = parser.parse_args()

    config = parse_config(args.config)
    game = GameState(seed=args.seed)
    room_code = generate_room_code()

    # Custom title: if arg is a file path, read its first line; otherwise use as text
    if args.title:
        if os.path.isfile(args.title):
            with open(args.title, 'r', encoding='utf-8') as f:
                game_title = f.readline().strip()
        else:
            game_title = args.title

    display_title = game_title or "Scrabble Duplicado"
    print(f"\n  {display_title} — Servidor Web")
    print(f"  Sala: {room_code}")
    print(f"  Moderador: http://0.0.0.0:{args.port}/host")
    print(f"  Jugador: http://0.0.0.0:{args.port}/play")
    print()

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
