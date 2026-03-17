# TODO — Roadmap to Make the Scrabble Tool More Appealing

> Target audience: **Spanish-speaking Scrabble players** (competitive and casual).
> All user-facing text, labels, instructions, and UI copy must be in **Spanish**.

---

## 1. Word Validity Lookup

**Goal:** Let users instantly check whether a word is valid in the FISE2 lexicon.

### Current Infrastructure

- A **trie** built from `LexiconFISE2.TXT` (~639K words) already exists in `analyze_board.py` (`build_trie()`, `_word_in_trie()`). It's cached as a pickle file for fast reloads.
- `preprocessing.tokenize_word()` handles digraph encoding (`ch`, `ll`, `rr`, `ñ`).
- The trie uses internal token encoding (`1`=ch, `2`=ll, `3`=rr, `4`=ñ) and a `TRIE_TERMINAL` sentinel to mark valid word endings.

### Implementation Plan

1. **Extract trie utilities into a shared module** (`scrabble/lexicon.py`):
   - Move `build_trie()`, `_build_trie_from_file()`, and `_word_in_trie()` out of `analyze_board.py`.
   - Expose a clean API:
     ```python
     def load_lexicon_trie() -> dict:
         """Load or build the FISE2 trie (pickle-cached)."""

     def is_valid_word(word: str, trie: dict) -> bool:
         """Check if a word exists in the lexicon. Handles digraphs transparently."""
     ```
   - `analyze_board.py` imports from this module instead of defining its own.

2. **Add to the CLI quiz** (`study/quiz.py`):
   - New option in the interactive menu: `c. Consultar palabra` (check word).
   - Flow: user types a word → system tokenizes it → checks trie → prints "VÁLIDA" / "NO VÁLIDA".
   - Show additional info for valid words: point value, token length, hooks, prefix/suffix (pull from `word_analysis.csv` if available, otherwise compute on the fly).

3. **Add to the web UI** (see Feature 5):
   - A search bar at the top of every page.
   - Real-time validation as the user types (debounced WebSocket message or REST endpoint).
   - Response includes: validity, point value, definition source link (if available), hooks.

### Notes

- The full FISE2 lexicon includes both verbs and non-verbs. The trie must be built from the complete lexicon, not from `No_verbos_filtrados.txt`.
- Consider adding fuzzy suggestions ("Did you mean...?") using edit-distance-1 candidates when a word is not found. The trie structure makes this efficient: traverse and try all single-token substitutions/insertions/deletions.

---

## 2. One-Letter Word Transformation

**Goal:** Given a word, generate all valid words formed by changing exactly one letter (token) to a different one.

### Current Infrastructure

- `study/chains.py` already implements `hamming(tokens1, tokens2)` for Hamming distance and `build_graph()` for adjacency at distance 1.
- The chain builder works on fixed-length word groups (6, 7, 8 tokens).
- All 28 tile types are listed in `config.ALL_TILES`.

### Implementation Plan

1. **Create a word transformation module** (`scrabble/study/transforms.py`):
   - Core function:
     ```python
     def one_letter_changes(word: str, trie: dict) -> list[dict]:
         """Return all valid words formed by changing one token in `word`.
         Each result: {'word': str, 'position': int, 'original': str, 'replacement': str, 'value': int}
         """
     ```
   - Algorithm: tokenize the input word. For each position, try all 28 tile types (from `config.ALL_TILES`). If the resulting token list exists in the trie and differs from the original, include it.
   - Complexity: `O(L * 28)` trie lookups per word, where L = token length. Very fast.

2. **Add as a quiz mode** (`study/quiz.py`):
   - New mode: **"Transformación"** — show a word and one position marked with `_`, the player must name a valid replacement letter that forms a new word.
   - Variant: show the word, player must list ALL valid one-letter transformations at any position (harder, scored by completeness like the hook quiz).
   - SRS integration: track words by how well the user knows their transformation neighbors.

3. **Add as a standalone CLI tool**:
   - `python -m study.transforms PALABRA` → prints all one-letter transformations grouped by position.
   - Useful for board analysis: "what words am I one tile away from?"

4. **Add to the web UI**:
   - Interactive explorer: click a letter position in the word → see all valid substitutions.
   - Visual: highlight the changed letter in a different color.

### Notes

- Skip identity replacements (same token at same position).
- Consider showing the point-value delta (new word value vs. original) to help players evaluate which transformations are strategically valuable.
- The existing `chains.py` graph builder can be refactored to use this function instead of its current O(n^2) word-pair comparison.

---

## 3. Word Extension with a Blank Tile

**Goal:** Given a word, find all valid words formed by inserting one additional letter at any position (before, between, or after existing letters).

### Current Infrastructure

- Hook data in `word_analysis.csv` already covers **prefix hooks** (letter before) and **suffix hooks** (letter after) for all 28 tile types.
- The trie supports validation of arbitrary token sequences.

### Implementation Plan

1. **Add to `study/transforms.py`**:
   ```python
   def insert_letter(word: str, trie: dict) -> list[dict]:
       """Return all valid words formed by inserting one token anywhere in `word`.
       Each result: {'word': str, 'position': int, 'inserted': str, 'value': int}
       """
   ```
   - Algorithm: tokenize the input. For each position 0..L (L+1 positions including before first and after last), try inserting each of the 28 tile types. Validate the resulting L+1 token sequence against the trie.
   - Complexity: `O((L+1) * 28)` trie lookups. Still very fast.

2. **Add as a quiz mode**:
   - Mode: **"Extensión"** — show a word and a blank slot at a random position, player names valid letters that can go there.
   - Variant: show the word with no position hint, player must find all extensions at all positions (advanced).
   - Integrates with SRS: harder extensions (rare letters, unusual positions) get reviewed more.

3. **Relation to existing hooks**:
   - Front hooks = insertions at position 0.
   - Back hooks = insertions at position L.
   - This feature generalizes hooks to **all positions**, which is new and powerful.
   - The hook quiz mode can be upgraded to test internal insertions too.

4. **Add to the web UI**:
   - Interactive: click between letters to see what can be inserted.
   - Show the resulting word highlighted with the new letter in a distinct color.

### Notes

- This is especially useful for competitive play: finding "extensions" of words already on the board.
- Consider also implementing **two-letter extensions** (insert two tiles) as an advanced option, though the search space grows to `O((L+2)^2 * 28^2)`.
- Group results by insertion position for readability.

---

## 4. Word Reduction (Remove One Letter)

**Goal:** Given a word, find all valid words formed by removing exactly one letter (token).

### Implementation Plan

1. **Add to `study/transforms.py`**:
   ```python
   def remove_letter(word: str, trie: dict) -> list[dict]:
       """Return all valid words formed by removing one token from `word`.
       Each result: {'word': str, 'position': int, 'removed': str, 'value': int}
       """
   ```
   - Algorithm: tokenize the input (length L). For each position 0..L-1, create a new token list with that position removed (length L-1). Check if it's valid in the trie.
   - Complexity: `O(L)` trie lookups. Trivially fast.

2. **Add as a quiz mode**:
   - Mode: **"Reducción"** — show a word, player must identify which letter(s) can be removed to leave a valid word.
   - Score by completeness (like hooks): finding all removable positions = quality 5, missing some = lower quality.

3. **Add to the web UI**:
   - Click a letter → system shows whether removing it produces a valid word.
   - Visual: strikethrough the removed letter, show the resulting word.

4. **Combine with Feature 3 for round-trip learning**:
   - Given word A, show that inserting X at position P gives word B; and that removing position P from B gives back A.
   - This bidirectional view helps players see word families as connected graphs.

### Notes

- This is the inverse of Feature 3. If "inserting S after AREA gives AREAS", then "removing position 4 from AREAS gives AREA".
- Especially valuable for defensive play: knowing which subwords exist inside your opponent's words.
- Group results: show the removed letter and resulting word side by side.

---

## 5. Web-Based Study UI

**Goal:** Deploy the entire study/quiz system as a web application accessible from any device (phone, tablet, laptop) via a browser.

### Current Infrastructure

- **FastAPI + WebSocket** stack already runs in `duplicate/server.py`.
- **uvicorn** ASGI server with static file serving.
- **HTML/JS/CSS** front-end pattern established in `duplicate/static/` (host.html, player.html).
- **SRS engine** (`srs.py`) and **deck system** (`decks.py`) are pure Python with no terminal dependencies.
- **Quiz modes** (`quiz.py`) use `input()` + `print()` — need to be decoupled from terminal I/O.

### Architecture Plan

```
scrabble/
├── web/                          # New web application
│   ├── app.py                    # FastAPI app: REST + WebSocket endpoints
│   ├── api.py                    # REST routes: /api/validate, /api/transform, etc.
│   ├── quiz_ws.py                # WebSocket handler for quiz sessions
│   ├── static/
│   │   ├── index.html            # Landing page / dashboard
│   │   ├── quiz.html             # Quiz session page
│   │   ├── explorer.html         # Word explorer (validate, transform, extend, reduce)
│   │   ├── css/
│   │   │   └── style.css         # Responsive styles (mobile-first)
│   │   └── js/
│   │       ├── quiz.js           # Quiz session logic (WebSocket client)
│   │       ├── explorer.js       # Word explorer client
│   │       └── common.js         # Shared utilities (fetch, display)
│   └── templates/                # Jinja2 templates (if needed for SSR)
```

### Implementation Phases

#### Phase 1: Backend API

1. **REST endpoints** (`web/api.py`):
   - `GET /api/validar/{word}` — word validity + metadata (points, hooks, prefix/suffix).
   - `GET /api/transformar/{word}` — one-letter changes (Feature 2).
   - `GET /api/extender/{word}` — insert-letter results (Feature 3).
   - `GET /api/reducir/{word}` — remove-letter results (Feature 4).
   - `GET /api/mazos` — list available study decks with word counts.
   - `GET /api/progreso` — SRS progress stats.
   - `GET /api/mazo/{name}` — get cards for a specific deck.

2. **WebSocket endpoint** (`web/quiz_ws.py`):
   - `WS /ws/quiz` — real-time quiz session.
   - Protocol:
     ```json
     Client → Server:
       {"type": "start", "mode": "anagram", "deck": "words-5", "size": 20}
       {"type": "answer", "card_index": 3, "answer": "palabra"}
       {"type": "rate", "card_index": 3, "quality": 4}

     Server → Client:
       {"type": "card", "index": 3, "total": 20, "prompt": {...}}
       {"type": "result", "correct": true, "quality": 5, "reveal": {...}}
       {"type": "summary", "reviewed": 20, "avg_quality": 3.8, "struggling": [...]}
     ```

3. **Refactor quiz logic**:
   - Extract the core logic of each quiz mode from `quiz.py` into pure functions that take input and return output (no `input()`/`print()`/`clear_screen()`).
   - Create `study/quiz_engine.py` with:
     ```python
     def generate_anagram_prompt(card) -> dict
     def check_anagram_answer(card, answer) -> dict
     def generate_hook_prompt(card) -> dict
     def check_hook_answer(card, given_hooks) -> dict
     # ... etc for each mode
     ```
   - Both `quiz.py` (CLI) and `quiz_ws.py` (web) call these same functions.

#### Phase 2: Frontend UI

1. **Landing page** (`index.html`):
   - Dashboard showing: words studied, due today, mastered count, streak.
   - Quick access to: "Iniciar repaso" (start review), "Explorar palabras" (word explorer), deck browser.
   - All text in Spanish.

2. **Quiz page** (`quiz.html`):
   - Deck selector: categorized list (por longitud, por terminación, verbos, patrones vocálicos).
   - Mode selector: Repaso, Anagrama, Ganchos, Patrón, Morfología, Transformación, Extensión, Reducción.
   - Session size slider (10–50).
   - Card display area:
     - Large word display with tile-style letter rendering.
     - Input area (text input or letter grid for mobile).
     - Quality rating buttons (0–5) with Spanish labels: Nulo / Error / Difícil / Correcto / Bien / Fácil.
     - Reveal panel: hooks, prefix/suffix, verb type, anagrams, point value.
   - Session progress bar and end-of-session summary.

3. **Word explorer** (`explorer.html`):
   - Central search bar: "Escribe una palabra..."
   - Real-time validation as user types.
   - For valid words, tabbed results panel:
     - **Info**: length, points, percentile, prefix, suffix, hooks, anagrams.
     - **Transformaciones**: one-letter changes grouped by position.
     - **Extensiones**: insertions grouped by position.
     - **Reducciones**: valid sub-words from removing one letter.
   - For invalid words: "No válida" + suggested corrections (edit-distance-1 valid words).

4. **Responsive design**:
   - Mobile-first CSS (phone is the primary study device).
   - Touch-friendly buttons and inputs.
   - Letter tiles rendered as styled `<span>` elements resembling Scrabble tiles (beige background, point value subscript).
   - Support landscape and portrait orientations.

#### Phase 3: Deployment

1. **Single-server deployment**:
   - One FastAPI app serves both the quiz web UI and the duplicate game.
   - Mount quiz at `/estudio/` and duplicate game at `/duplicada/`.
   - `python -m web.app --port 8000` starts everything.

2. **Data initialization**:
   - On first launch, build trie (cached as pickle), load CSV data.
   - Progress stored per-user. Initially support single-user (local `progress.json`). Later: user accounts with server-side storage.

3. **LAN deployment** (for study groups / clubs):
   - Same pattern as the duplicate game: host runs server, players connect via phone.
   - QR code on landing page with the server URL for easy mobile access.
   - Optional: room codes for multiplayer quiz races (who can solve the anagram fastest).

4. **Cloud deployment** (future):
   - Containerize with Docker (Dockerfile + docker-compose).
   - SQLite or PostgreSQL for multi-user progress persistence.
   - Deploy to a VPS or cloud service (Fly.io, Railway, etc.).
   - Add user authentication (simple username/password or OAuth).

### UI Language & Localization

All user-facing strings must be in **Spanish**. Key translations:

| English | Spanish |
|---------|---------|
| Review | Repaso |
| Anagram | Anagrama |
| Hooks | Ganchos |
| Pattern | Patrón |
| Morphology | Morfología |
| Transformation | Transformación |
| Extension | Extensión |
| Reduction | Reducción |
| Valid | Válida |
| Invalid | No válida |
| Front hooks | Ganchos delanteros |
| Back hooks | Ganchos traseros |
| Prefix | Prefijo |
| Suffix | Sufijo |
| Ending | Terminación |
| Points | Puntos |
| Letters | Letras |
| Difficulty | Dificultad |
| Progress | Progreso |
| Due today | Pendientes hoy |
| Mastered | Dominadas |
| Struggling | En dificultad |
| Start session | Iniciar sesión |
| Quit | Salir |
| Correct | Correcto |
| Wrong | Incorrecto |
| Reveal | Revelar |
| Next | Siguiente |
| Session summary | Resumen de sesión |
| Word explorer | Explorador de palabras |
| Check word | Consultar palabra |

---

## Implementation Priority

| Priority | Feature | Effort | Dependencies |
|----------|---------|--------|-------------|
| **1** | Word validity lookup (Feature 1) | Small | Extract trie into shared module |
| **2** | Transformation module (Features 2+3+4) | Medium | Feature 1 (shared trie) |
| **3** | New quiz modes for transforms | Medium | Feature 2 |
| **4** | Web backend API | Medium | Features 1–3 |
| **5** | Web frontend UI | Large | Feature 4 |
| **6** | Mobile-responsive design | Medium | Feature 5 |
| **7** | Multiplayer / cloud deployment | Large | Feature 5 |

Features 1–3 are independent of the web UI and can be used immediately via the CLI quiz. Feature 5 is the largest effort but has the highest impact on accessibility and appeal.
