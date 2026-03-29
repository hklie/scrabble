# Spanish Scrabble Word Study Tool

A Python toolkit for analyzing, filtering, ranking, and organizing a Spanish-language Scrabble lexicon into study materials. It includes a board image analyzer that uses OCR to read board and rack photos, then finds the highest-scoring plays using the Appel-Jacobson algorithm. It handles Spanish digraphs (`ch`, `ll`, `rr`) via an internal encoding system and produces categorized word lists, probability rankings, CSV exports, and flashcard-style transformation chains.

## Project Structure

```
scrabble/
├── Data/                          # Input lexicon files and all generated output
│   ├── Lexicon.TXT                # Complete Spanish Scrabble lexicon (~639K words)
│   ├── LexiconFISE2.TXT           # Official FISE2 lexicon
│   ├── No_verbos.txt              # Raw non-verb word list (latin-1 encoded)
│   ├── Verbos.txt                 # Raw verb list (latin-1 encoded)
│   ├── Master Copy/
│   │   └── Verbos_clasificados.TXT  # Verb classification data
│   ├── No_verbos_filtrados.txt    # [generated] Cleaned non-verb word list
│   ├── Ranked_Scrabble_Suggestions.txt  # [generated] Probability-ranked words
│   ├── Optimized_Study_List.txt   # [generated] Tiered study list
│   ├── word_analysis.csv          # [generated] Full word metadata CSV
│   ├── verbs.csv                  # [generated] Verb metadata CSV
│   ├── chains_study_list.txt      # [generated] Transformation chains
│   ├── synergy.csv                # [generated] Rack leave synergy values
│   ├── progress.json              # [generated] SRS quiz progress (per-word review state)
│   ├── ends_with_*.txt            # [generated] Words grouped by ending
│   ├── prefix_*.txt               # [generated] Words grouped by prefix
│   ├── suffix_*.txt               # [generated] Words grouped by suffix
│   ├── pattern_*.txt              # [generated] Words matching occurrence patterns
│   ├── words_only_TIER_*.txt      # [generated] Words filtered by consonant tier
│   ├── singleton_anagrams_*.txt   # [generated] Words with no anagrams
│   ├── five_token_ending_*.txt    # [generated] 5-token words by ending
│   └── six_token_ending_*.txt     # [generated] 6-token words by ending
│
├── boards/                        # Board and rack images for analyze_board.py
│   ├── Board1.jpg ... Board5.jpg  # Board photos
│   └── rack2.jpg ... rack5.jpg    # Rack photos
│
├── scrabble/                      # Source code package
│   ├── config.py                  # Central configuration (paths, constants, patterns)
│   ├── preprocessing.py           # Digraph tokenization/detokenization
│   ├── lexicon.py                 # Shared FISE2 trie: word validation, lookup, point values
│   ├── analyze_board.py           # Board image OCR + best move finder (→ migrating to play/)
│   ├── autoplay_scrabble.py       # Solitaire autoplay engine + board image renderer (→ migrating to play/)
│   │
│   ├── play/                      # Board training: analyze positions, evaluate moves, rack leave (planned)
│   │   └── TODO.md                # Roadmap: reorganize + interactive training + web UI
│   │
│   ├── study/                     # Word list generation, analysis & quiz tools
│   │   ├── quiz.py                # Interactive SRS quiz (8 modes + word lookup)
│   │   ├── srs.py                 # SM-2 spaced repetition engine + JSON persistence
│   │   ├── decks.py               # Deck generation, presets, and filters from CSV data
│   │   ├── transforms.py          # Word transformations: change, insert, remove one letter
│   │   ├── clean_no_verbs.py      # Lexicon cleaning and deduplication
│   │   ├── probability.py         # Probabilistic word ranking
│   │   ├── study_list.py          # Tiered study list generation
│   │   ├── generator.py           # Interactive word generator from letter constraints
│   │   ├── nouns_csv.py           # Comprehensive word analysis CSV export (28 hook columns)
│   │   ├── verbs_csv.py           # Verb classification CSV export
│   │   ├── vowel_patterns.py      # 7-letter word vowel/consonant pattern filters
│   │   ├── endings.py             # Filter words by ending letter
│   │   ├── prefixes.py            # Filter words by prefix pattern
│   │   ├── suffixes.py            # Filter words by suffix pattern
│   │   ├── ocurrences.py          # Filter words by internal letter patterns
│   │   ├── filter_by_tiers.py     # Filter words by consonant difficulty tiers
│   │   ├── unique_anagrams.py     # Find words with no anagrams in the lexicon
│   │   ├── endings_with_useful_plurals.py  # Targeted length+ending combinations
│   │   ├── chains.py              # Word transformation chain builder
│   │   └── synergy.py             # Rack leave synergy value computation
│   │
│   └── duplicate/                 # Duplicate Scrabble game (CLI + web)
│       ├── dupli_config.py        # Game configuration parser
│       ├── dupli_config.txt       # Example configuration file
│       ├── engine.py              # Core game engine (rack draw, move gen, scoring)
│       ├── server.py              # FastAPI + WebSocket web server
│       ├── main.py                # CLI entry point
│       ├── players.py             # Player registry and scoring
│       ├── ui.py                  # Terminal UI
│       ├── export.py              # CSV/Excel/HTML/PNG result export
│       ├── static/                # Web assets (host.html, player.html)
│       └── resultados/            # Game results output directory
│
├── requirements.txt               # Python dependencies
└── README.md
```

## Dependencies

Install all dependencies with:
```bash
pip install -r requirements.txt
```

### External

| Package | Used by | Purpose |
|---------|---------|---------|
| `numpy` | `nouns_csv.py`, `analyze_board.py` | Percentile calculations; image array operations |
| `opencv-python` | `analyze_board.py` | Image processing, thresholding, cell extraction |
| `easyocr` | `analyze_board.py` | Optical character recognition (Spanish model) |
| `regex` | `chains.py` | Advanced regex support |
| `Pillow` | `autoplay_scrabble.py` | Board image rendering |
| `pandas` | `duplicate/engine.py`, `duplicate/export.py` | Move DataFrames and result export |
| `openpyxl` | `duplicate/export.py` | Excel export (.xlsx) |
| `matplotlib` | `duplicate/export.py` | Graphical score progression charts |
| `fastapi` | `duplicate/server.py` | HTTP + WebSocket server |
| `uvicorn` | `duplicate/server.py` | ASGI server |
| `websockets` | `duplicate/server.py` | WebSocket protocol support |

**Note:** `numpy` must stay <2 for torchvision/scikit-image compatibility.

### Internal

All modules depend on:

- **`config.py`** -- file paths, Scrabble tile/point data, digraph mappings, pattern definitions, tier groupings, premium square map, board analysis constants.
- **`preprocessing.py`** -- `tokenize_word()` and `detokenize_word()` for digraph-aware text handling.
- **`lexicon.py`** -- shared FISE2 trie: `load_lexicon_trie()`, `is_valid_word()`, `word_value()`, `build_trie()`, `_word_in_trie()`. Used by `analyze_board.py`, `study/quiz.py`, and `study/transforms.py`.

## Digraph Encoding

Spanish Scrabble treats `ch`, `ll`, and `rr` as single tiles. The raw lexicon files encode them as digits:

| Digit | Digraph |
|-------|---------|
| `1`   | `ch`    |
| `2`   | `ll`    |
| `3`   | `rr`    |

`preprocessing.tokenize_word()` converts human-readable text (e.g., `"calle"`) into a token list (`['c', 'a', 'll', 'e']`) by detecting digraphs. `detokenize_word()` reverses the process. The `DIGRAPH_MAP` in config maps the digit codes to digraph strings and vice versa.

## Data Flow

```
                         ┌──────────────────┐
                         │  Raw Lexicon      │
                         │  No_verbos.txt    │
                         │  (latin-1)        │
                         └────────┬─────────┘
                                  │
                         study/clean_no_verbs.py
                         (decode digraphs, remove
                          digits, deduplicate, sort)
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │  No_verbos_filtrados.txt     │
                    │  (cleaned non-verb lexicon)  │
                    └──────────────┬──────────────┘
                                   │
          ┌────────────┬───────────┼───────────┬────────────┬───────────┬───────────┐
          │            │           │           │            │           │           │
          ▼            ▼           ▼           ▼            ▼           ▼           ▼
  probability.py  study_list.py  nouns_csv.py  Pattern    Anagram   chains.py  synergy.py
                                               Filters    Analysis
          │            │           │           │            │           │           │
          ▼            ▼           ▼           │            ▼           ▼           ▼
     Ranked_       Optimized_   word_         │      singleton_    chains_     synergy.
     Scrabble_     Study_       analysis.     │      anagrams_     study_      csv
     Suggestions   List.txt     csv           │      *.txt         list.txt
     .txt                                     │
                                              ├──► ends_with_*.txt
                                              ├──► prefix_*.txt
                                              ├──► suffix_*.txt
                                              ├──► pattern_*.txt
                                              ├──► words_only_TIER_*.txt
                                              └──► five/six_token_ending_*.txt


    ┌──────────────────┐     ┌─────────────────────────────┐
    │  Verbos.txt      │     │  Verbos_clasificados.TXT    │
    │  (verb list)     │     │  (verb type classifications)│
    └────────┬─────────┘     └──────────────┬──────────────┘
             │                              │
             └──────────┬───────────────────┘
                        │
               study/verbs_csv.py
                        │
                        ▼
                    verbs.csv
```

## File and Function Reference

### config.py

Central configuration hub. No functions -- exposes constants used by all other modules.

| Constant | Description |
|----------|-------------|
| `BASE_PATH` | Root path to the `Data/` directory |
| `LEXICON`, `VERBS`, `CATEGORIZED_VERBS`, `NO_VERBS_FILE`, `CLEAN_NO_VERBS_FILE`, `RANKED_SCRABBLE_WORDS`, `OUTPUT_STUDY_LIST` | File paths for inputs and outputs |
| `DIGRAPHS` / `DIGRAPH_MAP` | Bidirectional digit-to-digraph mapping |
| `SCRABBLE_TILES` | Number of tiles available per letter in Spanish Scrabble |
| `SCRABBLE_POINTS` | Point value per letter/digraph |
| `RARE_LETTERS` | Set of low-frequency, high-value letters |
| `TIER_1` through `TIER_4` | Consonant groupings by difficulty (common to rare) |
| `PREFIXES`, `SUFFIXES`, `ENDINGS`, `OCURRENCES` | Linguistic pattern sets for filtering |
| `EXTENSIVE_PREFIXES`, `EXTENSIVE_SUFIXES` | Extended pattern sets used by `nouns_csv.py` |
| `BOARDS_PATH` | Path to board/rack image directory |
| `PREMIUM_SQUARES` | Standard 15×15 premium square map (TW, DW, TL, DL) |
| `ALL_TILES` | All 28 playable tile types in internal representation |
| `INTERNAL_POINTS` | Point values with digraphs using encoded keys (`1`/`2`/`3`) |
| `POINTS_TO_TILES` | Reverse lookup: point value → set of tiles with that value |
| `TOTAL_BLANKS`, `TOTAL_TILES` | Tile bag totals (2 blanks, 100 total) |

### preprocessing.py

Foundation for all word manipulation across the project.

| Function | Signature | Description |
|----------|-----------|-------------|
| `tokenize_word` | `(word: str) -> list[str]` | Splits a word into tokens, detecting digraphs (`ch`, `ll`, `rr`) from their human-readable form and mapping them to internal codes |
| `detokenize_word` | `(tokens: list[str]) -> str` | Reconstructs a readable word from a token list, expanding digit codes back to digraph strings |

### lexicon.py

Shared FISE2 trie utilities. Extracted from `analyze_board.py` to provide a clean API for word validation across all modules.

| Function | Signature | Description |
|----------|-----------|-------------|
| `load_lexicon_trie` | `() -> dict` | Load or build the FISE2 trie (pickle-cached). Cached in memory after first call. |
| `is_valid_word` | `(word: str, trie=None) -> bool` | Check if a word exists in the FISE2 lexicon. Handles digraphs transparently. |
| `word_value` | `(word: str) -> int` | Compute the point value of a word. |
| `build_trie` | `(lexicon_path: str) -> dict` | Build/load a pickle-cached trie from any lexicon file. |
| `_word_in_trie` | `(trie: dict, tokens: list) -> bool` | Validate that a token list exists in a trie. |

### analyze_board.py

Board image analyzer and best move finder. Reads a photograph of a Spanish Scrabble board, identifies all tiles via OCR, and finds the highest-scoring legal plays using the Appel-Jacobson algorithm with a trie built from the FISE2 lexicon.

**Architecture:**

The module is organized in four parts:

- **Part A — OCR:** Grayscale tile detection with subscript-based validation. Each real tile has a point-value subscript digit; the module classifies tiles by subscript size (`1pt`, `multi`, `10pt`) to disambiguate OCR confusions (N/Ñ, R/RR, L/LL, Z). Blank tiles are detected via red-ink ratio. Post-OCR validation corrects bag-limit violations (e.g., excess Ñ → N).
- **Part B — Trie & Cross-checks:** Uses the shared trie from `lexicon.py` (FISE2 lexicon, cached as pickle). Computes per-cell cross-check sets for perpendicular word validation.
- **Part C — Scoring:** Computes move scores with premium squares (DW, TW, DL, TL), cross-word scores, and 50-point bingo bonus for using all 7 rack tiles.
- **Part D — Move Generation:** Appel-Jacobson algorithm with anchor-based search. Uses board transposition for vertical moves. Supports blank tiles (wildcards).

**Key functions:**

| Function | Description |
|----------|-------------|
| `cell_has_tile(cell_img)` | Grayscale + subscript detection (two paths: red-ink for blanks, dark-ink + subscript for normal tiles) |
| `_get_point_class(cell_img)` | Classifies tile by subscript digit: `1pt`, `multi`, `10pt`, or `unknown` |
| `recognize_letter(cell_img, blank, reader)` | Multi-pass OCR with point-class disambiguation and shape fallbacks |
| `validate_board_tiles(board)` | Post-OCR correction using bag limits (max 1 Ñ, 1 RR, 1 LL) |
| `read_board_image(path, reader)` | Full board OCR pipeline: find bounds → extract cells → detect tiles → recognize letters → validate |
| `read_rack_image(path, reader)` | OCR a rack image, returns up to 7 tile tokens |
| `build_trie(lexicon_path)` | Build/load cached trie from FISE2 lexicon (imported from `lexicon.py`) |
| `compute_cross_checks(board, trie)` | Per-cell valid tile sets based on perpendicular words |
| `find_best_moves(board, rack, trie)` | Appel-Jacobson move generation + scoring, returns sorted moves |
| `compute_remaining_tiles(board, rack)` | Calculates unseen tiles from bag minus board minus rack |

**CLI usage:**

```bash
# Board image + text rack
python scrabble/analyze_board.py boards/Board1.jpg --rack AGUEIDA

# Board image + rack image
python scrabble/analyze_board.py boards/Board2.jpg --rack boards/rack2.jpg

# With manual OCR corrections and debug output
python scrabble/analyze_board.py boards/Board5.jpg --rack "AAÑICJM" \
    --corrections "J2=V,D10=I" --debug --top 20

# Blank tiles in text rack use '?'
python scrabble/analyze_board.py boards/Board4.jpg --rack "?NOETOY"
```

**CLI options:**

| Option | Description |
|--------|-------------|
| `board` | Path to board image (required) |
| `--rack`, `-r` | Rack as image path or text (e.g., `AGUEIDA`, `?NOETOY` for blanks) |
| `--top`, `-n` | Number of top moves to display (default: 10) |
| `--debug`, `-d` | Print per-cell OCR info including point class |
| `--corrections`, `-c` | Manual OCR corrections, e.g., `"D6=Y,B14=Z"` |

**Position notation:** `B14` (row-col) = horizontal play, `14B` (col-row) = vertical play.

**Input:** Board/rack images (JPG) + `LexiconFISE2.TXT` | **Output:** Ranked list of legal moves with scores

### study/clean_no_verbs.py

Prepares the primary input file for all downstream analyses.

| Function | Description |
|----------|-------------|
| `get_clean_noverbos()` | Reads `No_verbos.txt` (latin-1), decodes digraphs, removes standalone digit entries, deduplicates, sorts, and writes `No_verbos_filtrados.txt` (UTF-8) |

**Input:** `No_verbos.txt` | **Output:** `No_verbos_filtrados.txt`

### study/probability.py

Ranks words by a combined probability score derived from natural letter frequency and Scrabble tile availability.

| Function | Signature | Description |
|----------|-----------|-------------|
| `tokenize_word` | `(word) -> list` | Local tokenizer that maps digit-encoded digraphs to real digraph strings |
| `compute_scrabble_score` | `(word) -> int` | Sums point values for each token in a word |
| `compute_letter_probabilities` | `(words) -> dict` | Computes normalized letter frequency distribution across the word list |
| `geometric_mean` | `(p1, w1, p2, w2) -> float` | Weighted geometric mean: `(p1^w1 * p2^w2) ^ (1/(w1+w2))` |
| `compute_combined_score` | `(min_word_length, max_word_length, w1, w2, top_n, debug)` | Main pipeline: computes P_NVF (letter frequency probability) and P_Scrabble (tile availability weighted by inverse point value), combines them via geometric mean, normalizes by `scrabble_points / word_length`, sorts, and writes ranked output |

**Scoring formula:** `score = geometric_mean(P_NVF, w1, P_Scrabble, w2) * (points / length)`

**Input:** `No_verbos_filtrados.txt` | **Output:** `Ranked_Scrabble_Suggestions.txt`

### study/study_list.py

Generates a tiered study list with metadata about prefixes, suffixes, and root-word relationships.

| Function | Signature | Description |
|----------|-----------|-------------|
| `normalize` | `(counter) -> dict` | Normalizes a Counter to sum to 1.0 |
| `compute_scrabble_score` | `(word) -> int` | Sums Scrabble point values per token |
| `compute_letter_frequencies` | `(words) -> dict` | Normalized letter frequency across the corpus |
| `compute_anagram_probabilities` | `(words, letter_probs) -> dict` | Groups words by sorted-token key (anagram class), computes probability per class, distributes evenly among class members |
| `classify_word` | `(word, prob) -> str or None` | Assigns a tier: **Tier 1** = 4-5 tokens with prob > 1e-5; **Tier 2** = 6-8 tokens with prob > 1e-6; **Tier 3** = contains rare letters |
| `match_prefix` | `(word) -> str or None` | Checks if decoded word starts with a known prefix |
| `match_suffix` | `(word) -> str or None` | Checks if decoded word ends with a known suffix |
| `optimize_study_list` | `(min_len, max_len)` | Orchestrates the full pipeline: frequency analysis, anagram probabilities, tier classification, prefix/suffix matching, root-word detection, and writes tiered output sorted by root-relatedness then probability |

**Input:** `No_verbos_filtrados.txt` | **Output:** `Optimized_Study_List.txt`

### study/generator.py

Interactive tool that generates a filtered word list from user-supplied letter constraints.

| Function | Signature | Description |
|----------|-----------|-------------|
| `normalize` | `(counter) -> dict` | Normalizes a Counter |
| `compute_letter_frequencies` | `(tokenized_words) -> dict` | Letter frequency from pre-tokenized word lists |
| `compute_anagram_probabilities` | `(tokenized_words, letter_probs) -> dict` | Anagram-class-based probability computation |
| `is_buildable` | `(word_tokens, allowed_token_types, max_occurrence) -> bool` | Checks if a word can be built from a restricted set of letters with occurrence limits |
| `contains_required_letters` | `(word_tokens, required_tokens) -> bool` | Validates that a word contains all required letters |
| `main` | `()` | Prompts user for: input letters, length range, max letter reuse, required letters. Filters the lexicon accordingly, ranks by anagram probability, and writes to a named output file |

**Input:** `No_verbos_filtrados.txt` + interactive user input | **Output:** `<most_probable_word>.txt`

### study/nouns_csv.py

Generates a comprehensive CSV with 20+ metadata columns per word, including percentile rankings.

| Function | Signature | Description |
|----------|-----------|-------------|
| `tokenize_word` | `(word) -> list` | Local digit-to-digraph tokenizer |
| `compute_scrabble_score` | `(word) -> int` | Point value per word |
| `compute_letter_probabilities` | `(words) -> dict` | Normalized letter frequencies |
| `match_prefix` | `(word) -> str` | Matches against `EXTENSIVE_PREFIXES`, supports `V`/`C` wildcards |
| `match_suffix` | `(word) -> str` | Matches against `EXTENSIVE_SUFIXES`, handles `V`-terminated and `CC`/`VV` patterns |
| `match_pattern` | `(word) -> str` | Finds occurrence patterns (`Vh`, `tl`, `VVV`, etc.) |
| `get_hooks` | `(word, word_set, chars) -> dict` | Tests prefix hooks for all 28 tile types: whether prepending each character forms a valid word |
| `get_suffix_hooks` | `(word, word_set, chars) -> dict` | Tests suffix hooks for all 28 tile types: whether appending each character forms a valid word |
| `sorted_letters` | `(word) -> str` | Alphabetically sorted characters (anagram key) |
| `count_anagrams` | `(word, anagram_dict) -> int` | Counts anagram siblings in the lexicon |
| `main` | `()` | Full pipeline: loads words, computes all per-word metadata (length, prefix, suffix, pattern, hooks for all 28 tile types, anagram count, probability, scrabble value), calculates percentile buckets (P10/P25/P50/P75/P90/Top10), writes CSV |

**CSV columns:** `word`, `length`, `prefix`, `suffix`, `probabilityx1000`, `percentile`, `pattern`, `ending`, `value`, `anagrams`, `category`, plus 56 hook columns for all 28 tile types (`a-hook`, `b-hook`, ..., `ch-hook`, ..., `hook-a`, `hook-b`, ..., `hook-ch`, ...).

**Input:** `No_verbos_filtrados.txt` | **Output:** `word_analysis.csv`

### study/verbs_csv.py

Exports verb metadata including conjugation type classification.

| Function | Signature | Description |
|----------|-----------|-------------|
| `get_verb_type` | `(c_digit) -> str` | Maps classification code: 0=antiguo, 1=transitivo, 2=intransitivo, 3+=pronominal |
| `parse_categorized_verbs` | `(filename) -> dict` | Parses `Verbos_clasificados.TXT` (dash-delimited format), extracts verb base form and type, strips `-se` pronominal endings |
| `decode_word` | `(word) -> str` | Expands digit codes to digraph characters |
| `load_verb_list` | `(filename) -> set` | Loads and decodes the raw verb file |
| `compute_token_length` | `(word) -> int` | Token count via digraph-aware tokenization |
| `main` | `()` | Joins verb list with classifications, filters to 2-8 token length, checks `a-` prefix hooks, writes CSV |

**CSV columns:** `word`, `length`, `type`, `a-hook`

**Input:** `Verbos.txt` + `Verbos_clasificados.TXT` | **Output:** `verbs.csv`

### study/endings.py

Groups words by their final token.

| Function | Signature | Description |
|----------|-----------|-------------|
| `filter_words_by_ending` | `(min_len, max_len)` | Reads the clean lexicon, groups words by their last token against the 21 endings defined in config, writes one sorted file per ending |

**Input:** `No_verbos_filtrados.txt` | **Output:** `ends_with_<letter>.txt` (up to 21 files)

### study/prefixes.py

Filters words by prefix patterns using a pattern language where `V` = vowel, `C` = consonant, and literals match exactly.

| Function | Signature | Description |
|----------|-----------|-------------|
| `token_matches_code` | `(token, code) -> bool` | Tests a single token against a pattern code |
| `matches_prefix` | `(tokens, prefix_pattern) -> bool` | Checks if the token list starts with the pattern |
| `match_words_by_prefixes` | `(min_len, max_len)` | Filters words against all configured prefixes, writes one file per match |

**Input:** `No_verbos_filtrados.txt` | **Output:** `prefix_<pattern>.txt` (15+ files)

### study/suffixes.py

Filters words by suffix patterns with special vowel-variant grouping.

| Function | Signature | Description |
|----------|-----------|-------------|
| `normalize_suffix_pattern` | `(suffix_pattern) -> list` | Tokenizes a suffix pattern string, handling digraphs (e.g., `"illV"` becomes `['i', 'll', 'V']`) |
| `matches_suffix` | `(tokens, suffix_tokens) -> bool` | Checks if the token list ends with the suffix pattern |
| `match_words_by_suffixes` | `(min_len, max_len)` | For `V`-terminated suffixes, groups words by stem and lists vowel variants on the same line (e.g., `"tinta, e, o"`). For other suffixes, lists matches directly. Writes one file per suffix |

**Input:** `No_verbos_filtrados.txt` | **Output:** `suffix_<pattern>.txt` (40+ files)

### study/ocurrences.py

Finds words containing specific internal letter patterns (e.g., `Vh`, `tl`, `VVV`).

| Function | Signature | Description |
|----------|-----------|-------------|
| `matches_pattern` | `(tokens, pattern) -> bool` | Slides a window across the token list and checks if any subsequence matches the pattern |
| `match_words_by_patterns` | `(min_len, max_len)` | Filters words against all occurrence patterns, writes one file per pattern |

**Input:** `No_verbos_filtrados.txt` | **Output:** `pattern_<name>.txt` (6 files)

### study/filter_by_tiers.py

Filters words by consonant difficulty tiers. Tier 1 words use only the most common consonants; higher tiers progressively include rarer ones.

| Function | Signature | Description |
|----------|-----------|-------------|
| `filter_words_by_consonant_tiers` | `(min_len, max_len)` | Builds four tier combinations (TIER_1, TIER_1_2, TIER_1_3, TIER_1_4). Each tier's output excludes words already captured by TIER_1 |

**Tier consonant sets:**
- TIER_1: `{l, s, r, n, t}`
- TIER_1_2: TIER_1 + `{c, g, m, p, b, d}`
- TIER_1_3: TIER_1 + `{v, ch, y, q, f, h}`
- TIER_1_4: TIER_1 + `{rr, ll, j, x, z, n}`

**Input:** `No_verbos_filtrados.txt` | **Output:** `words_only_TIER_<name>_<min>_<max>.txt` (4 files)

### study/unique_anagrams.py

Identifies words that have no anagrams within the lexicon (singleton anagram classes).

| Function | Signature | Description |
|----------|-----------|-------------|
| `find_singleton_anagrams` | `(min_len, max_len)` | Groups all words by their sorted-token key. Words whose anagram class has exactly one member are "singletons" -- they cannot be rearranged into any other valid word |

**Input:** `No_verbos_filtrados.txt` | **Output:** `singleton_anagrams_<min>_<max>.txt`

### study/endings_with_useful_plurals.py

Targeted filter for specific length-and-ending combinations useful for plural/conjugation study.

| Function | Signature | Description |
|----------|-----------|-------------|
| `list_and_group_words_by_ending` | `()` | Extracts 5-token words ending in `{l, r, s, z, n}` and 6-token words ending in `{u, i}` |

**Input:** `No_verbos_filtrados.txt` | **Output:** `five_token_ending_<letter>.txt`, `six_token_ending_<letter>.txt`

### study/chains.py

Builds word transformation chains where consecutive words differ by exactly one token (Hamming distance = 1). Designed for flashcard-style learning progressions.

| Function | Signature | Description |
|----------|-----------|-------------|
| `map_digits_to_digraphs` | `(word) -> str` | Replaces digit codes with digraph substrings |
| `prepare_word` | `(raw) -> str` | Full decoding pipeline for raw lexicon entries |
| `hamming` | `(tokens1, tokens2) -> int` | Counts differing positions between two token lists |
| `build_graph` | `(words) -> dict` | Builds an adjacency graph connecting words at Hamming distance 1, skipping `a`/`o` swaps at word-final position |
| `extract_chains` | `(graph) -> list` | DFS-based extraction of longest disjoint paths (capped at 25 nodes) |
| `make_flashcard_chains` | `(clean_file, max_dist)` | Processes words of lengths 6, 7, and 8. Builds graph per length, extracts chains, writes top 10 chains per length to output |

**Input:** `No_verbos_filtrados.txt` | **Output:** `chains_study_list.txt`

### study/transforms.py

Word transformation tools: find all valid words formed by changing, inserting, or removing one letter. Uses the shared FISE2 trie from `lexicon.py`.

| Function | Signature | Description |
|----------|-----------|-------------|
| `one_letter_changes` | `(word: str, trie=None) -> list[dict]` | All valid words formed by changing one token. Each result: `{word, position, original, replacement, value}` |
| `insert_letter` | `(word: str, trie=None) -> list[dict]` | All valid words formed by inserting one token anywhere. Each result: `{word, position, inserted, value}` |
| `remove_letter` | `(word: str, trie=None) -> list[dict]` | All valid words formed by removing one token. Each result: `{word, position, removed, value}` |

**Complexity:** `O(L * 28)` trie lookups for change/insert, `O(L)` for remove, where L = token length.

**CLI usage:**

```bash
cd scrabble

# One-letter changes (default)
python -m study.transforms CASA

# Insert a letter
python -m study.transforms --insert CASA

# Remove a letter
python -m study.transforms --remove CASA

# All transformations
python -m study.transforms --all CASA
```

**Input:** Word (CLI arg) + FISE2 lexicon trie | **Output:** Transformation results grouped by position

### study/synergy.py

Computes synergy scores for Scrabble rack leaves — the letter combinations remaining after playing a word. High-synergy leaves co-occur frequently in valid words. Uses an inverted index for fast submultiset matching against the corpus.

| Function | Signature | Description |
|----------|-----------|-------------|
| `load_corpus` | `() -> list[Counter]` | Reads `No_verbos_filtrados.txt`, tokenizes each word, returns a list of token-count Counters |
| `build_inverted_index` | `(corpus_counters) -> dict` | Builds `(token, min_count) -> set[word_index]` mapping for fast subset queries |
| `count_subset_matches` | `(combo_counter, inv_index) -> int` | Counts corpus words containing all tokens in a combination via set intersection |
| `is_valid_combo` | `(combo_counter) -> bool` | Checks that a combination respects `SCRABBLE_TILES` tile count limits |
| `compute_synergies_for_size` | `(token_pool, size, inv_index) -> list[(tuple, int)]` | Generates all multiset combinations of a given size from a token pool, scores valid ones |
| `percentile_normalize` | `(results) -> list[(tuple, int)]` | Maps raw word-counts to 0–100 integers by percentile rank within each length group |
| `format_combo` | `(combo_tokens) -> str` | Converts internal token codes to display format (e.g., `a+ch+s`) |
| `write_csv` | `(all_scored)` | Writes scored combinations to `synergy.csv` |
| `compute_synergy` | `()` | Main orchestrator: loads corpus, builds index, computes synergies for sizes 1–5, normalizes, writes output |

**Combination generation:** Sizes 1–2 use all 28 tokens exhaustively. Sizes 3–5 use the top 10 tokens by size-1 synergy as a pool.

**Input:** `No_verbos_filtrados.txt` | **Output:** `synergy.csv`

### study/vowel_patterns.py

Categorizes 7-token words by vowel/consonant distribution patterns. Handles accented vowels (`á`, `é`, `í`, `ó`, `ú`, `ü`) as their base vowel equivalents.

**Categories:**
- 2 vowels (5 consonants)
- 2 consonants (5 vowels)
- 3 consonants with at least 2 duplicated
- 3+ of a single vowel (i, u, or o)
- Vowels drawn only from {i, o, u}

**Input:** `No_verbos_filtrados.txt` | **Output:** `vowel_pattern_<category>_7.txt` (8 files)

### study/srs.py

SM-2 spaced repetition engine with JSON persistence. Tracks per-word review state and schedules future reviews based on recall quality.

| Function | Signature | Description |
|----------|-----------|-------------|
| `load_progress` | `(path) -> dict[str, CardState]` | Load `progress.json`, returns word → CardState mapping |
| `save_progress` | `(cards, path) -> None` | Atomically write progress (write .tmp then rename) |
| `update_card` | `(state, quality) -> CardState` | Apply SM-2 algorithm: quality 0–5 maps to interval/easiness updates |
| `get_due_cards` | `(cards, today) -> list[str]` | Words with `next_review <= today`, sorted most overdue first |
| `get_new_cards` | `(cards, pool, limit) -> list[str]` | Words from pool not yet studied, up to limit |
| `build_session` | `(cards, pool, session_size) -> list[str]` | Due cards first, then new cards to fill remaining slots |

**SM-2 quality scale:** 0=blackout, 1=wrong, 2=hard, 3=difficult, 4=ok, 5=easy.

**Persistence:** `Data/progress.json` — stores per-word easiness factor, interval, repetition count, next review date, total reviews, and lapse count.

### study/decks.py

Deck generation and filtering from `word_analysis.csv` and `verbs.csv`. Provides preset study decks and group-by helpers for organized study sessions.

| Function | Signature | Description |
|----------|-----------|-------------|
| `load_word_analysis` | `(path) -> list[dict]` | Load word CSV into card dicts with hooks, probability, morphology (cached) |
| `load_verbs` | `(path) -> list[dict]` | Load verb CSV into card dicts with type and computed point values (cached) |
| `filter_cards` | `(cards, **filters) -> list[dict]` | Filter by length, percentile, endings, prefix, suffix, pattern, tier, anagrams, hooks, value |
| `filter_verbs` | `(verbs, **filters) -> list[dict]` | Filter by length, beginning string, verb type, consonant tier |
| `apply_preset` | `(cards, name) -> list[dict]` | Apply a named word preset filter |
| `apply_verb_preset` | `(verbs, name) -> list[dict]` | Apply a named verb preset filter |
| `group_by_prefix` | `(cards, min_group) -> dict` | Group cards by prefix, sorted by group size |
| `group_by_suffix` | `(cards, min_group) -> dict` | Group cards by suffix, sorted by group size |
| `group_by_ending` | `(cards, min_group) -> dict` | Group cards by ending letter/digraph |
| `group_verbs_by_beginning` | `(verbs, prefix_len, min_group) -> dict` | Group verbs by first N characters |
| `group_verbs_by_type` | `(verbs) -> dict` | Group verbs by type (transitivo, intransitivo, etc.) |

**Preset decks:**

| Preset | Words | Description |
|--------|------:|-------------|
| `words-2` | 81 | 2-letter words |
| `words-3` | 387 | 3-letter words |
| `words-4` | 1,966 | 4-letter words |
| `words-5` | 4,963 | 5-letter words |
| `7L-2vowels` | 294 | 7-letter words with only 2 vowels |
| `7L-2cons` | 135 | 7-letter words with only 2 consonants |
| `high-prob` | 8,828 | High probability words (Top10, 4–8 letters) |
| `scoring-5` | 788 | 5-letter words with high-scoring letters |
| `scoring-6` | 1,260 | 6-letter words with high-scoring letters |
| `5L-end-d` | 16 | 5-letter words ending in D |
| `5L-end-l` | 322 | 5-letter words ending in L |
| `5L-end-n` | 388 | 5-letter words ending in N |
| `5L-end-r` | 214 | 5-letter words ending in R |
| `5L-end-z` | 71 | 5-letter words ending in Z |
| `verbs-3` | 5 | 3-letter verbs |
| `verbs-4` | 44 | 4-letter verbs |
| `verbs-5` | 484 | 5-letter verbs |
| `verbs-6` | 1,203 | 6-letter verbs |
| `verbs-7` | 1,945 | 7-letter verbs |
| `verbs-8` | 2,600 | 8-letter verbs |

### study/quiz.py

Interactive CLI quiz with 8 quiz modes, word lookup, spaced repetition scheduling, and organized study decks for both words and verbs.

**Quiz modes:**

| Mode | Description |
|------|-------------|
| **Review** | Self-assessment flashcards: show word, reveal hooks/morphology/anagrams, rate 0–5 |
| **Anagram** | Scrambled letters displayed, type the word (2 attempts, then reveal) |
| **Hooks** | Given a word, name the letters that can hook before/after it |
| **Pattern** | Word with high-value letters blanked out, fill in the full word |
| **Morphology** | Given a word, identify its prefix and suffix |
| **Transformation** | Given a word with one position blanked, name valid replacement letters |
| **Extension** | Given a word with an insertion slot, name valid letters to insert |
| **Reduction** | Given a word, identify which letters can be removed to leave a valid word |

**Study organization:**

| Option | Description |
|--------|-------------|
| **Preset decks** | 19 presets organized by length, vowel patterns, scoring, endings, and verb length |
| **Group study** | Study words grouped by shared prefix, suffix, or ending (paginated browser) |
| **Verb study** | Filter verbs by length, beginning, type, or browse grouped beginnings |
| **Word lookup** | Check word validity, see points, hooks, morphology, and transformation counts |

After each card, all modes show full reveal info: front/back hooks, prefix, suffix, ending, verb type, and anagram count.

**CLI usage:**

```bash
cd scrabble

# Interactive menu (recommended)
python -m study.quiz

# Direct mode with preset deck
python -m study.quiz --mode anagram --deck 7L-2vowels
python -m study.quiz --mode hooks --deck words-5
python -m study.quiz --mode transformation --deck words-4
python -m study.quiz --mode extension --deck words-3

# Filter by length, tier, or percentile
python -m study.quiz --mode review --length 7
python -m study.quiz --mode anagram --tier 4 --size 30

# Check progress
python -m study.quiz --stats
python -m study.quiz --list-decks
```

**CLI options:**

| Option | Description |
|--------|-------------|
| `--mode` | Quiz mode: `review`, `anagram`, `hooks`, `pattern`, `morphology`, `transformation`, `extension`, `reduction` |
| `--deck` | Preset deck name (see `--list-decks`) |
| `--length` | Filter by word length |
| `--tier` | Filter by consonant tier (1–4) |
| `--size` | Session size (default: 20) |
| `--min-percentile` | Minimum probability percentile (default: P25) |
| `--stats` | Show progress statistics |
| `--list-decks` | List all available preset decks |

**Interactive menu flow:**

```
── Scrabble Word Quiz ──

Words studied: 28   Due today: 12

Modes:
  1. Review           2. Anagram
  3. Hook quiz        4. Pattern fill
  5. Morphology       6. Transformation
  7. Extension        8. Reduction

Options:
  c. Check word      v. Verb study
  g. Group study     s. Stats
  d. List decks      q. Quit
```

**Input:** `word_analysis.csv`, `verbs.csv` | **Output:** `progress.json` (SRS state)

## Usage

Each module runs independently as a script.

### Board Analyzer

Analyze a board photo and find the best moves:

```bash
# Basic usage: board image + text rack
python scrabble/analyze_board.py boards/Board1.jpg --rack AGUEIDA

# Rack from image, show top 20 moves
python scrabble/analyze_board.py boards/Board3.jpg --rack boards/rack3.jpg --top 20

# With OCR corrections and blank tile in rack
python scrabble/analyze_board.py boards/Board4.jpg --rack "?NOETOY" --corrections "C13=V"

# Debug mode shows per-cell OCR details
python scrabble/analyze_board.py boards/Board5.jpg --rack "AAÑICJM" \
    --corrections "J2=V,D10=I" --debug
```

The board analyzer automatically detects tiles using grayscale analysis with subscript digit validation, disambiguates common OCR confusions (N/Ñ, R/RR, L/LL) using tile point values, and auto-detects Z tiles via their unique "10" subscript. Use `--corrections` for tiles that can't be auto-distinguished (e.g., V vs Y — both 4 pts).

### Interactive Quiz (Spaced Repetition)

```bash
cd scrabble

# Launch interactive menu
python -m study.quiz

# Direct modes
python -m study.quiz --mode anagram --deck words-5     # 5-letter anagram drill
python -m study.quiz --mode hooks --deck 5L-end-l      # Hook quiz for 5L ending in L
python -m study.quiz --mode pattern --deck verbs-7      # Pattern fill for 7-letter verbs
python -m study.quiz --mode morphology --length 6       # Prefix/suffix quiz for 6L words
python -m study.quiz --mode transformation --deck words-4  # One-letter change drill
python -m study.quiz --mode extension --deck words-3       # Insert-letter drill

# Progress
python -m study.quiz --stats
```

The quiz system uses SM-2 spaced repetition: words you struggle with appear more often, mastered words space out over days/weeks. Progress is saved in `Data/progress.json`.

### Word Study Tools

```bash
# Step 1: Prepare the clean lexicon (required before all other scripts)
python scrabble/study/clean_no_verbs.py

# Step 2: Run any analysis module independently
python scrabble/study/probability.py          # Ranked word suggestions
python scrabble/study/study_list.py           # Tiered study list
python scrabble/study/nouns_csv.py            # Full word analysis CSV (with 28-tile hooks)
python scrabble/study/verbs_csv.py            # Verb classification CSV
python scrabble/study/vowel_patterns.py       # 7-letter vowel pattern filters
python scrabble/study/generator.py            # Interactive word generator
python scrabble/study/chains.py              # Transformation chains
python scrabble/study/synergy.py             # Rack leave synergy values

# Word transformation tools (standalone CLI)
python -m study.transforms CASA              # One-letter changes
python -m study.transforms --insert CASA     # Insert a letter
python -m study.transforms --remove CASA     # Remove a letter
python -m study.transforms --all CASA        # All transformations

# Step 3: Run any filter module (most prompt for min/max token length)
python scrabble/study/endings.py
python scrabble/study/prefixes.py
python scrabble/study/suffixes.py
python scrabble/study/ocurrences.py
python scrabble/study/filter_by_tiers.py
python scrabble/study/unique_anagrams.py
python scrabble/study/endings_with_useful_plurals.py
```

Most filter scripts prompt interactively for minimum and maximum token length.

### Duplicate Scrabble

Multiplayer Duplicate Scrabble game where all players get the same rack and compete to find the best play.

**Web server mode** (players join via phone):

```bash
# Basic usage
python scrabble/duplicate/server.py scrabble/duplicate/dupli_config.txt

# With custom title and seed
python scrabble/duplicate/server.py scrabble/duplicate/dupli_config.txt \
    --title "CAMPEONATO MUNDIAL DE DUPLICADA - Partida 1" --seed 42

# Custom port
python scrabble/duplicate/server.py scrabble/duplicate/dupli_config.txt --port 9000
```

Host opens `http://localhost:8000/host` on a laptop/projector. Players join via `http://<host-ip>:8000/play` on their phones, enter the 4-digit room code, and submit plays as `WORD POSITION` (e.g., `CORTES H8`). Lowercase letters indicate blanks (e.g., `CORTEs H8` — the S is a blank).

**CLI mode** (terminal-based, moderator enters plays):

```bash
python scrabble/duplicate/main.py scrabble/duplicate/dupli_config.txt
```

**Configuration** (`dupli_config.txt`):

```
title = CAMPEONATO MUNDIAL DE DUPLICADA - Partida 1
rounds = 15
constraints = (2,15),(1,30)
time = 3:00
output = csv
```

- `rounds` — Number of rounds (0 = unlimited, play until bag depleted)
- `constraints` — `(min_vowels_and_consonants, until_round)` pairs
- `time` — Countdown per round in M:SS format
- `output` — Export format: csv, excel, html, or graphical

Results are exported to `scrabble/duplicate/resultados/` when the game ends.

### Solitaire Autoplay

Simulates a full solitaire Scrabble game, playing the best move each round:

```bash
python scrabble/autoplay_scrabble.py --seed 42 --rounds 0 --image
```
