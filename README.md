# Spanish Scrabble Word Study Tool

A Python toolkit for analyzing, filtering, ranking, and organizing a Spanish-language Scrabble lexicon into study materials. It handles Spanish digraphs (`ch`, `ll`, `rr`) via an internal encoding system and produces categorized word lists, probability rankings, CSV exports, and flashcard-style transformation chains.

## Project Structure

```
scrabble/
├── Data/                          # Input lexicon files and all generated output
│   ├── Lexicon.TXT                # Complete Spanish Scrabble lexicon
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
│   ├── ends_with_*.txt            # [generated] Words grouped by ending
│   ├── prefix_*.txt               # [generated] Words grouped by prefix
│   ├── suffix_*.txt               # [generated] Words grouped by suffix
│   ├── pattern_*.txt              # [generated] Words matching occurrence patterns
│   ├── words_only_TIER_*.txt      # [generated] Words filtered by consonant tier
│   ├── singleton_anagrams_*.txt   # [generated] Words with no anagrams
│   ├── five_token_ending_*.txt    # [generated] 5-token words by ending
│   └── six_token_ending_*.txt     # [generated] 6-token words by ending
│
├── scrabble/                      # Source code package
│   ├── config.py                  # Central configuration (paths, constants, patterns)
│   ├── preprocessing.py           # Digraph tokenization/detokenization
│   ├── clean_no_verbs.py          # Lexicon cleaning and deduplication
│   ├── probability.py             # Probabilistic word ranking
│   ├── study_list.py              # Tiered study list generation
│   ├── generator.py               # Interactive word generator from letter constraints
│   ├── nouns_csv.py               # Comprehensive word analysis CSV export
│   ├── verbs_csv.py               # Verb classification CSV export
│   ├── endings.py                 # Filter words by ending letter
│   ├── prefixes.py                # Filter words by prefix pattern
│   ├── suffixes.py                # Filter words by suffix pattern
│   ├── ocurrences.py              # Filter words by internal letter patterns
│   ├── filter_by_tiers.py         # Filter words by consonant difficulty tiers
│   ├── unique_anagrams.py         # Find words with no anagrams in the lexicon
│   ├── endings_with_useful_plurals.py  # Targeted length+ending combinations
│   └── chains.py                  # Word transformation chain builder
│
└── README.md
```

## Dependencies

### External

| Package | Used by | Purpose |
|---------|---------|---------|
| `numpy` | `nouns_csv.py` | Percentile calculations for word scoring |
| `regex` | `chains.py` | Advanced regex support (imported but lightly used) |

### Internal

All modules depend on:

- **`config.py`** -- file paths, Scrabble tile/point data, digraph mappings, pattern definitions, tier groupings.
- **`preprocessing.py`** -- `tokenize_word()` and `detokenize_word()` for digraph-aware text handling.

`probability.py` additionally imports from `generator.py` (for its `tokenize_word` and `detokenize_word`, though it redefines `tokenize_word` locally).

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
                         clean_no_verbs.py
                         (decode digraphs, remove
                          digits, deduplicate, sort)
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │  No_verbos_filtrados.txt     │
                    │  (cleaned non-verb lexicon)  │
                    └──────────────┬──────────────┘
                                   │
          ┌────────────┬───────────┼───────────┬────────────┬───────────┐
          │            │           │           │            │           │
          ▼            ▼           ▼           ▼            ▼           ▼
    probability.py  study_list.py  nouns_csv.py  Pattern    Anagram   chains.py
                                               Filters    Analysis
          │            │           │           │            │           │
          ▼            ▼           ▼           │            ▼           ▼
     Ranked_       Optimized_   word_         │      singleton_    chains_
     Scrabble_     Study_       analysis.     │      anagrams_     study_
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
                   verbs_csv.py
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

### preprocessing.py

Foundation for all word manipulation across the project.

| Function | Signature | Description |
|----------|-----------|-------------|
| `tokenize_word` | `(word: str) -> list[str]` | Splits a word into tokens, detecting digraphs (`ch`, `ll`, `rr`) from their human-readable form and mapping them to internal codes |
| `detokenize_word` | `(tokens: list[str]) -> str` | Reconstructs a readable word from a token list, expanding digit codes back to digraph strings |

### clean_no_verbs.py

Prepares the primary input file for all downstream analyses.

| Function | Description |
|----------|-------------|
| `get_clean_noverbos()` | Reads `No_verbos.txt` (latin-1), decodes digraphs, removes standalone digit entries, deduplicates, sorts, and writes `No_verbos_filtrados.txt` (UTF-8) |

**Input:** `No_verbos.txt` | **Output:** `No_verbos_filtrados.txt`

### probability.py

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

### study_list.py

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

### generator.py

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

### nouns_csv.py

Generates a comprehensive CSV with 20+ metadata columns per word, including percentile rankings.

| Function | Signature | Description |
|----------|-----------|-------------|
| `tokenize_word` | `(word) -> list` | Local digit-to-digraph tokenizer |
| `compute_scrabble_score` | `(word) -> int` | Point value per word |
| `compute_letter_probabilities` | `(words) -> dict` | Normalized letter frequencies |
| `match_prefix` | `(word) -> str` | Matches against `EXTENSIVE_PREFIXES`, supports `V`/`C` wildcards |
| `match_suffix` | `(word) -> str` | Matches against `EXTENSIVE_SUFIXES`, handles `V`-terminated and `CC`/`VV` patterns |
| `match_pattern` | `(word) -> str` | Finds occurrence patterns (`Vh`, `tl`, `VVV`, etc.) |
| `get_hooks` | `(word, word_set, chars) -> dict` | Tests prefix hooks: whether prepending each character forms a valid word |
| `get_suffix_hooks` | `(word, word_set, chars) -> dict` | Tests suffix hooks: whether appending each character forms a valid word |
| `sorted_letters` | `(word) -> str` | Alphabetically sorted characters (anagram key) |
| `count_anagrams` | `(word, anagram_dict) -> int` | Counts anagram siblings in the lexicon |
| `main` | `()` | Full pipeline: loads words, computes all per-word metadata (length, prefix, suffix, pattern, hooks for `aeioulnrst`, anagram count, probability, scrabble value), calculates percentile buckets (P10/P25/P50/P75/P90/Top10), writes CSV |

**CSV columns:** `word`, `length`, `prefix`, `suffix`, `probabilityx1000`, `percentile`, `pattern`, `ending`, `value`, `anagrams`, `category`, plus 20 hook columns (`a-hook`, `e-hook`, ..., `hook-a`, `hook-e`, ...).

**Input:** `No_verbos_filtrados.txt` | **Output:** `word_analysis.csv`

### verbs_csv.py

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

### endings.py

Groups words by their final token.

| Function | Signature | Description |
|----------|-----------|-------------|
| `filter_words_by_ending` | `(min_len, max_len)` | Reads the clean lexicon, groups words by their last token against the 21 endings defined in config, writes one sorted file per ending |

**Input:** `No_verbos_filtrados.txt` | **Output:** `ends_with_<letter>.txt` (up to 21 files)

### prefixes.py

Filters words by prefix patterns using a pattern language where `V` = vowel, `C` = consonant, and literals match exactly.

| Function | Signature | Description |
|----------|-----------|-------------|
| `token_matches_code` | `(token, code) -> bool` | Tests a single token against a pattern code |
| `matches_prefix` | `(tokens, prefix_pattern) -> bool` | Checks if the token list starts with the pattern |
| `match_words_by_prefixes` | `(min_len, max_len)` | Filters words against all configured prefixes, writes one file per match |

**Input:** `No_verbos_filtrados.txt` | **Output:** `prefix_<pattern>.txt` (15+ files)

### suffixes.py

Filters words by suffix patterns with special vowel-variant grouping.

| Function | Signature | Description |
|----------|-----------|-------------|
| `normalize_suffix_pattern` | `(suffix_pattern) -> list` | Tokenizes a suffix pattern string, handling digraphs (e.g., `"illV"` becomes `['i', 'll', 'V']`) |
| `matches_suffix` | `(tokens, suffix_tokens) -> bool` | Checks if the token list ends with the suffix pattern |
| `match_words_by_suffixes` | `(min_len, max_len)` | For `V`-terminated suffixes, groups words by stem and lists vowel variants on the same line (e.g., `"tinta, e, o"`). For other suffixes, lists matches directly. Writes one file per suffix |

**Input:** `No_verbos_filtrados.txt` | **Output:** `suffix_<pattern>.txt` (40+ files)

### ocurrences.py

Finds words containing specific internal letter patterns (e.g., `Vh`, `tl`, `VVV`).

| Function | Signature | Description |
|----------|-----------|-------------|
| `matches_pattern` | `(tokens, pattern) -> bool` | Slides a window across the token list and checks if any subsequence matches the pattern |
| `match_words_by_patterns` | `(min_len, max_len)` | Filters words against all occurrence patterns, writes one file per pattern |

**Input:** `No_verbos_filtrados.txt` | **Output:** `pattern_<name>.txt` (6 files)

### filter_by_tiers.py

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

### unique_anagrams.py

Identifies words that have no anagrams within the lexicon (singleton anagram classes).

| Function | Signature | Description |
|----------|-----------|-------------|
| `find_singleton_anagrams` | `(min_len, max_len)` | Groups all words by their sorted-token key. Words whose anagram class has exactly one member are "singletons" -- they cannot be rearranged into any other valid word |

**Input:** `No_verbos_filtrados.txt` | **Output:** `singleton_anagrams_<min>_<max>.txt`

### endings_with_useful_plurals.py

Targeted filter for specific length-and-ending combinations useful for plural/conjugation study.

| Function | Signature | Description |
|----------|-----------|-------------|
| `list_and_group_words_by_ending` | `()` | Extracts 5-token words ending in `{l, r, s, z, n}` and 6-token words ending in `{u, i}` |

**Input:** `No_verbos_filtrados.txt` | **Output:** `five_token_ending_<letter>.txt`, `six_token_ending_<letter>.txt`

### chains.py

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

## Usage

Each module runs independently as a script. The typical workflow is:

```bash
# Step 1: Prepare the clean lexicon (required before all other scripts)
python scrabble/clean_no_verbs.py

# Step 2: Run any analysis module independently
python scrabble/probability.py          # Ranked word suggestions
python scrabble/study_list.py           # Tiered study list
python scrabble/nouns_csv.py            # Full word analysis CSV
python scrabble/verbs_csv.py            # Verb classification CSV
python scrabble/generator.py            # Interactive word generator
python scrabble/chains.py              # Transformation chains

# Step 3: Run any filter module (most prompt for min/max token length)
python scrabble/endings.py
python scrabble/prefixes.py
python scrabble/suffixes.py
python scrabble/ocurrences.py
python scrabble/filter_by_tiers.py
python scrabble/unique_anagrams.py
python scrabble/endings_with_useful_plurals.py
```

Most filter scripts prompt interactively for minimum and maximum token length.
