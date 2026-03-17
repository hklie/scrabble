# Study — Word Study & Quiz System

A complete word study toolkit for Spanish Scrabble: lexicon analysis scripts that generate categorized word lists, plus an interactive spaced-repetition quiz system with 5 quiz modes, 20 preset decks, and SRS scheduling.

## Quick Start

### Interactive Quiz

```bash
cd scrabble

# Launch interactive menu
python -m study.quiz

# Direct mode with preset deck
python -m study.quiz --mode anagram --deck words-5
python -m study.quiz --mode hooks --deck 5L-end-l
python -m study.quiz --mode pattern --deck verbs-7
python -m study.quiz --mode morphology --length 6

# Check progress
python -m study.quiz --stats
python -m study.quiz --list-decks
```

### Regenerate Word Lists

```bash
# Step 1: Clean lexicon (required first)
python scrabble/study/clean_no_verbs.py

# Step 2: Generate metadata CSVs
python scrabble/study/nouns_csv.py       # word_analysis.csv (with 28-tile hooks)
python scrabble/study/verbs_csv.py       # verbs.csv

# Step 3: Generate word lists (all independent)
python scrabble/study/probability.py
python scrabble/study/study_list.py
python scrabble/study/vowel_patterns.py
python scrabble/study/endings.py
python scrabble/study/prefixes.py
python scrabble/study/suffixes.py
python scrabble/study/ocurrences.py
python scrabble/study/filter_by_tiers.py
python scrabble/study/unique_anagrams.py
python scrabble/study/endings_with_useful_plurals.py
python scrabble/study/chains.py
python scrabble/study/synergy.py
```

## Quiz System

### Modes

| Mode | Description |
|------|-------------|
| **Review** | Self-assessment flashcards. Show word, reveal hooks/morphology/anagrams, rate 0–5. |
| **Anagram** | Scrambled letters displayed, type the word. 2 attempts before reveal. |
| **Hooks** | Given a word, name the front and back hook letters (all 28 tile types). Scored by completeness. |
| **Pattern** | Word with ~40% of letters blanked (high-value letters hidden first). Fill in the full word. |
| **Morphology** | Given a word, identify its prefix and suffix. |

All modes show full reveal info after each card: hooks, prefix, suffix, ending, verb type, anagrams, and point value.

### Preset Decks

**Words by length:**
| Deck | Words | Description |
|------|------:|-------------|
| `words-2` | 81 | 2-letter words |
| `words-3` | 387 | 3-letter words |
| `words-4` | 1,966 | 4-letter words |
| `words-5` | 4,963 | 5-letter words |

**7-letter vowel patterns:**
| Deck | Words | Description |
|------|------:|-------------|
| `7L-2vowels` | 294 | 7-letter words with only 2 vowels |
| `7L-2cons` | 135 | 7-letter words with only 2 consonants |

**Probability and scoring:**
| Deck | Words | Description |
|------|------:|-------------|
| `high-prob` | 8,828 | High probability words (Top10, 4–8 letters) |
| `scoring-5` | 788 | 5-letter words with high-scoring letters |
| `scoring-6` | 1,260 | 6-letter words with high-scoring letters |

**5-letter words by ending:**
| Deck | Words | Description |
|------|------:|-------------|
| `5L-end-d` | 16 | Ending in D |
| `5L-end-l` | 322 | Ending in L |
| `5L-end-n` | 388 | Ending in N |
| `5L-end-r` | 214 | Ending in R |
| `5L-end-z` | 71 | Ending in Z |

**Verbs by length:**
| Deck | Words | Description |
|------|------:|-------------|
| `verbs-3` | 5 | 3-letter verbs |
| `verbs-4` | 44 | 4-letter verbs |
| `verbs-5` | 484 | 5-letter verbs |
| `verbs-6` | 1,203 | 6-letter verbs |
| `verbs-7` | 1,945 | 7-letter verbs |
| `verbs-8` | 2,600 | 8-letter verbs |

### Study Organization

- **Group study** (`g` in menu): Browse and study words grouped by shared prefix, suffix, or ending. Paginated browser for selecting groups.
- **Verb study** (`v` in menu): Filter verbs by length, beginning, type (transitivo/intransitivo/pronominal/antiguo), or browse beginnings grouped by first N characters.
- **Custom filter**: Specify length, consonant tier (1–4), ending letter, and minimum percentile.

### Spaced Repetition (SRS)

The quiz uses the **SM-2 algorithm** to schedule reviews:

- **Quality scale**: 0 (blackout) through 5 (easy recall).
- Words rated < 3 reset to short intervals; words rated 3+ space out over increasing intervals.
- Each session: due cards reviewed first, then new cards introduced (max 10 new per session).
- Progress saved in `Data/progress.json`.

Run `python -m study.quiz --stats` to see: total words studied, due today, mastered/learning/struggling counts, average ease factor.

### CLI Options

| Option | Description |
|--------|-------------|
| `--mode` | Quiz mode: `review`, `anagram`, `hooks`, `pattern`, `morphology` |
| `--deck` | Preset deck name (see `--list-decks`) |
| `--length` | Filter by word length |
| `--tier` | Filter by consonant tier (1–4) |
| `--size` | Session size (default: 20) |
| `--min-percentile` | Minimum probability percentile (default: P25) |
| `--stats` | Show progress statistics |
| `--list-decks` | List all available preset decks |

### Hook Input Format

The hook quiz accepts multiple formats:
- Space-separated: `s z t`
- Comma-separated: `s, z, t`
- Concatenated: `szt`
- Digraphs: `ch ll rr ñ`

## Word List Generation Scripts

### Data Pipeline

```
No_verbos.txt (raw, latin-1)
    │
    └─ clean_no_verbs.py ──► No_verbos_filtrados.txt (clean, UTF-8)
                                │
    ┌───────────────────────────┤
    │                           │
    ▼                           ▼
nouns_csv.py                 All filter scripts
    │                        (endings, prefixes, suffixes, etc.)
    ▼                           │
word_analysis.csv               ▼
(92K words, 56 hook cols)    *.txt word lists in Data/
```

### Scripts

| Script | Input | Output | Description |
|--------|-------|--------|-------------|
| `clean_no_verbs.py` | `No_verbos.txt` | `No_verbos_filtrados.txt` | Decode digraphs, deduplicate, sort |
| `nouns_csv.py` | Clean lexicon | `word_analysis.csv` | Full metadata: probability, hooks (28 tiles), prefix/suffix, anagrams, percentile |
| `verbs_csv.py` | `Verbos.txt` + classifications | `verbs.csv` | Verb type (transitivo/intransitivo/pronominal/antiguo) |
| `probability.py` | Clean lexicon | `Ranked_Scrabble_Suggestions.txt` | Combined NVF + Scrabble tile probability ranking |
| `study_list.py` | Clean lexicon | `Optimized_Study_List.txt` | 3-tier study list with prefix/suffix/root metadata |
| `generator.py` | Interactive input | `<word>.txt` | Generate words from letter constraints |
| `vowel_patterns.py` | Clean lexicon | `vowel_pattern_*.txt` (8 files) | 7-letter words by vowel/consonant count |
| `endings.py` | Clean lexicon | `ends_with_*.txt` (21 files) | Words grouped by final letter |
| `prefixes.py` | Clean lexicon | `prefix_*.txt` (15+ files) | Words matching prefix patterns |
| `suffixes.py` | Clean lexicon | `suffix_*.txt` (40+ files) | Words matching suffix patterns, vowel-variant grouping |
| `ocurrences.py` | Clean lexicon | `pattern_*.txt` (6 files) | Words containing internal patterns (VVV, Vh, tl, etc.) |
| `filter_by_tiers.py` | Clean lexicon | `words_only_TIER_*.txt` (4 files) | Words by consonant difficulty tier |
| `unique_anagrams.py` | Clean lexicon | `singleton_anagrams_*.txt` | Words with no anagram partners |
| `endings_with_useful_plurals.py` | Clean lexicon | `five/six_token_ending_*.txt` (7 files) | 5-letter words ending in L/R/S/Z/N, 6-letter in U/I |
| `chains.py` | Clean lexicon | `chains_study_list.txt` | Word transformation chains (Hamming distance 1) |
| `synergy.py` | Clean lexicon | `synergy.csv` | Rack leave synergy scores (0–100 percentile) |

### Consonant Tiers

| Tier | Consonants | Difficulty |
|------|-----------|------------|
| 1 | l, s, r, n, t | Most common |
| 2 | c, g, m, p, b, d | Medium |
| 3 | v, ch, y, q, f, h | Less common |
| 4 | rr, ll, j, x, z, ñ | Rare |

## Module Structure

| File | Type | Description |
|------|------|-------------|
| `quiz.py` | App | Interactive CLI quiz with 5 modes, verb/group study menus |
| `srs.py` | Engine | SM-2 spaced repetition algorithm + JSON persistence |
| `decks.py` | Library | Card loading, filtering, 20 presets, group-by helpers |
| `nouns_csv.py` | Generator | Full word metadata CSV (probability, hooks, morphology) |
| `verbs_csv.py` | Generator | Verb classification CSV |
| `clean_no_verbs.py` | Generator | Lexicon cleaning and deduplication |
| `probability.py` | Generator | Probabilistic word ranking |
| `study_list.py` | Generator | Tiered study list |
| `generator.py` | Interactive | Word generator from letter constraints |
| `vowel_patterns.py` | Generator | 7-letter vowel/consonant pattern filters |
| `endings.py` | Generator | Filter by ending letter |
| `prefixes.py` | Generator | Filter by prefix pattern |
| `suffixes.py` | Generator | Filter by suffix pattern (with vowel grouping) |
| `ocurrences.py` | Generator | Filter by internal letter patterns |
| `filter_by_tiers.py` | Generator | Filter by consonant difficulty |
| `unique_anagrams.py` | Generator | Singleton anagram detection |
| `endings_with_useful_plurals.py` | Generator | Targeted length + ending filters |
| `chains.py` | Generator | Word transformation chains |
| `synergy.py` | Generator | Rack leave synergy computation |
