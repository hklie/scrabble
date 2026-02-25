import os
import csv
import numpy as np
from collections import Counter, defaultdict
from config import (
    CLEAN_NO_VERBS_FILE,
    EXTENSIVE_PREFIXES,
    EXTENSIVE_SUFIXES,
    OCURRENCES,
    SCRABBLE_POINTS,
    DIGRAPH_MAP,
)

# === Tokenization & Utilities ===

def tokenize_word(word):
    tokens = []
    i = 0
    while i < len(word):
        if word[i] in DIGRAPH_MAP:
            tokens.append(DIGRAPH_MAP[word[i]])
        else:
            tokens.append(word[i])
        i += 1
    return tokens

def compute_scrabble_score(word):
    tokens = tokenize_word(word)
    return sum(SCRABBLE_POINTS.get(ch, 0) for ch in tokens)

def compute_letter_probabilities(words):
    all_letters = []
    for word in words:
        all_letters.extend(tokenize_word(word))
    total_letters = len(all_letters)
    freqs = Counter(all_letters)
    return {letter: count / total_letters for letter, count in freqs.items()}

def match_prefix(word):
    for prefix in EXTENSIVE_PREFIXES:
        if word.startswith(prefix):
            return prefix
    return ""

def match_suffix(word):
    vowels = "aeiou"
    for suffix in EXTENSIVE_SUFIXES:
        if suffix == "VV":
            if len(word) >= 2 and word[-2] in vowels and word[-1] in vowels:
                return "VV"
            continue
        if suffix == "CC":
            if len(word) >= 2 and word[-2] not in vowels and word[-1] not in vowels:
                return "CC"
            continue
        if "V" in suffix:
            base = suffix.replace("V", "")
            for v in vowels:
                if word.endswith(base + v):
                    return suffix
        else:
            if word.endswith(suffix):
                return suffix
    return ""

def match_pattern(word):
    for pat in OCURRENCES:
        if pat == "VVV":
            for i in range(len(word) - 2):
                if word[i] in "aeiou" and word[i+1] in "aeiou" and word[i+2] in "aeiou":
                    return "VVV"
        elif pat in word:
            return pat
    return ""

def get_hooks(word, word_set, chars):
    return {f"{c}-hook": ("yes" if c + word in word_set else "no") for c in chars}

def get_suffix_hooks(word, word_set, chars):
    return {f"hook-{c}": ("yes" if word + c in word_set else "no") for c in chars}

def sorted_letters(word):
    return ''.join(sorted(word))

def count_anagrams(word, anagram_dict):
    key = sorted_letters(word)
    return len(anagram_dict[key])

# === Main Process ===

def main():
    print("Reading words...")
    with open(CLEAN_NO_VERBS_FILE, "r", encoding="utf-8") as f:
        raw_words = [line.strip() for line in f if line.strip()]
        word_set = set(raw_words)

    print(f"Total words: {len(raw_words)}")
    print("Computing letter probabilities...")
    letter_probs = compute_letter_probabilities(raw_words)

    print("Building anagram dictionary...")
    anagram_dict = defaultdict(set)
    for word in raw_words:
        key = sorted_letters(word)
        anagram_dict[key].add(word)

    print("Scoring words...")
    rows = []
    score_list = []

    for idx, word in enumerate(raw_words, 1):
        tokens = tokenize_word(word)
        length = len(tokens)

        prob = 1.0
        for ch in tokens:
            prob *= letter_probs.get(ch, 1e-6)

        scrabble_pts = compute_scrabble_score(word)
        score = prob * (scrabble_pts / length)
        scaled_score = score * 1000

        score_list.append(scaled_score)

        entry = {
            "word": word,
            "length": length,
            "prefix": match_prefix(word),
            "suffix": match_suffix(word),
            "probabilityx1000": scaled_score,  # temporarily a float
            "pattern": match_pattern(word),
            "ending": word[-1],
            "value": scrabble_pts,
            "anagrams": count_anagrams(word, anagram_dict),
            "category": "unknown"
        }

        entry.update(get_hooks(word, word_set, "aeioulnrst"))
        entry.update(get_suffix_hooks(word, word_set, "aeioulnrst"))

        rows.append(entry)

        if idx % 10000 == 0:
            print(f"Processed {idx} words...")

    # Compute percentiles
    print("Calculating percentiles...")
    score_array = np.array(score_list)
    p10, p25, p50, p75, p90 = np.percentile(score_array, [10, 25, 50, 75, 90])

    def get_percentile_label(val):
        if val <= p10:
            return "P10"
        elif val <= p25:
            return "P25"
        elif val <= p50:
            return "P50"
        elif val <= p75:
            return "P75"
        elif val <= p90:
            return "P90"
        else:
            return "Top10"

    # Inject percentile and format score as string
    for entry in rows:
        val = entry["probabilityx1000"]
        entry["percentile"] = get_percentile_label(val)
        entry["probabilityx1000"] = f"{val:.6f}"  # formatted string

    # Reorder columns: insert percentile after probability
    first_keys = [
        "word", "length", "prefix", "suffix",
        "probabilityx1000", "percentile", "pattern", "ending",
        "value", "anagrams", "category"
    ]
    all_keys = list(rows[0].keys())
    hook_keys = [k for k in all_keys if k not in first_keys]
    fieldnames = first_keys + sorted(hook_keys)

    output_path = os.path.join(os.path.dirname(CLEAN_NO_VERBS_FILE), "word_analysis.csv")
    print(f"Writing to file: {output_path}")

    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Done! CSV saved to:", output_path)

if __name__ == "__main__":
    main()
