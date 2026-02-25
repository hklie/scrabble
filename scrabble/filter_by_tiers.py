import os
from config import (
    CLEAN_NO_VERBS_FILE,
    TIER_1, TIER_2, TIER_3, TIER_4
)
from preprocessing import tokenize_word, detokenize_word

VOWELS = {"a", "e", "i", "o", "u"}

# Tier groups
CONSONANT_TIERS = {
    "TIER_1": TIER_1,
    "TIER_1_2": TIER_1.union(TIER_2),
    "TIER_1_3": TIER_1.union(TIER_3),
    "TIER_1_4": TIER_1.union(TIER_4),
}

def is_vowel(token):
    return token in VOWELS

def filter_words_by_consonant_tiers(min_len, max_len):
    with open(CLEAN_NO_VERBS_FILE, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]

    tokenized_words = [(word, tokenize_word(word)) for word in words]

    base_dir = os.path.dirname(CLEAN_NO_VERBS_FILE)
    tier_matches = {}

    # Step 1 — collect words for TIER_1 (no exclusions)
    tier_1_label = "TIER_1"
    tier_1_matches = set()
    for word, tokens in tokenized_words:
        if not (min_len <= len(tokens) <= max_len):
            continue
        if all(is_vowel(t) or t in TIER_1 for t in tokens):
            tier_1_matches.add(detokenize_word(tokens))
    tier_matches[tier_1_label] = tier_1_matches

    # Step 2 — collect for other tiers, excluding tier_1 words
    for label, allowed_consonants in CONSONANT_TIERS.items():
        if label == "TIER_1":
            continue

        matches = set()
        for word, tokens in tokenized_words:
            if not (min_len <= len(tokens) <= max_len):
                continue
            if all(is_vowel(t) or t in allowed_consonants for t in tokens):
                word_str = detokenize_word(tokens)
                if word_str not in tier_1_matches:
                    matches.add(word_str)
        tier_matches[label] = matches

    # Step 3 — write output files
    for label, words in tier_matches.items():
        file_path = os.path.join(base_dir, f"words_only_{label}_{min_len}_{max_len}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            for w in sorted(words):
                f.write(w + "\n")
        print(f"{len(words)} words written to {file_path}")

def main():
    try:
        min_len = int(input("Enter minimum word length (in tokens): ").strip())
        max_len = int(input("Enter maximum word length (in tokens): ").strip())
        filter_words_by_consonant_tiers(min_len, max_len)
    except ValueError:
        print("Invalid input. Please enter valid integers.")

if __name__ == "__main__":
    main()
