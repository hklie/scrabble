import os
from config import CLEAN_NO_VERBS_FILE, OCURRENCES
from preprocessing import tokenize_word, detokenize_word

VOWELS = {"a", "e", "i", "o", "u"}
ALL_DIGRAPHS = {"ch", "ll", "rr"}

def is_vowel(token):
    return token in VOWELS

def is_consonant(token):
    return token.isalpha() and not is_vowel(token)

def token_matches_code(token, code):
    if code == "V":
        return is_vowel(token)
    elif code == "C":
        return is_consonant(token)
    else:
        return token == code.lower()

def matches_pattern(tokens, pattern):
    """Return True if any subsequence in tokens matches the pattern string"""
    pat_len = len(pattern)
    for i in range(len(tokens) - pat_len + 1):
        window = tokens[i:i + pat_len]
        if all(token_matches_code(tok, p) for tok, p in zip(window, pattern)):
            return True
    return False

def match_words_by_patterns(min_len, max_len):
    with open(CLEAN_NO_VERBS_FILE, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]

    base_dir = os.path.dirname(CLEAN_NO_VERBS_FILE)
    results_by_pattern = {pat: [] for pat in OCURRENCES}

    for word in words:
        tokens = tokenize_word(word)
        if not (min_len <= len(tokens) <= max_len):
            continue

        for pattern in OCURRENCES:
            if matches_pattern(tokens, pattern):
                results_by_pattern[pattern].append(detokenize_word(tokens))

    for pattern, matches in results_by_pattern.items():
        if matches:
            output_file = os.path.join(base_dir, f"pattern_{pattern}.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                for w in sorted(set(matches)):
                    f.write(f"{w}\n")
            print(f"{len(matches)} words matched pattern '{pattern}' → saved to {output_file}")
        else:
            print(f"No matches found for pattern '{pattern}'")

def main():
    try:
        min_len = int(input("Enter minimum word length (in tokens): ").strip())
        max_len = int(input("Enter maximum word length (in tokens): ").strip())
        match_words_by_patterns(min_len, max_len)
    except ValueError:
        print("Invalid input. Please enter valid integers.")


if __name__ == "__main__":
    main()
