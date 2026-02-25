import os
from config import CLEAN_NO_VERBS_FILE, PREFIXES
from preprocessing import tokenize_word, detokenize_word

VOWELS = {"a", "e", "i", "o", "u"}

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

def matches_prefix(tokens, prefix_pattern):
    """
    Checks if the token list starts with the given pattern.
    Pattern can include V (vowel), C (consonant), or literal letters.
    """
    if len(tokens) < len(prefix_pattern):
        return False

    prefix_tokens = tokens[:len(prefix_pattern)]
    return all(token_matches_code(tok, code) for tok, code in zip(prefix_tokens, prefix_pattern))

def match_words_by_prefixes(min_len, max_len):
    with open(CLEAN_NO_VERBS_FILE, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]

    base_dir = os.path.dirname(CLEAN_NO_VERBS_FILE)
    results_by_prefix = {prefix: [] for prefix in PREFIXES}

    for word in words:
        tokens = tokenize_word(word)
        if not (min_len <= len(tokens) <= max_len):
            continue

        for prefix in PREFIXES:
            if matches_prefix(tokens, prefix):
                results_by_prefix[prefix].append(detokenize_word(tokens))

    for prefix, matches in results_by_prefix.items():
        if matches:
            output_file = os.path.join(base_dir, f"prefix_{prefix}.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                for word in sorted(set(matches)):
                    f.write(f"{word}\n")
            print(f"{len(matches)} words matched prefix '{prefix}' → saved to {output_file}")
        else:
            print(f"No matches found for prefix '{prefix}'")

def main():
    try:
        min_len = int(input("Enter minimum word length (in tokens): ").strip())
        max_len = int(input("Enter maximum word length (in tokens): ").strip())
        match_words_by_prefixes(min_len, max_len)
    except ValueError:
        print("Invalid input. Please enter valid integers for token range.")

if __name__ == "__main__":
    main()
