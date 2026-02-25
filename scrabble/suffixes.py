import os
from collections import defaultdict
from config import CLEAN_NO_VERBS_FILE, SUFFIXES, DIGRAPH_MAP, DIGRAPHS
from preprocessing import tokenize_word, detokenize_word

VOWELS = {"a", "e", "i", "o", "u"}

def is_vowel(token):
    return token in VOWELS

def is_consonant(token):
    return token.isalpha() and not is_vowel(token)

def token_matches_code(token, code):
    """
    Matches a token against a pattern code:
    V = any vowel, C = any consonant, literal = exact token
    """
    if code == "V":
        return is_vowel(token)
    elif code == "C":
        return is_consonant(token)
    else:
        return token == code.lower()

def normalize_suffix_pattern(suffix_pattern):
    """
    Converts suffix like 'illV' into tokenized form with digraphs handled:
    e.g. 'illV' -> ['i', '2', 'V'] because 'll' maps to '2'
    """
    tokens = []
    i = 0
    while i < len(suffix_pattern):
        if i + 1 < len(suffix_pattern):
            pair = suffix_pattern[i:i+2]
            if pair in DIGRAPH_MAP:  # e.g. 'll', 'ch', 'rr'
                tokens.append(DIGRAPH_MAP[pair])
                i += 2
                continue
        tokens.append(suffix_pattern[i])
        i += 1
    return tokens

def matches_suffix(tokens, suffix_tokens):
    """
    Checks if the given tokens end with the suffix_tokens pattern.
    """
    if len(tokens) < len(suffix_tokens):
        return False
    end_slice = tokens[-len(suffix_tokens):]
    return all(token_matches_code(tok, pat) for tok, pat in zip(end_slice, suffix_tokens))

def match_words_by_suffixes(min_len, max_len):
    # Read words
    with open(CLEAN_NO_VERBS_FILE, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]

    base_dir = os.path.dirname(CLEAN_NO_VERBS_FILE)
    results_by_suffix = {suffix: [] for suffix in SUFFIXES}

    for word in words:
        tokens = tokenize_word(word)
        if not (min_len <= len(tokens) <= max_len):
            continue

        for suffix in SUFFIXES:
            suffix_tokens = normalize_suffix_pattern(suffix)
            if matches_suffix(tokens, suffix_tokens):
                results_by_suffix[suffix].append(tokens)

    # Write results
    for suffix, token_lists in results_by_suffix.items():
        if not token_lists:
            print(f"No matches for suffix '{suffix}'")
            continue

        suffix_tokens = normalize_suffix_pattern(suffix)
        output_file = os.path.join(base_dir, f"suffix_{suffix}.txt")
        grouped_lines = []

        if suffix.endswith("V"):
            # Group by stem and list vowel variants
            groups = defaultdict(set)
            for tokens in token_lists:
                if len(tokens) < len(suffix_tokens):
                    continue
                stem = tuple(tokens[:-1])  # everything except final vowel
                last_token = tokens[-1]
                if is_vowel(last_token):
                    groups[stem].add(last_token)

            for stem_tokens, vowel_set in sorted(groups.items()):
                # build a base word with the first vowel variant
                base_word = detokenize_word(list(stem_tokens) + [sorted(vowel_set)[0]])
                extra_vowels = sorted(v for v in vowel_set if v != base_word[-1])
                line = base_word
                if extra_vowels:
                    line += ", " + ", ".join(extra_vowels)
                grouped_lines.append(line)

        else:
            # No vowel generalization: just list all matches normally
            grouped_lines = sorted(detokenize_word(toks) for toks in token_lists)

        with open(output_file, "w", encoding="utf-8") as f:
            for line in grouped_lines:
                f.write(line + "\n")

        print(f"{len(grouped_lines)} entries written to {output_file}")

def main():
    try:
        min_len = int(input("Enter minimum word length (in tokens): ").strip())
        max_len = int(input("Enter maximum word length (in tokens): ").strip())
        match_words_by_suffixes(min_len, max_len)
    except ValueError:
        print("Invalid input. Please enter integers for min and max token length.")

if __name__ == "__main__":
    main()
