import os
from collections import defaultdict, Counter

# Assumes these are in a 'config.py' file and are imported by 'preprocessing.py'
# from config import DIGRAPHS, DIGRAPH_MAP

# Use the official functions from your preprocessing file
from preprocessing import tokenize_word, detokenize_word


def normalize(counter):
    """Normalizes the values of a counter to sum to 1."""
    total = sum(counter.values())
    # Avoid division by zero if the counter is empty
    return {k: v / total for k, v in counter.items()} if total > 0 else {}


def compute_letter_frequencies(tokenized_words):
    """Computes letter frequencies from a list of tokenized words."""
    all_letters = [letter for tokens in tokenized_words for letter in tokens]
    return normalize(Counter(all_letters))


def compute_anagram_probabilities(tokenized_words, letter_probs):
    """Computes the probability of each word based on its anagram class."""
    anagram_classes = defaultdict(list)
    for tokens in tokenized_words:
        token_key = ''.join(sorted(tokens))
        anagram_classes[token_key].append(tokens)

    base_probs = {}
    for key in anagram_classes:
        prob = 1.0
        for ch in key:
            prob *= letter_probs.get(ch, 1e-6)
        base_probs[key] = prob

    total_base = sum(base_probs.values())
    normalized_base_probs = {k: p / total_base for k, p in base_probs.items()} if total_base > 0 else {}

    final_probs = {}
    for key, token_list in anagram_classes.items():
        n = len(token_list)
        for tokens in token_list:
            # Use an immutable tuple of tokens as the dictionary key
            final_probs[tuple(tokens)] = normalized_base_probs.get(key, 0) / n

    return final_probs


def is_buildable(word_tokens, allowed_token_types, max_occurrence):
    """
    Checks if a word can be built from a set of allowed token types,
    with a maximum occurrence for any single token.
    """
    word_counter = Counter(word_tokens)

    # 1. Check if all tokens used in the word are present in the allowed types.
    if not set(word_counter.keys()).issubset(allowed_token_types):
        return False

    # 2. Check if the count of any single token in the word exceeds the maximum.
    if any(count > max_occurrence for count in word_counter.values()):
        return False

    return True


def contains_required_letters(word_tokens, required_tokens):
    """Checks if a word contains all required tokens."""
    word_token_set = set(word_tokens)
    return all(req in word_token_set for req in required_tokens)


def main():
    # --- Assume CLEAN_NO_VERBS_FILE is configured elsewhere ---
    from config import CLEAN_NO_VERBS_FILE

    # --- User Input ---
    input_letters_str = input("Enter a string of letters: ").strip().lower()
    input_tokens = tokenize_word(input_letters_str)
    allowed_token_types = set(input_tokens)

    min_len = int(input("Enter minimum word length: ").strip())
    max_len = int(input("Enter maximum word length: ").strip())

    max_occurrence_input = input("Enter max number of times each letter can be used [default=1]: ").strip()
    max_occurrence = int(max_occurrence_input) if max_occurrence_input else 1

    required_input = input("Enter required letters/digraphs (comma-separated), or leave blank: ").strip().lower()
    required_tokens = [r.strip() for r in required_input.split(',') if r] if required_input else []

    # --- File Processing ---
    with open(CLEAN_NO_VERBS_FILE, "r", encoding="utf-8") as f:
        all_tokenized_words = [tokenize_word(line.strip()) for line in f if line.strip()]

    # --- Word Selection ---
    valid_tokenized_words = [
        w_tokens for w_tokens in all_tokenized_words
        if is_buildable(w_tokens, allowed_token_types, max_occurrence)
           and min_len <= len(w_tokens) <= max_len
           and contains_required_letters(w_tokens, required_tokens)
    ]

    if not valid_tokenized_words:
        print("No valid words found matching all conditions.")
        return

    # --- Analysis and Output ---
    letter_probs = compute_letter_frequencies(valid_tokenized_words)
    word_probs = compute_anagram_probabilities(valid_tokenized_words, letter_probs)

    valid_tokenized_words.sort()

    most_probable_tokens = max(valid_tokenized_words, key=lambda w: word_probs.get(tuple(w), 0))

    restored_word = detokenize_word(most_probable_tokens)
    output_file = os.path.join(os.path.dirname(CLEAN_NO_VERBS_FILE), f"{restored_word}.txt")

    with open(output_file, "w", encoding="utf-8") as f:
        for tokens in valid_tokenized_words:
            f.write(f"{detokenize_word(tokens)}\n")

    print(f"Study list written to: {output_file}")


if __name__ == "__main__":
    main()