import os
from collections import defaultdict

from config import CLEAN_NO_VERBS_FILE
from preprocessing import tokenize_word, detokenize_word

def find_singleton_anagrams(min_len=1, max_len=100):
    anagram_classes = defaultdict(list)

    # Load and tokenize words
    with open(CLEAN_NO_VERBS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip()
            if not word:
                continue

            tokens = tokenize_word(word)
            token_key = tuple(sorted(tokens))
            anagram_classes[token_key].append(tokens)

    # Filter singleton classes
    singleton_words = [
        detokenize_word(tokens_list[0])
        for tokens_list in anagram_classes.values()
        if len(tokens_list) == 1 and min_len <= len(tokens_list[0]) <= max_len
    ]

    # Output to file
    output_file = os.path.join(os.path.dirname(CLEAN_NO_VERBS_FILE), f"singleton_anagrams_{min_len}_{max_len}.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        for word in sorted(singleton_words):
            f.write(f"{word}\n")

    print(f"{len(singleton_words)} singleton anagram words written to: {output_file}")

def main():
    try:
        min_len = int(input("Enter minimum word length (in tokens): ").strip())
        max_len = int(input("Enter maximum word length (in tokens): ").strip())
        find_singleton_anagrams(min_len=min_len, max_len=max_len)
    except ValueError:
        print("Invalid input. Please enter valid integers.")

if __name__ == "__main__":
    main()
