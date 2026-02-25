import os
from config import CLEAN_NO_VERBS_FILE, ENDINGS
from preprocessing import tokenize_word, detokenize_word


def filter_words_by_ending(min_len=1, max_len=100):
    # Read and tokenize all words from file
    with open(CLEAN_NO_VERBS_FILE, "r", encoding="utf-8") as f:
        all_words = [line.strip() for line in f if line.strip()]

    tokenized_words = [(word, tokenize_word(word)) for word in all_words]

    base_dir = os.path.dirname(CLEAN_NO_VERBS_FILE)
    total_written = 0

    for ending in ENDINGS:
        matched_words = [
            detokenize_word(tokens)
            for word, tokens in tokenized_words
            if len(tokens) >= min_len
            and len(tokens) <= max_len
            and tokens[-1] == ending
        ]

        if not matched_words:
            continue

        file_name = f"ends_with_{ending}.txt"
        output_path = os.path.join(base_dir, file_name)

        with open(output_path, "w", encoding="utf-8") as f:
            for w in sorted(matched_words):
                f.write(f"{w}\n")

        print(f"Saved {len(matched_words)} words ending in '{ending}' to {file_name}")
        total_written += 1

    if total_written == 0:
        print("No words matched any of the given endings.")
    else:
        print(f"Created {total_written} files.")


def main():
    try:
        min_len = int(input("Enter minimum word length (in tokens): ").strip())
        max_len = int(input("Enter maximum word length (in tokens): ").strip())
        filter_words_by_ending(min_len=min_len, max_len=max_len)
    except ValueError:
        print("Invalid input. Please enter integers.")


if __name__ == "__main__":
    main()
