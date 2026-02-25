import os
from config import CLEAN_NO_VERBS_FILE
from preprocessing import tokenize_word, detokenize_word


# Configuration
FIVE_TOKEN_ENDINGS = {"l", "r", "s", "z", "n"}
SIX_TOKEN_ENDINGS = {"u", "i"}


def list_and_group_words_by_ending():
    with open(CLEAN_NO_VERBS_FILE, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]

    base_dir = os.path.dirname(CLEAN_NO_VERBS_FILE)

    # Create output buckets
    five_token_groups = {ending: [] for ending in FIVE_TOKEN_ENDINGS}
    six_token_groups = {ending: [] for ending in SIX_TOKEN_ENDINGS}

    for word in words:
        tokens = tokenize_word(word)
        token_len = len(tokens)
        last_token = tokens[-1] if tokens else None

        if token_len == 5 and last_token in FIVE_TOKEN_ENDINGS:
            five_token_groups[last_token].append(detokenize_word(tokens))
        elif token_len == 6 and last_token in SIX_TOKEN_ENDINGS:
            six_token_groups[last_token].append(detokenize_word(tokens))

    # Write output for 5-token groups
    for ending, word_list in five_token_groups.items():
        if word_list:
            file_path = os.path.join(base_dir, f"five_token_ending_{ending}.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                for word in sorted(word_list):
                    f.write(f"{word}\n")
            print(f"{len(word_list)} words written to {file_path}")

    # Write output for 6-token groups
    for ending, word_list in six_token_groups.items():
        if word_list:
            file_path = os.path.join(base_dir, f"six_token_ending_{ending}.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                for word in sorted(word_list):
                    f.write(f"{word}\n")
            print(f"{len(word_list)} words written to {file_path}")


if __name__ == "__main__":
    list_and_group_words_by_ending()
