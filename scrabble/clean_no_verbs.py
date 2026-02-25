from config import NO_VERBS_FILE, CLEAN_NO_VERBS_FILE
from preprocessing import detokenize_word

def get_clean_noverbos():
    input_file = NO_VERBS_FILE
    output_file = CLEAN_NO_VERBS_FILE
    words = []

    with open(input_file, 'r', encoding='latin-1') as file:
        for line in file:
            parts = line.strip().split()
            for part in parts:
                if not part.isdigit():  # Ignore standalone digits
                    transformed = detokenize_word(part)
                    words.append(transformed)

    valid_words = sorted(set(words))

    with open(output_file, 'w', encoding='utf-8') as file:
        for word in valid_words:
            file.write(f"{word}\n")


if __name__ == '__main__':
    get_clean_noverbos()
