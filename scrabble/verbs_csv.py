import csv
import os
from config import VERBS, CATEGORIZED_VERBS, DIGRAPHS
from preprocessing import tokenize_word  # only to count token length

# Map C digit to verb type: ≥3 = pronominal
def get_verb_type(c_digit):
    try:
        n = int(c_digit)
        if n == 0:
            return "antiguo"
        elif n == 1:
            return "transitivo"
        elif n == 2:
            return "intransitivo"
        else:
            return "pronominal"
    except ValueError:
        return "unknown"

def parse_categorized_verbs(filename):
    """
    Parses CATEGORIZED_VERBS file and returns a dict {verb: type}.
    """
    verb_types = {}
    with open(filename, "r", encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            dash_parts = line.split("-")
            if len(dash_parts) < 4:
                continue

            c_digit = dash_parts[1]  # second segment
            verb_type = get_verb_type(c_digit)

            # Last part after dash contains verb and maybe trailing number
            verb_section = dash_parts[-1]
            verb_name = verb_section.split()[0].lower()  # take first word

            verb_name = verb_name[:-2] if verb_name.endswith('se') else verb_name
            verb_name = verb_name.split('(')[0].strip() if verb_name.endswith(')') else verb_name

            verb_types[verb_name] = verb_type
    return verb_types

def decode_word(word):
    """
    Decodes '1','2','3' back to 'ch','ll','rr'
    """
    result = []
    for ch in word:
        if ch in DIGRAPHS:
            result.append(DIGRAPHS[ch])
        else:
            result.append(ch)
    return ''.join(result)

def load_verb_list(filename):
    """
    Loads raw verb list from VERBS file and decodes digraphs.
    """
    verbs = set()
    with open(filename, "r", encoding="latin-1") as f:
        for line in f:
            encoded = line.strip().lower()
            if encoded:
                decoded = decode_word(encoded)
                verbs.add(decoded)
    return verbs

def compute_token_length(word):
    """
    Computes the token length using digraph tokenization.
    """
    return len(tokenize_word(word))

def main():
    verb_types = parse_categorized_verbs(CATEGORIZED_VERBS)
    all_verbs = load_verb_list(VERBS)

    output_path = os.path.join(os.path.dirname(VERBS), "verbs.csv")

    with open(output_path, "w", encoding="utf-8", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["word", "length", "type", "a-hook"])

        for verb in sorted(all_verbs):
            token_length = compute_token_length(verb)
            if 2 <= token_length <= 8:
                vtype = verb_types.get(verb, "unknown")
                a_hook_verb = f"a{verb}"
                a_hook = "yes" if a_hook_verb in all_verbs else ""
                writer.writerow([verb, token_length, vtype, a_hook])

    print(f"CSV written to {output_path}")

if __name__ == "__main__":
    main()
