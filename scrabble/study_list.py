from collections import defaultdict, Counter

from config import CLEAN_NO_VERBS_FILE, OUTPUT_STUDY_LIST, RARE_LETTERS, SUFFIXES, PREFIXES, SCRABBLE_POINTS

from preprocessing import tokenize_word, detokenize_word

def normalize(counter):
    total = sum(counter.values())
    return {k: v / total for k, v in counter.items()}

def compute_scrabble_score(word):
    return sum(SCRABBLE_POINTS.get(ch, 0) for ch in tokenize_word(word))

def compute_letter_frequencies(words):
    all_letters = []
    for word in words:
        all_letters.extend(tokenize_word(word))
    return normalize(Counter(all_letters))

def compute_anagram_probabilities(words, letter_probs):
    anagram_classes = defaultdict(list)
    for word in words:
        token_key = ''.join(sorted(tokenize_word(word)))
        anagram_classes[token_key].append(word)

    base_probs = {}
    for key in anagram_classes:
        prob = 1.0
        for ch in key:
            prob *= letter_probs.get(ch, 1e-6)
        base_probs[key] = prob

    total_base = sum(base_probs.values())
    base_probs = {k: p / total_base for k, p in base_probs.items()}

    final_probs = {}
    for key, word_list in anagram_classes.items():
        n = len(word_list)
        for word in word_list:
            final_probs[word] = base_probs[key] / n

    return final_probs

def classify_word(word, prob):
    length = len(tokenize_word(word))
    if length in [4, 5] and prob > 1e-5:
        return "Tier 1"
    elif length in [6, 7, 8] and prob > 1e-6:
        return "Tier 2"
    elif any(ch in RARE_LETTERS for ch in tokenize_word(word)):
        return "Tier 3"
    return None

def match_prefix(word):
    decoded = detokenize_word(word)
    return next((p for p in PREFIXES if decoded.startswith(p)), None)

def match_suffix(word):
    decoded = detokenize_word(word)
    return next((s for s in SUFFIXES if decoded.endswith(s)), None)


# === MAIN FUNCTION ===

def optimize_study_list(min_len=4, max_len=8):
    with open(CLEAN_NO_VERBS_FILE, 'r', encoding='utf-8') as f:
        all_words = [line.strip().lower() for line in f if line.strip()]
        words = [w for w in all_words if min_len <= len(tokenize_word(w)) <= max_len]

    if not words:
        print("No valid words found in that length range.")
        return

    letter_probs = compute_letter_frequencies(words)
    anagram_probs = compute_anagram_probabilities(words, letter_probs)

    # Optional: known roots (2-3 letter words)
    known_roots = set(w for w in all_words if 2 <= len(tokenize_word(w)) <= 3)

    # Organize by tiers
    tiered = defaultdict(list)

    for word in words:
        prob = anagram_probs.get(word, 0)
        tier = classify_word(word, prob)
        if not tier:
            continue

        prefix = match_prefix(word) or ""
        suffix = match_suffix(word) or ""
        root_related = any(detokenize_word(word).startswith(detokenize_word(root)) for root in known_roots)

        tiered[tier].append({
            "word": word,
            "decoded": detokenize_word(word),
            "prob": prob,
            "score": compute_scrabble_score(word),
            "prefix": prefix,
            "suffix": suffix,
            "built_from_known": root_related
        })

    # Write to file
    with open(OUTPUT_STUDY_LIST, "w", encoding="utf-8") as f:
        for tier in ["Tier 1", "Tier 2", "Tier 3"]:
            f.write(f"\n=== {tier} ===\n")
            entries = sorted(tiered[tier], key=lambda x: (-x["built_from_known"], -x["prob"]))
            for e in entries:
                note = " (root)" if e["built_from_known"] else ""
                f.write(f"{e['decoded']}\tScore: {e['score']:<3}  Prob: {e['prob']:.6e}  Prefix: {e['prefix']}  Suffix: {e['suffix']}{note}\n")

    print("Optimized study list written to:", OUTPUT_STUDY_LIST)

    

if __name__ == "__main__":
    optimize_study_list(min_len=4, max_len=8)

