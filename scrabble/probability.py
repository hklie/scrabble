from collections import Counter

from generator import tokenize_word, detokenize_word
from config import SCRABBLE_TILES, SCRABBLE_POINTS, RANKED_SCRABBLE_WORDS, CLEAN_NO_VERBS_FILE, DIGRAPHS, DIGRAPH_MAP

# === HELPERS ===

def tokenize_word(word):
    """Convert digit-based digraphs to real digraphs and split into tokens."""
    tokens = []
    i = 0
    while i < len(word):
        if word[i] in DIGRAPH_MAP:
            tokens.append(DIGRAPH_MAP[word[i]])
            i += 1
        else:
            tokens.append(word[i])
            i += 1
    return tokens

def compute_scrabble_score(word):
    tokens = tokenize_word(word)
    return sum(SCRABBLE_POINTS.get(ch, 0) for ch in tokens)

def compute_letter_probabilities(words):
    all_letters = []
    for word in words:
        all_letters.extend(tokenize_word(word))
    total_letters = len(all_letters)
    freqs = Counter(all_letters)
    return {letter: count / total_letters for letter, count in freqs.items()}

def geometric_mean(p1, w1, p2, w2):
    return (p1 ** w1 * p2 ** w2) ** (1 / (w1 + w2))



# === MAIN FUNCTION ===

def compute_combined_score(
    min_word_length=2,
    max_word_length=8,
    w1=1.5,
    w2=1.0,
    top_n=20,
    debug=False
):
    # --- Load and filter words ---
    with open(CLEAN_NO_VERBS_FILE, "r", encoding="utf-8") as f:
        raw_words = [line.strip().lower() for line in f if line.strip()]
        words = [w for w in raw_words if min_word_length <= len(tokenize_word(w)) <= max_word_length]

    if not words:
        print("No valid words found within length constraints.")
        return

    # --- Step 1: NVF Letter Probabilities ---
    letter_probs = compute_letter_probabilities(words)
    p_nvf_raw = {}
    for word in words:
        prob = 1.0
        for ch in tokenize_word(word):
            prob *= letter_probs.get(ch, 1e-6)  # fallback for rare letters
        p_nvf_raw[word] = prob

    # Normalize P_NVF
    total_nvf = sum(p_nvf_raw.values())
    p_nvf = {w: p / total_nvf for w, p in p_nvf_raw.items()}

    # --- Step 2: Scrabble Probabilities (tile freq * inverse point) ---
    total_tiles = sum(SCRABBLE_TILES.values())
    p_scrabble_raw = {}
    for word in words:
        prob = 1.0
        tiles_left = total_tiles
        for ch in tokenize_word(word):
            tile_freq = SCRABBLE_TILES.get(ch, 1)
            point_val = SCRABBLE_POINTS.get(ch, 1)
            letter_prob = tile_freq / tiles_left
            weight = 1 / point_val
            prob *= letter_prob * weight
            tiles_left = max(1, tiles_left - 1)
        p_scrabble_raw[word] = prob

    # Normalize P_Scrabble
    total_scrabble = sum(p_scrabble_raw.values())
    p_scrabble = {w: p / total_scrabble for w, p in p_scrabble_raw.items()}

    # --- Step 3: Combine and Score ---
    final_scores = {}
    debug_info = {}

    for word in words:
        p1 = p_nvf[word]
        p2 = p_scrabble[word]
        p_combined = geometric_mean(p1, w1, p2, w2)

        tokens = tokenize_word(word)
        length = len(tokens)
        score_pts = compute_scrabble_score(word)

        score = p_combined * (score_pts / length)
        final_scores[word] = score

        if debug:
            debug_info[word] = {
                'tokens': tokens,
                'p_nvf': p1,
                'p_scrabble': p2,
                'scrabble_pts': score_pts,
                'length': length,
                'score': score
            }

    # --- Sort and Output ---
    sorted_words = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

    # --- Sort and Output ---
    sorted_words = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

    with open(RANKED_SCRABBLE_WORDS, "w", encoding="utf-8") as f:
        for word, score in sorted_words:
            f.write(f"{detokenize_word(word)}\t{score:.8f}\n")

    # --- Print Top N ---
    print(f"\nTop {top_n} words (w1 = {w1}, w2 = {w2}):\n")
    for i in range(min(top_n, len(sorted_words))):
        word, score = sorted_words[i]
        print(f"{i+1:>2}. {detokenize_word(word):<12} Score: {score:.8f}")
        if debug:
            info = debug_info[word]
            print(f"    Tokens: {info['tokens']}")
            print(f"    P_NVF: {info['p_nvf']:.6e}  P_Scrabble: {info['p_scrabble']:.6e}")
            print(f"    Points: {info['scrabble_pts']}  Length: {info['length']}\n")


# Example usage:
if __name__ == "__main__":
    compute_combined_score(
        min_word_length=4,
        max_word_length=5,
        w1=1.5,
        w2=1.0,
        top_n=20,
        debug=True  # Turn off for faster runs
    )
