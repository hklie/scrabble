import os
import regex  # pip install regex
from collections import defaultdict, deque
from preprocessing import tokenize_word, decode_digraphs

from config import CLEAN_NO_VERBS_FILE, DIGRAPH_MAP, DIGRAPHS, BASE_PATH

# maximum chain length
MAX_CHAIN_LEN = 25
# lengths to process
TARGET_LENGTHS = [6, 7, 8]


def map_digits_to_digraphs(word):
    for digit, dg in DIGRAPH_MAP.items():
        word = word.replace(digit, dg)
    return word


def prepare_word(raw):
    # first map digits to actual digraph letters, then decode digraph placeholders
    return decode_digraphs(map_digits_to_digraphs(raw))


def hamming(tokens1, tokens2):
    return sum(t1 != t2 for t1, t2 in zip(tokens1, tokens2))


def build_graph(words):
    graph = defaultdict(set)
    toks = {w: tokenize_word(w) for w in words}
    print(f"  Building graph: comparing {len(words)} words of same length...")
    for i, w1 in enumerate(words):
        t1 = toks[w1]
        for w2 in words[i+1:]:
            t2 = toks[w2]
            if len(t1) != len(t2):
                continue
            # skip a<->o token swaps at end
            if (t1[-1], t2[-1]) in [( 'a','o'), ('o','a')]:
                continue
            if hamming(t1, t2) == 1:
                graph[w1].add(w2)
                graph[w2].add(w1)
    return graph


def extract_chains(graph):
    visited = set()
    chains = []
    for node in graph:
        if node in visited:
            continue
        # depth-first search for longest chain from this root
        stack = [(node, [node])]
        best_chain = []
        while stack:
            curr, path = stack.pop()
            if len(path) > len(best_chain):
                best_chain = path
            if len(path) >= MAX_CHAIN_LEN:
                continue
            for nbr in graph[curr]:
                if nbr not in path:
                    stack.append((nbr, path + [nbr]))
        # mark chain nodes visited
        for w in best_chain:
            visited.add(w)
        chains.append(best_chain)
    return chains


def make_flashcard_chains(clean_file, max_dist=1):
    print(f"Loading words from {clean_file}...")
    with open(clean_file, encoding='utf-8') as f:
        all_raw = [w.strip().lower() for w in f if w.strip()]
    print(f"  Total words loaded: {len(all_raw)}")

    out_file = os.path.join(BASE_PATH, "chains_study_list.txt")
    print(f"Writing results to {out_file}...\n")
    with open(out_file, "w", encoding='utf-8') as out:
        for length in TARGET_LENGTHS:
            # prepare decoded words of exact token length
            dec_words = [prepare_word(w) for w in all_raw]
            words = [w for w in dec_words if len(tokenize_word(w)) == length]
            print(f"Processing length={length} ({len(words)} words)")
            if not words:
                print(f"  No words of length {length}, skipping.\n")
                continue

            graph = build_graph(words)
            print(f"  Graph: {len(graph)} nodes, "
                  f"{sum(len(nbrs) for nbrs in graph.values())//2} edges")

            chains = extract_chains(graph)
            print(f"  Extracted {len(chains)} disjoint chains.")
            out.write(f"# Length {length}: Chains (up to {MAX_CHAIN_LEN})\n")
            for i, chain in enumerate(chains[:10], 1):
                out.write(f"Chain {i} ({len(chain)}):\n")
                out.write(" -> ".join(chain) + "\n\n")

    print("Done.")

if __name__ == '__main__':
    make_flashcard_chains(CLEAN_NO_VERBS_FILE)