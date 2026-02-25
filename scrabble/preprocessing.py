# preprocessing.py

from config import DIGRAPHS, DIGRAPH_MAP

def tokenize_word(word):
    word = word.lower()
    result = []
    i = 0
    while i < len(word):
        if i + 1 < len(word) and word[i:i+2] in DIGRAPH_MAP:
            result.append(DIGRAPH_MAP[word[i:i+2]])
            i += 2
        else:
            result.append(word[i])
            i += 1
    return result

def detokenize_word(tokens):
    return ''.join(DIGRAPHS.get(tok, tok) for tok in tokens)
