import os

BASE_PATH = '/home/hk/Code/BackEnd/scrabble/Data'

LEXICON = os.path.join(BASE_PATH, "Lexicon.TXT")
VERBS = os.path.join(BASE_PATH, "Verbos.txt")
CATEGORIZED_VERBS = os.path.join(BASE_PATH, "Master Copy", "Verbos_clasificados.TXT")
NO_VERBS_FILE = os.path.join(BASE_PATH, "No_verbos.txt")
CLEAN_NO_VERBS_FILE = os.path.join(BASE_PATH, "No_verbos_filtrados.txt")
RANKED_SCRABBLE_WORDS = os.path.join(BASE_PATH, "Ranked_Scrabble_Suggestions.txt")
OUTPUT_STUDY_LIST = os.path.join(BASE_PATH, "Optimized_Study_List.txt")

DIGRAPHS = {'1': 'ch', '2': 'll', '3': 'rr'}
DIGRAPH_MAP= {v: k for k, v in DIGRAPHS.items()}

SCRABBLE_TILES = {
    'a': 12, 'b': 2, 'c': 4, 'd': 5, 'e': 12, 'f': 1, 'g': 2, 'h': 2, 'i': 6,
    'j': 1, 'l': 4, 'm': 2, 'n': 5, 'ñ': 1, 'o': 9, 'p': 2, 'q': 1, 'r': 5,
    's': 6, 't': 4, 'u': 5, 'v': 1, 'x': 1, 'y': 1, 'z': 1, 'ch': 1, 'll':1, 'rr': 1
}

SCRABBLE_POINTS = {
    'a': 1, 'b': 3, 'c': 3, 'd': 2, 'e': 1, 'f': 4, 'g': 2, 'h': 4, 'i': 1,
    'j': 8, 'l': 1, 'm': 3, 'n': 1, 'ñ': 8, 'o': 1, 'p': 3, 'q': 5, 'r': 1,
    's': 1, 't': 1, 'u': 1, 'v': 4, 'x': 8, 'y': 4, 'z': 10, 'ch': 5, 'll':8, 'rr': 8
}

RARE_LETTERS = {"j", "q", "z", "ñ", "x", "ch", "ll", "rr"}

TIER_1 = {'l', 's', 'r', 'n', 't'}
TIER_2 = {'c', 'g', 'm', 'p', 'b', 'd'}
TIER_3 = {'v', 'ch', 'y', 'q', 'f', 'h'}
TIER_4 = {'rr', 'll', 'j', 'x', 'z', 'ñ'}

OCURRENCES = {"Vh", "tl", "sh", "th", "lh", "VVV"}
ENDINGS = {"b", "c", "ch", "d", "f", "g", "h", "i", "j", "l", "ll", "m", "n", "p", "r", "t", "u", "v", "x", "y", "z"}

PREFIXES = {"ae", "ai", "ao", "au", "eo", "eu", "io", "oe", "oo", "gua", "hua", "gl", "bi", "tri"}
SUFFIXES = {"aceV", "ajV", "alV", "anV", "añV", "ante", "atV", "azo", "ciV", "cion", "dor", "dora", "enV",
            "ense", "eñV", "erV", "esV", "etV", "ble", "icV", "inV", "io", "uV", "ismo", "itV", "ivV", "izV",
            "l", "lV", "azo", "miV", "illV", "osV", "sion", "udV", "CC", "VV", "urV", "yV"}

EXTENSIVE_PREFIXES = PREFIXES.union({'re', 'des', 'dis', 'sub', 'ana', 'anti', 'geo', 'iso', 'mal', 'para', 'per',
                                     'pre', 'pro', 'sub', 'tele', 'teo', 'meso', 'ex', 'trans', 'in', 'im', 'lito',
                                     'apo', 'cata', 'en', 'hemi', 'peri', 'bar', 'alo', 'cito', 'contra', 'sobre'})

EXTENSIVE_SUFIXES = SUFFIXES.union({'able', 'acha', 'acho', 'afV', 'agV', 'ao', 'biV', 'ego', 'giV', 'ie', 'miV',
                                    'omV', 'rrV'})

