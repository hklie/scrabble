import os

BASE_PATH = '/home/hk/Code/BackEnd/scrabble/Data'

LEXICON = os.path.join(BASE_PATH, "Lexicon.TXT")
LEXICON_FISE2 = os.path.join(BASE_PATH, "Master Copy", "LexiconFISE2.TXT")
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

EXTENSIVE_SUFIXES = SUFFIXES.union({'able', 'acha', 'acho', 'ucho', 'ucha', 'afV', 'agV', 'ao', 'biV', 'ego', 'giV',
                                    'ie', 'miV', 'omV', 'rrV'})

# --- Board Analysis Constants ---

BOARDS_PATH = os.path.join(os.path.dirname(BASE_PATH), 'boards')
TOTAL_BLANKS = 2
TOTAL_TILES = sum(SCRABBLE_TILES.values()) + TOTAL_BLANKS  # 100

# All 28 playable tile types in internal representation (digraphs encoded as 1/2/3)
ALL_TILES = list('abcdefghijlmnñopqrstuvxyz') + ['1', '2', '3']

# Internal point values (digraphs use their encoded keys)
INTERNAL_POINTS = {k: v for k, v in SCRABBLE_POINTS.items()}
INTERNAL_POINTS['1'] = SCRABBLE_POINTS['ch']
INTERNAL_POINTS['2'] = SCRABBLE_POINTS['ll']
INTERNAL_POINTS['3'] = SCRABBLE_POINTS['rr']

# Standard 15x15 premium square map (same for Spanish Scrabble)
PREMIUM_SQUARES = {}
for _r, _c in [(0,0),(0,7),(0,14),(7,0),(7,14),(14,0),(14,7),(14,14)]:
    PREMIUM_SQUARES[(_r,_c)] = 'TW'
for _r, _c in [(1,1),(1,13),(2,2),(2,12),(3,3),(3,11),(4,4),(4,10),(7,7),
               (10,4),(10,10),(11,3),(11,11),(12,2),(12,12),(13,1),(13,13)]:
    PREMIUM_SQUARES[(_r,_c)] = 'DW'
for _r, _c in [(1,5),(1,9),(5,1),(5,5),(5,9),(5,13),(9,1),(9,5),(9,9),(9,13),(13,5),(13,9)]:
    PREMIUM_SQUARES[(_r,_c)] = 'TL'
for _r, _c in [(0,3),(0,11),(2,6),(2,8),(3,0),(3,7),(3,14),(6,2),(6,6),(6,8),(6,12),
               (7,3),(7,11),(8,2),(8,6),(8,8),(8,12),(11,0),(11,7),(11,14),(12,6),(12,8),
               (14,3),(14,11)]:
    PREMIUM_SQUARES[(_r,_c)] = 'DL'

# Reverse lookup: point value → set of tiles that have that value
POINTS_TO_TILES = {}
for _tile, _pts in SCRABBLE_POINTS.items():
    POINTS_TO_TILES.setdefault(_pts, set()).add(_tile)

