NUM_FEAT = {
    'letter': 16,
}

NUM_CLASS = {
    'letter': 26,
}

CONTINUITY = {
    'letter': [1 for _ in range(16)],
}

CAT_DROP = {
    'adult': [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    'covid': [1, 1, 0, 1, 1, 1, 1, 0]
}

MISS_CONFIG = {
    'letter': {
        'two': [7, 14],
        'three': [7, 8, 14]
    },
}