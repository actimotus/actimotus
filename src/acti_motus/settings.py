SYSTEM_SF = 30

BOUTS_LENGTH = {
    'sit': 5,
    'stand': 2,
    'move': 2,
    'walk': 2,
    'run': 2,
    'stairs': 5,
    'bicycle': 15,
    'row': 15,
    'lie': 1,
}

ACTIVITIES = {
    0: 'non-wear',
    1: 'lie',
    2: 'sit',
    3: 'stand',
    4: 'move',
    5: 'walk',
    6: 'run',
    7: 'stairs',
    8: 'bicycle',
    9: 'row',
    10: 'kneel',
    11: 'squat',
}

RAW = [
    'acc_x',
    'acc_y',
    'acc_z',
]

FEATURES = [
    'x',
    'y',
    'z',
    'sd_x',
    'sd_y',
    'sd_z',
    'sum_x',
    'sum_z',
    'sq_sum_x',
    'sq_sum_z',
    'sum_dot_xz',
    'hl_ratio',
    'walk_feature',
    'run_feature',
    'sf',
]

# Sens backend specific settings
SENS__FLOAT_FACTOR = 1_000_000
SENS__NORMALIZATION_FACTOR = -4 / 512

SENS__ACTIVITY_VALUES = [
    'steps',
    'trunk_inclination',
    'trunk_side_tilt',
    'trunk_direction',
    'arm_inclination',
]  # "activity" is always present in the dataframe
