CONFIG = {
    'thigh': {
        'sit': {
            'bout': 5,
            'inclination_angle': 45,
        },
        'lie': {
            'bout': 1,
            'orientation_angle': 65,
        },
        'stand': {
            'bout': 2,
            'inclination_angle': 45,
            'movement_threshold': 0.1,
        },
        'walk': {
            'bout': 2,
            'inclination_angle': 45,
            'movement_threshold': 0.1,
            'run_threshold': 0.72,
        },
        'stairs': {
            'bout': 5,
            'inclination_angle': 45,
            'movement_threshold': 0.1,
            'run_threshold': 0.72,
            'direction_threshold': 40,
            'stairs_threshold': 4,
        },
        'run': {
            'bout': 2,
            'inclination_angle': 45,
            'run_threshold': 0.72,
            'step_frequency': 2.5,
        },
        'bicycle': {
            'bout': 15,
            'movement_threshold': 0.1,
            'direction_threshold': 40,
        },
        'row': {
            'bout': 15,
            'movement_threshold': 0.1,
        },
        'shuffle': {
            'bout': 2,
        },
    }
}

ACTIVITIES = {
    0: 'non-wear',
    1: 'lie',
    2: 'sit',
    3: 'stand',
    4: 'shuffle',
    5: 'walk',
    6: 'run',
    7: 'stairs',
    8: 'bicycle',
    9: 'row',
    10: 'kneel',
    11: 'squat',
}

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
