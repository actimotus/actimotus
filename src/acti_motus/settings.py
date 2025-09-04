LEGACY_CONFIG = {
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
            'anterior_posterior_angle': 25,
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
            'anterior_posterior_angle': 25,
            'direction_threshold': 40,
            'inclination_angle': 90,
        },
        'row': {
            'bout': 15,
            'movement_threshold': 0.1,
            'inclination_angle': 90,
        },
        'shuffle': {
            'bout': 2,
        },
    },
    'trunk': {
        'lie': {
            'inclination_angle': 45,
            'orientation_angle': 65,
        }
    },
}

CONFIG = {
    'thigh': {
        'sit': {
            'bout': 5,
            'inclination_angle': 47.5,
        },
        'lie': {
            'bout': 1,
            'orientation_angle': 65,
        },
        'stand': {
            'bout': 2,
            'inclination_angle': 47.5,
            'movement_threshold': 0.075,
        },
        'walk': {
            'bout': 2,
            'inclination_angle': 47.5,
            'movement_threshold': 0.075,
            'run_threshold': 0.7,
        },
        'stairs': {
            'bout': 5,
            'inclination_angle': 47.5,
            'movement_threshold': 0.075,
            'run_threshold': 0.7,
            'direction_threshold': 35.0,
            'stairs_threshold': 5,
            'anterior_posterior_angle': 20,
        },
        'run': {
            'bout': 2,
            'inclination_angle': 47.5,
            'run_threshold': 0.7,
            'step_frequency': 2.5,
        },
        'bicycle': {
            'bout': 15,
            'movement_threshold': 0.075,
            'anterior_posterior_angle': 20,
            'direction_threshold': 35.0,
            'inclination_angle': 87.5,
        },
        'row': {
            'bout': 15,
            'movement_threshold': 0.075,
            'inclination_angle': 87.5,
        },
        'shuffle': {
            'bout': 2,
        },
    },
    'trunk': {
        'lie': {
            'inclination_angle': 47.5,
            'orientation_angle': 65,
        }
    },
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
