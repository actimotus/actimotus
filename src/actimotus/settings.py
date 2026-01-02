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
        'fast-walk': {
            'bout': 60,
            'steps': 100,
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
            'steps': 2.5,
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
            'run_threshold': 0.65,
        },
        'fast-walk': {
            'bout': 15,
            'steps': 27,
        },
        'stairs': {
            'bout': 5,
            'inclination_angle': 47.5,
            'movement_threshold': 0.075,
            'run_threshold': 0.65,
            'direction_threshold': 35.0,
            'stairs_threshold': 5,
            'anterior_posterior_angle': 20,
        },
        'run': {
            'bout': 2,
            'inclination_angle': 47.5,
            'run_threshold': 0.65,
            'steps': 2.5,
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
    12: 'fast-walk',
}

FUSED_ACTIVITIES = {
    'lie': 'sedentary',
    'sit': 'sedentary',
    'kneel': 'sedentary',
    'shuffle': 'stand',
    'squat': 'stand',
    'fast-walk': 'walk',
    'stairs': 'walk',
}

PLOT = {
    'activities': {
        'non-wear': {'text': 'Non-wear', 'color': '#BDBDBD'},
        'lie': {'text': 'Lying', 'color': '#42A5F5'},
        'sit': {'text': 'Sitting', 'color': '#1565C0'},
        'stand': {'text': 'Standing', 'color': '#26A69A'},
        'shuffle': {'text': 'Shuffling', 'color': '#00695C'},
        'walk': {'text': 'Walking', 'color': '#66BB6A'},
        'fast-walk': {'text': 'Fast walking', 'color': '#2E7D32'},
        'run': {'text': 'Running', 'color': '#FF7043'},
        'stairs': {'text': 'Stairs', 'color': '#D84315'},
        'bicycle': {'text': 'Bicycling', 'color': '#E53935'},
        'row': {'text': 'Rowing', 'color': '#EC407A'},
        'kneel': {'text': 'Kneeling', 'color': '#26C6DA'},
        'squat': {'text': 'Squatting', 'color': '#00838F'},
    },
    'title': '24/7 Movement Behaviour',
    'x': 'Time',
    'y': 'Day',
    'legend': 'Activity',
    'weekdays': {
        'Monday': 'Monday',
        'Tuesday': 'Tuesday',
        'Wednesday': 'Wednesday',
        'Thursday': 'Thursday',
        'Friday': 'Friday',
        'Saturday': 'Saturday',
        'Sunday': 'Sunday',
    },
}


PLOT_FUSED = {
    'activities': {
        'non-wear': {'text': 'Non-wear', 'color': '#BDBDBD'},
        'sedentary': {'text': 'Sedentary', 'color': '#42A5F5'},
        'stand': {'text': 'Standing', 'color': '#26A69A'},
        'walk': {'text': 'Walking', 'color': '#66BB6A'},
        'run': {'text': 'Running', 'color': '#FF7043'},
        'bicycle': {'text': 'Bicycling', 'color': '#EF5350'},
        'row': {'text': 'Rowing', 'color': '#AB47BC'},
    },
    'title': '24/7 Movement Behaviour',
    'x': 'Time',
    'y': 'Day',
    'legend': 'Activity',
    'weekdays': {
        'Monday': 'Monday',
        'Tuesday': 'Tuesday',
        'Wednesday': 'Wednesday',
        'Thursday': 'Thursday',
        'Friday': 'Friday',
        'Saturday': 'Saturday',
        'Sunday': 'Sunday',
    },
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
