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
        # The same three paces as DEFAULT, each walking second judged by the mean step rate of the walking
        # seconds in the minute around it. 2.3.3 had two classes split at 100 steps a minute over
        # 60 s blocks; its light walking is now `slow-walk`, its moderate walking `walk` and `fast-walk`.
        'pace': {
            'slow': 100,
            'fast': 115,
            'window': 60,
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
        # No step-rate rule turns walking into running: with a correct step rate it only turned fast
        # child walking into running, and `run_threshold` catches every run.
        'run': {
            'bout': 2,
            'inclination_angle': 45,
            'run_threshold': 0.72,
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
        # Walking is split into three paces by the step rate of each second, in steps a minute, with
        # no window: `slow-walk` below 100, `walk` from 100 to below 115, `fast-walk` from 115.
        # `slow-walk` is walking below 4 km/h, the Compendium's edge between light and moderate.
        # Tuned on three walking-speed datasets.
        'pace': {
            'slow': 100,
            'fast': 115,
            'window': 1,
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
        # No step-rate rule turns walking into running: with a correct step rate it only turned fast
        # child walking into running, and `run_threshold` catches every run.
        'run': {
            'bout': 2,
            'inclination_angle': 47.5,
            'run_threshold': 0.65,
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
            'inclination_upper': 110.0,
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

# From still to active. Renumbered after 2.3.3: the codes of 2.3.3 and earlier mean other classes.
ACTIVITIES = {
    0: 'non-wear',
    1: 'lie',
    2: 'sit',
    3: 'kneel',
    4: 'squat',
    5: 'stand',
    6: 'shuffle',
    7: 'slow-walk',
    8: 'walk',
    9: 'fast-walk',
    10: 'run',
    11: 'stairs',
    12: 'bicycle',
    13: 'row',
}

FUSED_ACTIVITIES = {
    'lie': 'sedentary',
    'sit': 'sedentary',
    'kneel': 'sedentary',
    'shuffle': 'stand',
    'squat': 'stand',
    'slow-walk': 'walk',
    'fast-walk': 'walk',
    'stairs': 'walk',
}

# Every activity in exactly one intensity band, so the four bands add up to the whole recording.
# `none` is counted in no band: non-wear, and standing, which is neither sedentary nor light here.
# `Exposures(intensity=...)` takes a mapping of the same shape.
INTENSITY = {
    'sedentary': ['lie', 'sit', 'kneel'],
    'lpa': ['squat', 'shuffle', 'slow-walk'],
    'mvpa': ['walk', 'fast-walk', 'run', 'stairs', 'bicycle', 'row'],
    'none': ['non-wear', 'stand'],
}

PLOT = {
    'activities': {
        'non-wear': {'text': 'Non-wear', 'color': '#BDBDBD'},
        'lie': {'text': 'Lying', 'color': '#42A5F5'},
        'sit': {'text': 'Sitting', 'color': '#1565C0'},
        'kneel': {'text': 'Kneeling', 'color': '#26C6DA'},
        'squat': {'text': 'Squatting', 'color': '#00838F'},
        'stand': {'text': 'Standing', 'color': '#26A69A'},
        'shuffle': {'text': 'Shuffling', 'color': '#00695C'},
        'slow-walk': {'text': 'Slow walking', 'color': '#A5D6A7'},
        'walk': {'text': 'Walking', 'color': '#66BB6A'},
        'fast-walk': {'text': 'Fast walking', 'color': '#2E7D32'},
        'run': {'text': 'Running', 'color': '#FF7043'},
        'stairs': {'text': 'Stairs', 'color': '#D84315'},
        'bicycle': {'text': 'Bicycling', 'color': '#E53935'},
        'row': {'text': 'Rowing', 'color': '#EC407A'},
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
    'steps',  # No longer computed, so always 0; the slot is kept so later columns do not shift.
    'trunk_inclination',
    'trunk_side_tilt',
    'trunk_direction',
    'arm_inclination',
]  # "activity" is always present in the dataframe
