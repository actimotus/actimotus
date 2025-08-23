from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import scipy
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from acti_motus import Activities, Features

MAP_THIGH_TRUNK = {
    'standing': 'stand',
    'shuffling': 'shuffle',
    'walking': 'walk',
    'stairs (descending)': 'stairs',
    'stairs (ascending)': 'stairs',
    'sitting': 'sit',
    'cycling (sit)': 'bicycle',
    'lying': 'lie',
    'cycling (sit, inactive)': 'bicycle',
    'cycling (stand)': 'bicycle',
    'running': 'run',
    'cycling (stand, inactive)': 'bicycle',
}

MAP_THIGH = {
    'standing': 'stand',
    'standing_static': 'stand',
    'standing_dynamic': 'shuffle',
    'sitting': 'sit',
    'lying_f': 'lie',
    'lying_s': 'lie',
    'lying_b': 'lie',
    'walking': 'walk',
    'walking_slow': 'walk',
    'walking_mod': 'walk',
    'walking_fast': 'walk',
    'running_slow': 'run',
    'running_mod': 'run',
    'running_fast': 'run',
    'cycling_slow': 'bicycle',
    'cycling_mod': 'bicycle',
    'cycling_fast': 'bicycle',
    'cycling_dynamic': 'bicycle',
    'cycling_static': 'bicycle',
    'cycling_standing': 'bicycle',
    'running': 'run',
    'lying_down': 'lie',
    'undefined': 'remove',
    'undefined_covered': 'remove',
    'walking_stairs': 'stairs',
    'start': 'remove',
    'end': 'remove',
    'heel_3': 'remove',
    'heel_2': 'remove',
    'heel_1': 'remove',
}

CONFUSSION_MATRIX = {
    'true': 'True',
    'predicted': 'Predicted',
}


def rotate_by_90_degrees_over_x(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    y_new = -df['acc_z']
    z_new = df['acc_y']

    df['acc_y'] = y_new
    df['acc_z'] = z_new

    return df


def process_motus(
    folder: str,
    vendor: str,
    map: dict[str, str],
    trunk_sensor: bool,
    rotate: bool,
    orientation: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = Path(folder)
    files = (path / 'thigh').glob('*.parquet')

    features = Features(chunking=False, calibrate=False)
    activities = Activities(vendor=vendor, chunks=False, orientation=orientation)  # type: ignore

    results = []
    results_trunk = []

    for thigh in files:
        gt = thigh.parent.parent / 'ground_truth' / thigh.name

        if not gt.exists():
            print(f'Ground truth file does not exist for {thigh.name}, skipping.')
            continue

        ground_truth = pd.read_parquet(gt)
        df = pd.read_parquet(thigh)

        if rotate:
            df = rotate_by_90_degrees_over_x(df)

        extracted_features = features.compute(df)
        activity, references = activities.compute(extracted_features)

        df = ground_truth.join(activity, how='left')
        df.dropna(subset=['activity'], inplace=True)
        df['id'] = thigh.stem
        results.append(df)

        if trunk_sensor:
            trunk = thigh.parent.parent / 'trunk' / thigh.name

            if not trunk.exists():
                print(f'Trunk file does not exist for {thigh.name}, skipping.')
                continue

            trunk = pd.read_parquet(trunk)

            features_trunk = features.compute(trunk)
            activity_trunk, references_trunk = activities.compute(extracted_features, trunk=features_trunk)
            trunk_df = ground_truth.join(activity_trunk, how='left')
            trunk_df.dropna(subset=['activity'], inplace=True)
            trunk_df['id'] = thigh.stem
            results_trunk.append(trunk_df)

    results = pd.concat(results)
    results['ground_truth'] = results['ground_truth'].map(map)
    results['activity'] = results['activity'].cat.rename_categories({'move': 'shuffle'})
    results.to_parquet(path / 'processed_thigh.parquet', index=True)

    if trunk_sensor:
        results_trunk = pd.concat(results_trunk)
        results_trunk['ground_truth'] = results_trunk['ground_truth'].map(map)
        results_trunk['activity'] = results_trunk['activity'].cat.rename_categories({'move': 'shuffle'})
        results_trunk.to_parquet(path / 'processed_thigh+trunk.parquet', index=True)
    else:
        results_trunk = pd.DataFrame()

    return results, results_trunk


def get_validity_metrics(true: pd.Series, pred: pd.Series) -> pd.DataFrame:
    stats = classification_report(
        true,
        pred,
        output_dict=True,
        zero_division=np.nan,  # type: ignore
    )

    stats = pd.DataFrame(stats).T.round(4)

    stats = stats.drop(index=['macro avg', 'weighted avg'])
    stats['support'] = (stats['support'] / 3600).round(2)
    stats[['precision', 'recall', 'f1-score']] = stats[['precision', 'recall', 'f1-score']] * 100
    stats = stats.T
    stats.at['support', 'accuracy'] = pd.NA

    return stats


def get_confusion_matrix(
    true: pd.Series,
    pred: pd.Series,
    labels: list[str] | None = None,
    title: str = 'Confusion Matrix',
    color: str = 'purples',
    hide_yaxis: bool = False,
    hide_xaxis_title: bool = False,
) -> alt.LayerChart:
    font = 'sans-serif'
    axis = alt.Axis(
        titleFont=font,
        labelFont=font,
        titleFontSize=14,
        labelFontSize=12,
    )

    if not labels:
        labels = true.unique().tolist() + pred.cat.categories.tolist()
        labels = list(set(labels))  # type: ignore

    matrix = confusion_matrix(true, pred, labels=labels, normalize='true').round(2)
    matrix = np.flip(matrix, axis=1)
    matrix = pd.DataFrame(matrix, index=labels, columns=labels[::-1]).reset_index().melt(id_vars='index')
    matrix.columns = ['Labeled', 'Predicted', 'Value']

    axis_y = None if hide_yaxis else axis
    axis_x_title = None if hide_xaxis_title else 'Predicted'

    cm = (
        alt.Chart(matrix)
        .mark_rect()
        .encode(
            x=alt.X('Predicted:O', sort=labels[::-1], axis=axis, title=axis_x_title),
            y=alt.Y('Labeled:O', sort=labels, axis=axis_y),
            color=alt.Color('Value:Q', scale=alt.Scale(scheme=color), legend=None),  # type: ignore
            tooltip=['Labeled', 'Predicted', 'Value'],
        )
        .properties(title=alt.Title(text=title, fontSize=14, font=font), width=300, height=300)
    )

    text = cm.mark_text(baseline='middle', fontSize=11, font=font).encode(
        text=alt.condition(
            alt.datum.Value == 0,
            alt.value(''),  # If the count is 0, display an empty string
            alt.Text('Value:Q', format='.2f'),  # Otherwise, display the count
        ),
        color=alt.condition(alt.datum.Value > 0.75, alt.value('white'), alt.value('black')),
    )
    chart = cm + text

    return chart


def prepare_metric(df: pd.DataFrame, name: str):
    n = len(df)
    mean = df.mean(axis=0)
    std = df.std(axis=0)
    t = scipy.stats.t.ppf(0.95, df=n - 1)
    e = t * (std / np.sqrt(n))
    lower, upper = mean - e, mean + e
    lower, upper = np.clip(lower, 0, 1), np.clip(upper, 0, 1)

    df = pd.DataFrame({'mean': mean, 'std': std, 'lower': lower, 'upper': upper}).round(2)
    df.index.name = 'value'
    df['table'] = df.apply(lambda x: f'{x["mean"]:.2f} [{x["lower"]:.2f}, {x["upper"]:.2f}]', axis=1)
    df['n'] = n
    df = df.T
    df['metric'] = name

    return df.set_index('metric', append=True)


def report(df: pd.DataFrame, true: str, pred: str, labels: list[str], subject_id: str):
    subjects = df.groupby(subject_id)

    precisions, recalls, f1s, accuracies = [], [], [], []

    for _, data in subjects:
        y_true = data[true].to_numpy()
        y_pred = data[pred].to_numpy()

        precision = precision_score(y_true, y_pred, zero_division=np.nan, average=None, labels=labels)
        recall = recall_score(y_true, y_pred, zero_division=np.nan, average=None, labels=labels)
        f1 = f1_score(y_true, y_pred, zero_division=np.nan, average=None, labels=labels)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        accuracies.append(accuracy_score(y_true, y_pred))

    mean_precision = prepare_metric(pd.DataFrame(precisions, columns=labels), 'precision')
    mean_recall = prepare_metric(pd.DataFrame(recalls, columns=labels), 'recall')
    mean_f1 = prepare_metric(pd.DataFrame(f1s, columns=labels), 'f1')
    mean_accuracy = prepare_metric(
        pd.DataFrame(np.array([accuracies for _ in range(len(labels))]).T, columns=labels), 'accuracy'
    )

    return pd.concat([mean_precision, mean_recall, mean_f1, mean_accuracy], axis=0)
