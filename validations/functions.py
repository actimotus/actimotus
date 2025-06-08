import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.metrics import classification_report, confusion_matrix

CONFUSSION_MATRIX = {
    'true': 'True',
    'predicted': 'Predicted',
}


def get_validity_metrics(true: pd.Series, pred: pd.Series, classes: bool = True) -> pd.DataFrame:
    stats = classification_report(
        true,
        pred,
        output_dict=True,
        zero_division=np.nan,
    )

    stats = pd.DataFrame(stats).T.round(4)

    if not classes:
        stats = stats.loc[['accuracy', 'macro avg', 'weighted avg']]

    return stats


def get_confusion_matrix(true: pd.Series, pred: pd.Series, labels: list[str] | None = None) -> px.imshow:
    if not labels:
        labels = true.unique().tolist() + pred.unique().tolist()
        labels = list(set(labels))

    matrix = confusion_matrix(true, pred, labels=labels, normalize='true').round(4) * 100
    matrix = np.flip(matrix, axis=1)
    y_labels = labels
    x_labels = labels[::-1]

    fig = px.imshow(
        matrix,
        text_auto=True,
        x=x_labels,
        y=y_labels,
        labels=dict(
            x=CONFUSSION_MATRIX['predicted'],
            y=CONFUSSION_MATRIX['true'],
            color='Recall (%)',
        ),
    )
    fig.update_layout(
        title=None,
        coloraxis_showscale=False,
        height=400,
        width=400,
        margin=dict(l=60, r=60, t=60, b=60),
        font_family='Verdana',
        template='plotly_white',
    )

    return fig
