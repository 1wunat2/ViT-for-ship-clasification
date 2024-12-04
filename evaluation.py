"""
evaluation.py contains all of the function for evaluation, creating the confusion matrix and getting
the evaluation metrics
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    recall_score,
    precision_score,
    accuracy_score,
)

def conf_matrix_plot(cf_matrix: np.ndarray, title: str = ""):
    """
    Return matplotlib fig of confusion matrix
    """
    fig, axs = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=range(5),
        yticklabels=range(5),
        ax=axs,
    )
    fig.suptitle(title)
    return fig


def evaluation_metrics(y_true: np.ndarray, y_preds: np.ndarray):
    """
    Return a dictionary of the evaluation metrics calculated
    Accuracy, f1 score, precision and recall
    """
    conf_matrix = confusion_matrix(y_true, y_preds)
    accuracy, f1, precision, recall = (
        accuracy_score(y_true, y_preds),
        f1_score(y_true, y_preds, zero_division=0.0, average="macro"),
        precision_score(y_true, y_preds, zero_division=0.0, average="macro"),
        recall_score(y_true, y_preds, zero_division=0.0, average="macro"),
    )
    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
    }, conf_matrix