from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


TARGET_COLUMN = "y"
POSITIVE_LABEL = "yes"
NEGATIVE_LABEL = "no"
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_NAMES = (
    "Logistic Regression",
    "Decision Tree",
    "K-Nearest Neighbors",
    "Gaussian Naive Bayes",
    "Support Vector Machine",
)


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = features.select_dtypes(exclude=["number"]).columns.tolist()

    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_columns),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_columns,
            ),
        ],
        remainder="drop",
        sparse_threshold=0,
    )


def build_models() -> dict[str, Any]:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1500, class_weight="balanced", random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=42,
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=11, weights="distance", n_jobs=-1
        ),
        "Gaussian Naive Bayes": GaussianNB(),
        "Support Vector Machine": LinearSVC(
            class_weight="balanced", max_iter=5000, random_state=42
        ),
    }


def build_pipeline(model: Any, features: pd.DataFrame) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(features)),
            ("classifier", model),
        ]
    )


def evaluate_model(
    model: Pipeline, features: pd.DataFrame, labels: pd.Series
) -> dict[str, Any]:
    predictions = model.predict(features)
    classes = list(model.classes_)

    if hasattr(model, "predict_proba"):
        positive_scores = model.predict_proba(features)[:, classes.index(POSITIVE_LABEL)]
    else:
        positive_scores = model.decision_function(features)
        if classes.index(POSITIVE_LABEL) == 0:
            positive_scores = -positive_scores

    has_both_classes = labels.nunique() == 2
    auc = roc_auc_score(labels, positive_scores) if has_both_classes else float("nan")

    return {
        "Accuracy": accuracy_score(labels, predictions),
        "AUC": auc,
        "Precision": precision_score(
            labels, predictions, pos_label=POSITIVE_LABEL, zero_division=0
        ),
        "Recall": recall_score(
            labels, predictions, pos_label=POSITIVE_LABEL, zero_division=0
        ),
        "F1": f1_score(labels, predictions, pos_label=POSITIVE_LABEL, zero_division=0),
        "MCC": matthews_corrcoef(labels, predictions),
        "Confusion matrix": confusion_matrix(
            labels, predictions, labels=[NEGATIVE_LABEL, POSITIVE_LABEL]
        ),
        "Classification report": classification_report(
            labels,
            predictions,
            labels=[NEGATIVE_LABEL, POSITIVE_LABEL],
            target_names=[NEGATIVE_LABEL, POSITIVE_LABEL],
            output_dict=True,
            zero_division=0,
        ),
    }


def metric_row(result: dict[str, Any]) -> dict[str, float]:
    return {
        metric: float(result[metric])
        for metric in ("Accuracy", "AUC", "Precision", "Recall", "F1", "MCC")
    }
