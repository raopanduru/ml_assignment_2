from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from modeling import MODEL_DIR, MODEL_NAMES, TARGET_COLUMN, build_models, build_pipeline, evaluate_model, metric_row


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
TEST_DATA_PATH = PROJECT_DIR / "test_data.csv"


def read_dataset_from_archive(archive: zipfile.ZipFile) -> pd.DataFrame:
    for name in archive.namelist():
        if name.lower().endswith("bank-full.csv"):
            with archive.open(name) as dataset_file:
                return pd.read_csv(dataset_file, sep=";")

    for name in archive.namelist():
        if name.lower().endswith(".zip"):
            nested_bytes = archive.read(name)
            with zipfile.ZipFile(io.BytesIO(nested_bytes)) as nested_archive:
                try:
                    return read_dataset_from_archive(nested_archive)
                except FileNotFoundError:
                    continue

    raise FileNotFoundError("Could not find bank-full.csv in the UCI archive.")


def load_dataset() -> pd.DataFrame:
    with urlopen(DATASET_URL, timeout=60) as response:
        archive_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        data = read_dataset_from_archive(archive)

    data[TARGET_COLUMN] = data[TARGET_COLUMN].astype(str).str.strip().str.lower()
    return data


def main() -> None:
    data = load_dataset()
    features = data.drop(columns=[TARGET_COLUMN])
    labels = data[TARGET_COLUMN]

    if features.shape[1] < 12 or len(data) < 500:
        raise ValueError("The selected dataset does not meet the task's size requirements.")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    test_data = x_test.copy()
    test_data[TARGET_COLUMN] = y_test
    test_data.to_csv(TEST_DATA_PATH, index=False)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, float]] = {}
    for model_name, estimator in build_models().items():
        pipeline = build_pipeline(estimator, x_train)
        pipeline.fit(x_train, y_train)
        joblib.dump(pipeline, MODEL_DIR / f"{model_name.lower().replace(' ', '_')}.joblib")
        results[model_name] = metric_row(evaluate_model(pipeline, x_test, y_test))
        print(f"Trained and saved: {model_name}")

    metrics = pd.DataFrame.from_dict(results, orient="index").reindex(MODEL_NAMES)
    metrics.index.name = "ML Model Name"
    print("\nHoldout metrics (20% stratified test split):")
    print(metrics.to_string(float_format=lambda value: f"{value:.4f}"))
    print(f"\nDataset: {len(data):,} rows, {features.shape[1]} input features")
    print(f"Test data: {TEST_DATA_PATH}")
    print(f"Saved models: {MODEL_DIR}")


if __name__ == "__main__":
    main()