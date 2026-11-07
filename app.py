from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from modeling import MODEL_DIR, MODEL_NAMES, TARGET_COLUMN, evaluate_model, metric_row


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_FILES = {
    model_name: MODEL_DIR / f"{model_name.lower().replace(' ', '_')}.joblib"
    for model_name in MODEL_NAMES
}
METRIC_NAMES = ("Accuracy", "AUC", "Precision", "Recall", "F1", "MCC")


st.set_page_config(
    page_title="Bank Marketing Model Lab",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root { --ink: #17231f; --muted: #64736c; --accent: #176b53; --coral: #c85c43; }
      .stApp { background: linear-gradient(135deg, #f8faf8 0%, #eff5f1 100%); color: var(--ink); }
      [data-testid="stSidebar"] { background: #e8f0eb; border-right: 1px solid #d5e0d9; }
      h1, h2, h3 { color: var(--ink); }
      div[data-testid="stMetric"] { background: white; border: 1px solid #dce5df; border-radius: 6px; padding: 14px; }
      div[data-testid="stMetricLabel"] { color: var(--muted); }
      .eyebrow { color: var(--accent); font-size: .78rem; font-weight: 700; text-transform: uppercase; }
      .subhead { color: var(--muted); max-width: 760px; }
      div.stButton > button[kind="primary"] { background: var(--accent); border-color: var(--accent); }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)


def load_input_data(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.read_csv(PROJECT_DIR / "test_data.csv")
    return pd.read_csv(uploaded_file)


def show_model_details(model_name: str, result: dict) -> None:
    st.subheader(model_name)
    metric_columns = st.columns(6)
    for column, metric in zip(metric_columns, METRIC_NAMES):
        value = result[metric]
        column.metric(metric, "N/A" if pd.isna(value) else f"{value:.4f}")

    confusion = pd.DataFrame(
        result["Confusion matrix"],
        index=["Actual no", "Actual yes"],
        columns=["Predicted no", "Predicted yes"],
    )
    report = pd.DataFrame(result["Classification report"]).T
    left, right = st.columns([1, 1.4])
    with left:
        st.caption("Confusion matrix")
        st.dataframe(confusion, width="stretch")
    with right:
        st.caption("Classification report")
        st.dataframe(report.round(3), width="stretch")


st.markdown('<p class="eyebrow">UCI Bank Marketing · Binary classification</p>', unsafe_allow_html=True)
st.title("Bank Marketing Model Lab")
st.markdown(
    '<p class="subhead">Evaluate five saved classifiers against labeled campaign data. '
    'Models are trained once; uploaded test data is only used for scoring.</p>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Evaluation")
    uploaded_file = st.file_uploader("Upload labeled test data (CSV)", type=["csv"])
    selected_model = st.selectbox("Model", ["Compare all models", *MODEL_NAMES])
    st.download_button(
        "Download sample test data",
        data=(PROJECT_DIR / "test_data.csv").read_bytes(),
        file_name="test_data.csv",
        mime="text/csv",
        width="stretch",
    )
    evaluate_clicked = st.button("Evaluate data", type="primary", width="stretch")
    st.caption("CSV must include all 16 input features and the target column `y` (yes/no).")

if not evaluate_clicked:
    st.info("Choose a model and run evaluation. The included stratified test sample is used unless you upload a CSV.")
    st.stop()

try:
    data = load_input_data(uploaded_file)
except Exception as error:
    st.error(f"Could not read the CSV file: {error}")
    st.stop()

if len(data) == 0:
    st.error("The uploaded CSV contains no rows.")
    st.stop()
if len(data) > 20000:
    st.error("Please upload test data with no more than 20,000 rows for this free-tier app.")
    st.stop()
if TARGET_COLUMN not in data.columns:
    st.error(f"The uploaded CSV must include the target column `{TARGET_COLUMN}`.")
    st.stop()

labels = data[TARGET_COLUMN].astype(str).str.strip().str.lower()
if not labels.isin(["yes", "no"]).all():
    st.error("Target column `y` must contain only `yes` and `no` labels.")
    st.stop()

try:
    reference_model = load_model(str(MODEL_FILES[MODEL_NAMES[0]]))
    feature_names = list(reference_model.feature_names_in_)
    missing_columns = [column for column in feature_names if column not in data.columns]
    if missing_columns:
        st.error("Missing required input features: " + ", ".join(missing_columns))
        st.stop()
    features = data[feature_names]

    with st.spinner("Scoring the test data with saved models..."):
        model_names = MODEL_NAMES if selected_model == "Compare all models" else (selected_model,)
        results = {
            model_name: evaluate_model(
                load_model(str(MODEL_FILES[model_name])), features, labels
            )
            for model_name in model_names
        }
except Exception as error:
    st.error(f"Evaluation could not be completed: {error}")
    st.stop()

st.caption(f"Scored {len(data):,} rows · Positive class: `yes` · No model was retrained")

if selected_model == "Compare all models":
    comparison = pd.DataFrame(
        {model_name: metric_row(result) for model_name, result in results.items()}
    ).T
    comparison.index.name = "ML Model Name"
    comparison = comparison.sort_values("F1", ascending=False)
    st.subheader("Model comparison")
    st.dataframe(comparison.style.format("{:.4f}"), width="stretch")
    st.caption("The winner is ranked by F1 for the positive `yes` class; the dataset is class-imbalanced.")
    tabs = st.tabs(list(results))
    for tab, (model_name, result) in zip(tabs, results.items()):
        with tab:
                show_model_details(model_name, result)
else:
            show_model_details(selected_model, results[selected_model])
