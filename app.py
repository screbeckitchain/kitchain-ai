from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(layout="wide")
st.title("AI Matchmaker: Predict Best Area–Brand Fit")

# === Load model ===
MODEL_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_model(use_xgb: bool = False):
    """Load either the classic scikit-learn model or the XGBoost model."""
    model_path = (
        MODEL_DIR / "xgb_model.json"
        if use_xgb
        else MODEL_DIR / "kitchain_match_model.joblib"
    )

    if not model_path.exists():
        st.error(
            f"Model file `{model_path.name}` not found. Please train the model or place "
            "it in the application directory."
        )
        st.stop()

    if use_xgb:
        try:
            import xgboost as xgb  # noqa: F401
        except ModuleNotFoundError:
            st.error(
                "The `xgboost` package is required to use the XGBoost model. "
                "Install it with `pip install xgboost`."
            )
            st.stop()

        try:
            model = xgb.XGBRegressor()
            model.load_model(model_path))
        except Exception as exc:
            st.error(f"Failed to load the XGBoost model: {exc}")
            st.stop()
    else:
        try:
            model = joblib.load(model_path)
        except Exception as exc:
            st.error(f"Failed to load the model: {exc}")
            st.stop()

    return model


st.sidebar.header("Model")
model_choice = st.sidebar.selectbox(
    "Choose model type",
    ["RandomForest", "XGBoost"],
    index=0,
)

model = load_model(use_xgb=model_choice == "XGBoost")

# === Upload data ===
st.sidebar.header("Upload Your Data")
brands_file = st.sidebar.file_uploader(
    "Upload Brands File", type=["csv", "xls", "xlsx"]
)
areas_file = st.sidebar.file_uploader("Upload Areas File", type=["csv", "xls", "xlsx"])


def load_table(file):
    """Return DataFrame from CSV or Excel file."""
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    elif name.endswith((".xls", ".xlsx")):
