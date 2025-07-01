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
            model.load_model(model_path)
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
    if file is None:
        return None

    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    elif name.endswith((".xls", ".xlsx")):
        return pd.read_excel(file)
    else:
        st.error(f"Unsupported file type: {name}")
        st.stop()


def get_data(brands_file, areas_file):
    """Load uploaded files or fall back to sample data."""
    if brands_file is None:
        brands = pd.read_csv(MODEL_DIR / "sample_brands.csv")
    else:
        brands = load_table(brands_file)

    if areas_file is None:
        areas = pd.read_csv(MODEL_DIR / "sample_areas.csv")
    else:
        areas = load_table(areas_file)

    return brands, areas


def build_features(brands: pd.DataFrame, areas: pd.DataFrame) -> pd.DataFrame:
    """Return feature matrix and pair identifiers."""
    rows = []
    pairs = []
    for _, b in brands.iterrows():
        for _, a in areas.iterrows():
            rows.append(
                {
                    "area_aov": a["AOV_area"],
                    "order_freq": a["Frequency"],
                    "competition_cuisine_1": a["Competition1"],
                    "competition_cuisine_2": a["Competition2"],
                    "competition_cuisine_3": a["Competition3"],
                    "brand_aov": b["AOV"],
                    "agg_position": b["AggregatorScore"],
                    "brand_orders": b["MonthlyOrders"],
                }
            )
            pairs.append({"Brand": b["Brand"], "Area": a["Area"]})

    X = pd.DataFrame(rows)
    pairs_df = pd.DataFrame(pairs)
    return pairs_df, X


brands_df, areas_df = get_data(brands_file, areas_file)
pairs_df, features = build_features(brands_df, areas_df)

preds = model.predict(features)
results = pairs_df.assign(Score=preds).sort_values("Score", ascending=False)

st.subheader("Top Matches")
st.dataframe(results)
