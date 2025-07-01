from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(layout="wide")
st.title("AI Matchmaker: Predict Best Area–Brand Fit")

# === Load model ===
MODEL_DIR = Path(__file__).resolve().parent
WEIGHTS_FILE = MODEL_DIR / "feature_weights_full.xlsx"


@st.cache_data
def load_weights(path: Path = WEIGHTS_FILE) -> dict:
    """Return normalized feature weights used for explanations."""
    df = pd.read_excel(path, header=None)
    df.columns = ["feature", "weight"]
    df = df.dropna()
    df = df[df["feature"] != "Feature"]
    weights = dict(zip(df["feature"], df["weight"]))

    used_features = {
        "area_aov",
        "order_freq",
        "competition_cuisine_1",
        "competition_cuisine_2",
        "competition_cuisine_3",
        "brand_aov",
        "agg_position",
        "brand_orders",
    }
    weights = {k: v for k, v in weights.items() if k in used_features}

    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


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
    """Load uploaded files or fall back to sample data.

    Validates that required columns are present and applies fallback column
    names when possible.
    """
    if brands_file is None:
        brands = pd.read_csv(MODEL_DIR / "sample_brands.csv")
    else:
        brands = load_table(brands_file)

    if areas_file is None:
        areas = pd.read_csv(MODEL_DIR / "sample_areas.csv")
    else:
        areas = load_table(areas_file)

    # Apply fallback column names if provided
    fallback_cols = {"Brand_Cuisine": "Cuisine"}
    for df in (brands, areas):
        for alt, canon in fallback_cols.items():
            if alt in df.columns and canon not in df.columns:
                df.rename(columns={alt: canon}, inplace=True)

    required_brand_cols = {
        "Brand",
        "Cuisine",
        "AOV",
        "AggregatorScore",
        "MonthlyOrders",
    }
    required_area_cols = {
        "Area",
        "AOV_area",
        "Top1Cuisine",
        "Top2Cuisine",
        "Top3Cuisine",
        "Frequency",
        "Competition1",
        "Competition2",
        "Competition3",
    }

    missing_brand = required_brand_cols - set(brands.columns)
    missing_area = required_area_cols - set(areas.columns)

    if missing_brand or missing_area:
        msgs = []
        if missing_brand:
            msgs.append("Brands: " + ", ".join(sorted(missing_brand)))
        if missing_area:
            msgs.append("Areas: " + ", ".join(sorted(missing_area)))
        message = "Missing required column(s): " + "; ".join(msgs)
        st.error(message)
        raise ValueError(message)


    return brands, areas


def build_features(brands: pd.DataFrame, areas: pd.DataFrame) -> pd.DataFrame:
    """Return feature matrix and pair identifiers."""
    weights_df = pd.read_excel(WEIGHTS_FILE, header=None, names=["feature", "weight"])
    weights_df = weights_df.dropna()
    weights_df.columns = ["feature", "weight"]
    weights_df = weights_df[weights_df["feature"] != "Feature"]
    weights = dict(zip(weights_df["feature"], weights_df["weight"]))

    aov_w = weights.get("aov_alignment_score", 1)
    
    thr_vals = (
        weights_df[
            weights_df["feature"].str.startswith("AOV GRADE")
            & ~weights_df["feature"].str.contains("Score")
        ]["weight"].astype(float).tolist()
    )
    score_vals = (
        weights_df[
            weights_df["feature"].str.startswith("AOV GRADE")
            & weights_df["feature"].str.contains("Score")
        ]["weight"].astype(float).tolist()
    )
    cuisine_vals = (
        weights_df[weights_df["feature"].str.startswith("cuisine_match_score")][
            "weight"
        ]
        .astype(float)
        .tolist()
    )
    if len(thr_vals) < 4:
        thr_vals += [0.1, 0.2, 0.3, 0.4][len(thr_vals) :]
    if len(score_vals) < 4:
        score_vals += [10, 6, 4, 0][len(score_vals) :]
    if len(cuisine_vals) < 3:
        cuisine_vals += [10, 8, 6][len(cuisine_vals) :]

    rows = []
    pairs = []
    for _, b in brands.iterrows():
        for _, a in areas.iterrows():
            diff_ratio = abs(b["AOV"] - a["AOV_area"]) / b["AOV"]
            if diff_ratio <= thr_vals[0]:
                aov_base = score_vals[0]
            elif diff_ratio <= thr_vals[1]:
                aov_base = score_vals[1]
            elif diff_ratio <= thr_vals[2]:
                aov_base = score_vals[2]
            else:
                aov_base = score_vals[3]
            aov_score = aov_base * aov_w

            if b["Cuisine"] == a["Top1Cuisine"]:
                cuisine_base = cuisine_vals[0]
            elif b["Cuisine"] == a["Top2Cuisine"]:
                cuisine_base = cuisine_vals[1]
            elif b["Cuisine"] == a["Top3Cuisine"]:
                cuisine_base = cuisine_vals[2]
            else:
                cuisine_base = 0
            cuisine_score = cuisine_base * cuisine_w

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
                    "aov_alignment_score": aov_score,
                    "cuisine_match_score": cuisine_score,
                }
            )
            pairs.append({"Brand": b["Brand"], "Area": a["Area"]})

    X = pd.DataFrame(rows)
    pairs_df = pd.DataFrame(pairs)
    return pairs_df, X

