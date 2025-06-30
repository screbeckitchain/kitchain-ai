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
        MODEL_DIR / "xgb_model.pkl"
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
            import xgboost  # noqa: F401
        except ModuleNotFoundError:
            st.error(
                "The `xgboost` package is required to use the XGBoost model. "
                "Install it with `pip install xgboost`."
            )
            st.stop()
  
    try:
        return joblib.load(model_path)
    except Exception as exc:
        st.error(f"Failed to load the model: {exc}")
        st.stop()


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
        return pd.read_excel(file)
    raise ValueError("Unsupported file type: please upload a CSV or Excel file.")


def resolve_column(df, *names):
    """Return the first matching column name from a list of possibilities."""
    for name in names:
        if name in df.columns:
            return name

    normalized = {c.lower().replace(" ", ""): c for c in df.columns}
    for name in names:
        key = name.lower().replace(" ", "")
        if key in normalized:
            return normalized[key]
    raise KeyError(f"None of the columns {names} were found in {df.columns.tolist()}")


if brands_file and areas_file:
    brands_df = load_table(brands_file)
    areas_df = load_table(areas_file)

    st.success("Files uploaded successfully!")

    st.subheader("Select a Brand")
    brand_names = brands_df["Brand"].tolist()
    selected_brand = st.selectbox("Choose a brand to evaluate:", brand_names)

    brand_row = brands_df[brands_df["Brand"] == selected_brand].iloc[0]

    results = []

    # Map possible column name variations
    brand_orders_col = resolve_column(brands_df, "Orders Per Month", "MonthlyOrders")
    agg_score_col = resolve_column(brands_df, "Aggregator Score", "AggregatorScore")
    area_aov_col = resolve_column(areas_df, "AOV", "AOV_area")
    order_freq_col = resolve_column(areas_df, "Order Frequency", "Frequency")
    comp1_col = resolve_column(
        areas_df,
        "Comp Score Cuisine 1",
        "Competition1",
        "Competition 1",
    )
    comp2_col = resolve_column(
        areas_df,
        "Comp Score Cuisine 2",
        "Competition2",
        "Competition 2",
    )
    comp3_col = resolve_column(
        areas_df,
        "Comp Score Cuisine 3",
        "Competition3",
        "Competition 3",
    )

    for _, area_row in areas_df.iterrows():
        # Build feature vector using detected column names
        features = {
            "area_aov": area_row[area_aov_col],
            "order_freq": area_row[order_freq_col],
            "competition_cuisine_1": area_row[comp1_col],
            "competition_cuisine_2": area_row[comp2_col],
            "competition_cuisine_3": area_row[comp3_col],
            "brand_aov": brand_row["AOV"],
            "agg_position": brand_row[agg_score_col],
            "brand_orders": brand_row[brand_orders_col],
        }

        X = pd.DataFrame([features])
        predicted_score = model.predict(X)[0]

        results.append({"Area": area_row["Area"], "Score": round(predicted_score, 2)})

    results_df = pd.DataFrame(results).sort_values(by="Score", ascending=False)

    st.markdown(
        "Higher scores indicate areas where the brand is likely to perform well."
        "Graphs below show predicted scores by area."
    )

    st.subheader("Top Matching Areas")
    st.dataframe(results_df, use_container_width=True)

    st.download_button(
        label="Download Results CSV",
        data=results_df.to_csv(index=False).encode("utf-8"),
        file_name="match_results.csv",
        mime="text/csv",
    )
else:
    st.info("Please upload both brand and area CSV/Excel files.")
