import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(layout="wide")
st.title("AI Matchmaker: Predict Best Area–Brand Fit")

# === Load model ===
MODEL_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_model(use_xgb: bool = False):
    """Load either the classic scikit-learn model or the XGBoost model."""
    if use_xgb:
        model_path = MODEL_DIR / "xgb_model.pkl"
        try:
            return joblib.load(model_path)
        except Exception:
            st.error(
                "Failed to load the XGBoost model. Install the `xgboost` package"
                " to enable this option."
            )
            st.stop()
    return joblib.load(MODEL_DIR / "kitchain_match_model.joblib")

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
areas_file = st.sidebar.file_uploader(
    "Upload Areas File", type=["csv", "xls", "xlsx"]
)


def load_table(file):
    """Return DataFrame from CSV or Excel file."""
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
@@ -76,66 +81,66 @@ if brands_file and areas_file:

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

        results.append({
            "Area": area_row["Area"],
            "Score": round(predicted_score, 2)
        })

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
        mime="text/csv"
    )
else:
    st.info("Please upload both brand and area CSV/Excel files.")
