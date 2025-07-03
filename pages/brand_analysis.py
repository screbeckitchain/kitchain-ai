from pathlib import Path
import pandas as pd
import streamlit as st

try:
    import openai  # type: ignore
except Exception:
    openai = None

from app import (
    load_table,
    _get_openai_key,
    build_features,
    load_model,
    load_max_score,
    generate_explanation,
)

APP_DIR = Path(__file__).resolve().parents[1]

st.title("Brand Fit Across Areas")

brand_file = st.sidebar.file_uploader(
    "Upload Brands File", type=["csv", "xls", "xlsx"], key="brand_file_upload"
)
area_file = st.sidebar.file_uploader(
    "Upload Areas File", type=["csv", "xls", "xlsx"], key="area_file_upload"
)

if brand_file is None:
    brands_df = pd.read_csv(APP_DIR / "sample_brands.csv")
else:
    brands_df = load_table(brand_file)

if area_file is None:
    areas_df = pd.read_csv(APP_DIR / "sample_areas.csv")
else:
    areas_df = load_table(area_file)

if brands_df.empty or areas_df.empty:
    st.error("Brand or area data not available")
    st.stop()

brand_names = sorted(brands_df["Brand"].unique())
brand_choice = st.selectbox("Choose a brand", brand_names)

model_choice = st.sidebar.selectbox(
    "Model type", ["RandomForest", "XGBoost"], index=1
)
model = load_model(use_xgb=model_choice == "XGBoost")

if brand_choice:
    brand_row = brands_df[brands_df["Brand"] == brand_choice].iloc[[0]]
    pairs_df, feature_df = build_features(
        brand_row,
        areas_df,
        use_two_feature=model_choice == "XGBoost",
    )
    preds = model.predict(feature_df)
    max_score = load_max_score()

    results = pairs_df.copy()
    results["Score (%)"] = (preds / max_score * 100).round(1)
    results["Explanation"] = ""

    for idx, res_row in results.iterrows():
        area_row = areas_df[areas_df["Area"] == res_row["Area"]].iloc[0]
        results.at[idx, "Explanation"] = generate_explanation(
            brand_row.iloc[0],
            area_row,
            results.at[idx, "Score (%)"],
        )

        # Sort areas by descending score for easier analysis
    results = results.sort_values(by="Score (%)", ascending=False, ignore_index=True)

    st.subheader("Area Fit Scores")
    st.dataframe(results, use_container_width=True)

    st.subheader("Score by Area")
    chart_data = results.set_index("Area")["Score (%)"]
    st.bar_chart(chart_data)
