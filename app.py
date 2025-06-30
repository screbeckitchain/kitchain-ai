
Fix bug in important codebase part
Jun 30
screbeckitchain/kitchain-ai
main
+46
-15

Archive

Share

Create PR


Pick a part of the codebase that seems important and find and fix a bug.
Copy
Summary

Updated the app to accept both CSV and Excel uploads, providing a helper to load either format seamlessly

Mapped uploaded data columns to the model’s expected feature names and ensured the prediction DataFrame matches that order

Testing

✅ python -m py_compile app.py


File (1)

app.py
+46
-15


Diff
Logs
app.py
+46
-15

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(layout="wide")
st.title("AI Matchmaker: Predict Best Area–Brand Fit")

# === Load model ===
@st.cache_resource
def load_model():
    return joblib.load("kitchain_match_model.joblib")

model = load_model()

# === Upload data ===
st.sidebar.header("Upload Your Data")
brands_file = st.sidebar.file_uploader("Upload Brands Excel", type=["xlsx"])
areas_file = st.sidebar.file_uploader("Upload Areas Excel", type=["xlsx"])
# Accept both CSV and Excel to make the demo files work out of the box
brands_file = st.sidebar.file_uploader(
    "Upload Brands File", type=["xlsx", "csv"]
)
areas_file = st.sidebar.file_uploader(
    "Upload Areas File", type=["xlsx", "csv"]
)

def load_table(file):
    """Load CSV or Excel file into a DataFrame."""
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)

if brands_file and areas_file:
    brands_df = pd.read_excel(brands_file)
    areas_df = pd.read_excel(areas_file)
    brands_df = load_table(brands_file)
    areas_df = load_table(areas_file)

    st.success("Files uploaded successfully!")

    st.subheader("Select a Brand")
    brand_names = brands_df["Brand"].tolist()
    selected_brand = st.selectbox("Choose a brand to evaluate:", brand_names)

    brand_row = brands_df[brands_df["Brand"] == selected_brand].iloc[0]

    results = []

    # Determine column names used in the uploaded files
    brand_orders_col = (
        "Orders Per Month" if "Orders Per Month" in brands_df.columns else "MonthlyOrders"
    )
    agg_score_col = (
        "Aggregator Score" if "Aggregator Score" in brands_df.columns else "AggregatorScore"
    )
    area_aov_col = "AOV" if "AOV" in areas_df.columns else "AOV_area"
    order_freq_col = (
        "Order Frequency" if "Order Frequency" in areas_df.columns else "Frequency"
    )
    comp1_col = (
        "Comp Score Cuisine 1" if "Comp Score Cuisine 1" in areas_df.columns else "Competition1"
    )
    comp2_col = (
        "Comp Score Cuisine 2" if "Comp Score Cuisine 2" in areas_df.columns else "Competition2"
    )
    comp3_col = (
        "Comp Score Cuisine 3" if "Comp Score Cuisine 3" in areas_df.columns else "Competition3"
    )

    for _, area_row in areas_df.iterrows():
        # Build feature vector
        # Build feature vector expected by the model
        features = {
            "brand_aov": brand_row["AOV"],
            "brand_orders": brand_row["Orders Per Month"],
            "brand_aggregator_score": brand_row["Aggregator Score"],
            "area_population": area_row["Population"],
            "area_households": area_row["Households"],
            "area_aov": area_row["AOV"],
            "area_order_freq": area_row["Order Frequency"],
            "comp_score_1": area_row["Comp Score Cuisine 1"],
            "comp_score_2": area_row["Comp Score Cuisine 2"],
            "comp_score_3": area_row["Comp Score Cuisine 3"]
            "brand_orders": brand_row[brand_orders_col],
            "agg_position": brand_row[agg_score_col],
            "area_aov": area_row[area_aov_col],
            "order_freq": area_row[order_freq_col],
            "competition_cuisine_1": area_row[comp1_col],
            "competition_cuisine_2": area_row[comp2_col],
            "competition_cuisine_3": area_row[comp3_col],
        }

        X = pd.DataFrame([features])
        # Ensure the feature order matches the model's expectation
        X = pd.DataFrame([features], columns=model.feature_names_in_)
        predicted_score = model.predict(X)[0]

        results.append({
            "Area": area_row["Area"],
            "Score": round(predicted_score, 2)
        })

    results_df = pd.DataFrame(results).sort_values(by="Score", ascending=False)

    st.subheader("Top Matching Areas")
    st.dataframe(results_df, use_container_width=True)

    st.download_button(
        label="Download Results CSV",
        data=results_df.to_csv(index=False).encode("utf-8"),
        file_name="match_results.csv",
        mime="text/csv"
    )
else:
    st.info("Please upload both brand and area Excel files.")
