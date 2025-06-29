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

if brands_file and areas_file:
    brands_df = pd.read_excel(brands_file)
    areas_df = pd.read_excel(areas_file)

    st.success("Files uploaded successfully!")

    st.subheader("Select a Brand")
    brand_names = brands_df["Brand"].tolist()
    selected_brand = st.selectbox("Choose a brand to evaluate:", brand_names)

    brand_row = brands_df[brands_df["Brand"] == selected_brand].iloc[0]

    results = []

    for _, area_row in areas_df.iterrows():
        # Build feature vector
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
        }

        X = pd.DataFrame([features])
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
