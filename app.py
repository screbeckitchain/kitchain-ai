import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- SETTINGS ---
st.set_page_config(layout="wide")
st.title("Kitchain AI Matcher")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return joblib.load("kitchain_match_model.joblib")

model = load_model()

# --- FILE UPLOAD ---
brand_file = st.file_uploader("Upload Brand Data (Excel)", type=["xlsx"])
area_file = st.file_uploader("Upload Area Data (Excel)", type=["xlsx"])

if brand_file and area_file:
    brand_df = pd.read_excel(brand_file)
    area_df = pd.read_excel(area_file)

    selected_brand = st.selectbox("Select a Brand:", brand_df["Brand"].unique())
    brand_row = brand_df[brand_df["Brand"] == selected_brand].iloc[0]

    st.markdown("---")
    st.subheader(f"Match Results for: {selected_brand}")

    # --- PREPROCESS ---
    combined = []
    for _, area_row in area_df.iterrows():
        data = {
            'Brand_Cuisine': brand_row['Cuisine'],
            'Brand_AOV': brand_row['AOV'],
            'Brand_Position': brand_row['Position'],
            'Brand_Orders': brand_row['Orders'],
            'Area_Population': area_row['Population'],
            'Area_Households': area_row['Households'],
            'Top_Nationality_1': area_row['Top Nationality 1'],
            'Top_Nationality_2': area_row['Top Nationality 2'],
            'Top_Nationality_3': area_row['Top Nationality 3'],
            'Top_Cuisine_1': area_row['Top Cuisine 1'],
            'Top_Cuisine_2': area_row['Top Cuisine 2'],
            'Top_Cuisine_3': area_row['Top Cuisine 3'],
            'Area_AOV': area_row['AOV'],
            'Order_Frequency': area_row['Order Frequency'],
            'Competition_Cuisine_1': area_row['Competition 1'],
            'Competition_Cuisine_2': area_row['Competition 2'],
            'Competition_Cuisine_3': area_row['Competition 3']
        }
        combined.append({**data, 'Area': area_row['Area']})

    match_data = pd.DataFrame(combined)

    # --- ENCODING ---
    encoded = pd.get_dummies(match_data.drop(columns=["Area"]), drop_first=True)

    # --- PREDICT ---
    match_data["Score"] = model.predict(encoded)

    # --- RESULTS ---
    match_data_sorted = match_data.sort_values(by="Score", ascending=False)

    # --- VISUALIZATION ---
    st.subheader("Top 3 Best-Fit Areas")
    top3 = match_data_sorted.head(3)[["Area", "Score"]]
    st.dataframe(top3)

    st.subheader("Match Score Chart")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=match_data_sorted, x="Score", y="Area", palette="Blues_d", ax=ax)
    st.pyplot(fig)

    st.subheader("Full Results")
    st.dataframe(match_data_sorted[["Area", "Score"]])

    # --- DOWNLOAD ---
    st.download_button(
        label="Download Match Scores",
        data=match_data_sorted.to_csv(index=False).encode('utf-8'),
        file_name="match_results.csv",
        mime="text/csv"
    )
