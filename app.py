import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.set_page_config(layout="wide")
st.title("Kitchain AI Matcher - ML Powered")

@st.cache_resource
def load_model():
    return joblib.load("kitchain_match_model.joblib")

model = load_model()

# === FILE UPLOAD ===
st.sidebar.header("Upload Input Files")
brand_file = st.sidebar.file_uploader("Upload Brands File (Excel)", type=["xlsx"])
area_file = st.sidebar.file_uploader("Upload Areas File (Excel)", type=["xlsx"])

if brand_file and area_file:
    brands_df = pd.read_excel(brand_file)
    areas_df = pd.read_excel(area_file)

    brand_names = brands_df["Brand"].tolist()
    selected_brand = st.selectbox("Select a brand to analyze:", brand_names)
    brand_row = brands_df[brands_df["Brand"] == selected_brand].iloc[0]

    features = []
    for _, area in areas_df.iterrows():
        cuisine_match = int(brand_row['Cuisine'] in area['Top Cuisines'])
        top_cuisine_idx = [area['Top Cuisine 1'], area['Top Cuisine 2'], area['Top Cuisine 3']].index(brand_row['Cuisine']) if brand_row['Cuisine'] in [area['Top Cuisine 1'], area['Top Cuisine 2'], area['Top Cuisine 3']] else -1
        cuisine1_comp = area['Cuisine 1 Competition'] if brand_row['Cuisine'] == area['Top Cuisine 1'] else 5
        cuisine2_comp = area['Cuisine 2 Competition'] if brand_row['Cuisine'] == area['Top Cuisine 2'] else 5
        cuisine3_comp = area['Cuisine 3 Competition'] if brand_row['Cuisine'] == area['Top Cuisine 3'] else 5

        features.append({
            "Area": area["Area"],
            "Cuisine Match": cuisine_match,
            "AOV Diff": abs(brand_row['AOV'] - area['AOV']),
            "Order Freq": area['Order Frequency'],
            "Cuisine Competition": min(cuisine1_comp, cuisine2_comp, cuisine3_comp),
            "Population": area['Population'],
            "Households": area['Households']
        })

    features_df = pd.DataFrame(features)
    input_X = features_df.drop(columns=["Area"])
    predictions = model.predict(input_X)
    features_df["Match Score"] = predictions

    st.subheader("Top 3 Recommended Areas")
    top3 = features_df.sort_values("Match Score", ascending=False).head(3)
    st.dataframe(top3)

    st.subheader("All Area Scores")
    st.dataframe(features_df)

    st.subheader("Match Score Chart")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=features_df.sort_values("Match Score", ascending=False), x="Match Score", y="Area", palette="gray", ax=ax)
    ax.set_title(f"Match Score by Area for '{selected_brand}'")
    st.pyplot(fig)

    st.download_button("Download Scores", data=features_df.to_csv(index=False), file_name="area_scores.csv")

else:
    st.info("Please upload both brand and area Excel files to proceed.")
