
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# === CONFIG ===
st.set_page_config(layout="wide")
st.title("Kitchain AI: ML-Based Brand–Area Matcher")

# === LOAD MODEL ===
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# === LOAD DATA ===
areas_df = pd.read_csv("sample_areas.csv")
brands_df = pd.read_csv("sample_brands.csv")

# === SELECT BRAND ===
st.subheader("Select a Brand to Analyze")
selected_brand = st.selectbox("Brand", brands_df["Brand"].unique())

brand_row = brands_df[brands_df["Brand"] == selected_brand].iloc[0]

# === PREDICT ===
X = []
area_names = []

for _, area in areas_df.iterrows():
    X.append([
        area["Population"],
        area["Households"],
        area["Competition1"],
        area["Competition2"],
        area["Competition3"],
        area["AOV_area"],
        area["Frequency"],
        brand_row["AOV"],
        brand_row["AggregatorScore"],
        brand_row["MonthlyOrders"]
    ])
    area_names.append(area["Area"])

predictions = model.predict(X)
results_df = pd.DataFrame({
    "Area": area_names,
    "Score": predictions
}).sort_values(by="Score", ascending=False)

# === VISUAL OUTPUT ===
st.subheader("Top 3 Best Areas")
st.dataframe(results_df.head(3), use_container_width=True)

st.subheader("All Area Scores")
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(data=results_df, y="Area", x="Score", palette="Greys_r", ax=ax)
ax.set_title(f"Predicted Scores for '{selected_brand}'", fontsize=14)
ax.set_xlabel("Score", fontsize=12)
ax.set_ylabel("Area", fontsize=12)
st.pyplot(fig)
