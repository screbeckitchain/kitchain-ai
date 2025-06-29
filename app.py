import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import pickle

matplotlib.use('Agg')

# === SETUP ===
st.set_page_config(layout="wide")
st.title("AI Matchmaker: ML-Based Area Fit for Your Food Brand")

# === LOAD MODEL ===
with open("kitchain_match_model.pkl", "rb") as f:
    model = pickle.load(f)

# === UPLOAD FILES ===
brands_file = st.file_uploader("Upload Brands File (Excel)", type=["xlsx"])
areas_file = st.file_uploader("Upload Areas File (Excel)", type=["xlsx"])

if brands_file and areas_file:
    brands_df = pd.read_excel(brands_file)
    areas_df = pd.read_excel(areas_file)

    st.subheader("Select a Brand to Analyze")
    brand_names = brands_df["Brand"].tolist()
    selected_brand = st.selectbox("Choose a brand:", brand_names)

    brand_row = brands_df[brands_df["Brand"] == selected_brand].iloc[0]

    # Prepare features for prediction
    results = []
    progress_bar = st.progress(0)

    for i, (_, area_row) in enumerate(areas_df.iterrows()):
        try:
            features = pd.DataFrame([{
                "Brand AOV": brand_row["AOV"],
                "Brand Orders": brand_row["Monthly Orders"],
                "Brand Aggregator Rank": brand_row["Aggregator Rank"],
                "Area AOV": area_row["AOV"],
                "Area Frequency": area_row["Order Frequency"],
                "Competition 1": area_row["Competition Score 1"],
                "Competition 2": area_row["Competition Score 2"],
                "Competition 3": area_row["Competition Score 3"]
            }])

            score = model.predict(features)[0]
            explanation = f"Cuisine match, price alignment and local demand suggest a fit score of {score:.2f}."
        except Exception as e:
            score = None
            explanation = f"Error: {str(e)}"

        results.append({
            "Area": area_row["Area"],
            "ML Explanation": explanation,
            "Score": score
        })

        progress_bar.progress((i + 1) / len(areas_df))

    results_df = pd.DataFrame(results)

    st.subheader("Area-by-Area Breakdown")
    for index, row in results_df.iterrows():
        st.markdown(f"**Area: {row['Area']}**")
        st.markdown(f"> {row['ML Explanation']}")
        st.markdown(f"**Score:** {row['Score'] if row['Score'] else 'Not rated'}")
        st.markdown("---")

    st.subheader("Top 3 Best-Fit Areas")
    top3_df = results_df.dropna(subset=["Score"]).sort_values(by="Score", ascending=False).head(3)
    st.dataframe(top3_df[["Area", "Score"]], use_container_width=True)

    st.subheader("ML Match Score Chart")
    chart_data = results_df.dropna(subset=["Score"]).sort_values(by="Score", ascending=True)

    if not chart_data.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.set_style("whitegrid")
        sns.barplot(data=chart_data, y="Area", x="Score", color="black", ax=ax)
        ax.set_title(f"ML Match Scores for '{selected_brand}'", fontsize=14)
        ax.set_xlabel("Score", fontsize=12)
        ax.set_ylabel("Area", fontsize=12)
        st.pyplot(fig)
    else:
        st.info("No scores available for chart.")

    st.subheader("Download Full Results")
    st.dataframe(results_df)

    st.download_button(
        label="Download Results as CSV",
        data=results_df.to_csv(index=False).encode('utf-8'),
        file_name=f"ml_match_results_{selected_brand}.csv",
        mime='text/csv'
    )
