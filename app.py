import openai
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import re
import seaborn as sns
import matplotlib
matplotlib.use('Agg')

# === SETUP ===
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.set_page_config(layout="wide")

st.title("AI Matchmaker: Best Areas for Your Food Brand")

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
    brand_cuisine = str(brand_row['Cuisine']).strip()

    st.markdown("---")
    st.subheader(f"Match Results for Brand: {selected_brand}")

    results = []
    progress_bar = st.progress(0)
    total_areas = len(areas_df)

    for i, (_, area) in enumerate(areas_df.iterrows()):
        top_cuisines = [c.strip() for c in str(area['Top Cuisines']).split(',')]
        cuisine_match = brand_cuisine in top_cuisines

        prompt = f"""
        You are a food & delivery market analyst. Assess how well the following area fits the given brand.

        ### Brand
        - Name: {brand_row['Brand']}
        - Cuisine: {brand_row['Cuisine']}
        - AOV: {brand_row['AOV']}

        ### Area
        - Name: {area['Area']}
        - Top Cuisines: {area['Top Cuisines']}
        - AOV: {area['AOV']}
        - Order Frequency: {area['Order Frequency']}
        - Cuisine Match: {"Yes" if cuisine_match else "No"}

        Please provide:

        Return the rating as the first line in the format:
        Score: X/10

        Then write your explanation including:
        1. Cuisine match
        2. AOV alignment
        3. Demand level
        4. Any numeric insights or observations
        """

        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            answer = response.choices[0].message.content.strip()

            rating_match = re.search(r'Score:\s*(\d+(\.\d+)?)', answer)
            score = float(rating_match.group(1)) if rating_match else None
        except Exception as e:
            answer = f"Error: {str(e)}"
            score = None

        results.append({
            "Area": area['Area'],
            "GPT Response": answer,
            "Score": score
        })

        progress_bar.progress((i + 1) / total_areas)

    results_df = pd.DataFrame(results)

    st.subheader("Area-by-Area Breakdown")
    for index, row in results_df.iterrows():
        st.markdown(f"**Area: {row['Area']}**")
        st.markdown(f"> {row['GPT Response']}")
        st.markdown(f"**Score:** {row['Score'] if row['Score'] else 'Not rated'}")
        st.markdown("---")

    st.subheader("Top 3 Best-Fit Areas")
    top3_df = results_df.dropna(subset=["Score"]).sort_values(by="Score", ascending=False).head(3)
    st.dataframe(top3_df[["Area", "Score"]], use_container_width=True)

    st.subheader("GPT Match Score Chart")
    chart_data = results_df.dropna(subset=["Score"]).sort_values(by="Score", ascending=True)

    if not chart_data.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.set_style("whitegrid")
        barplot = sns.barplot(data=chart_data, y="Area", x="Score", color="black", ax=ax)
        ax.set_title(f"GPT Match Scores for '{selected_brand}'", fontsize=14)
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
        file_name=f"match_results_{selected_brand}.csv",
        mime='text/csv'
    )
