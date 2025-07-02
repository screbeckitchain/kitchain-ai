from pathlib import Path
import pandas as pd
import streamlit as st

try:
    import openai  # type: ignore
except Exception:
    openai = None

from app import load_table, _get_openai_key

APP_DIR = Path(__file__).resolve().parents[1]

st.title("Brand Analysis")

uploaded_file = st.file_uploader("Upload Brands File", type=["csv", "xls", "xlsx"])
if uploaded_file is None:
    brands_df = pd.read_csv(APP_DIR / "sample_brands.csv")
else:
    brands_df = load_table(uploaded_file)

if brands_df.empty:
    st.error("No brand data available")
    st.stop()

brand_names = sorted(brands_df["Brand"].unique())
brand_choice = st.selectbox("Choose a brand", brand_names)

if brand_choice:
    row = brands_df[brands_df["Brand"] == brand_choice].iloc[0]

    def generate_brand_analysis(brow: pd.Series) -> str:
        """Return a detailed brand analysis using OpenAI."""
        api_key = _get_openai_key()
        if openai is None:
            st.error("The openai package is required for analysis.")
            return ""
        if not api_key:
            st.warning("OpenAI API key not configured; analysis disabled.")
            return ""

        prompt = (
            "Brand: {brand} ({cuisine}) has an AOV of {aov}, aggregator score {agg} "
            "and {orders} monthly orders. Provide a short paragraph analyzing the "
            "brand's strengths, potential challenges and opportunities for "
            "expansion.".format(
                brand=brow["Brand"],
                cuisine=brow["Cuisine"],
                aov=brow["AOV"],
                agg=brow["AggregatorScore"],
                orders=brow["MonthlyOrders"],
            )
        )
        try:
            if hasattr(openai, "OpenAI"):
                client = openai.OpenAI(api_key=api_key)
                resp = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                )
            else:
                openai.api_key = api_key
                resp = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                )
        except Exception as exc:
            st.error(f"OpenAI request failed: {exc}")
            return ""
        return resp.choices[0].message.content.strip()

    analysis = generate_brand_analysis(row)
    if analysis:
        st.write(analysis)
