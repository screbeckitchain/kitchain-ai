
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(layout="wide")
st.title("Kitchain AI: Brand-Area Match Score")

@st.cache_resource
def load_model():
    return joblib.load("kitchain_match_model.joblib")

model = load_model()

st.sidebar.header("Upload Data")
brand_file = st.sidebar.file_uploader("Upload Brand CSV", type=["csv"])
area_file = st.sidebar.file_uploader("Upload Area CSV", type=["csv"])

if brand_file and area_file:
    brand_df = pd.read_csv(brand_file)
    area_df = pd.read_csv(area_file)

    st.subheader("Input Preview")
    st.write("Brand Data:")
    st.dataframe(brand_df)
    st.write("Area Data:")
    st.dataframe(area_df)

    st.subheader("Predictions")
    merged_df = brand_df.assign(key=1).merge(area_df.assign(key=1), on="key").drop("key", axis=1)

    try:
        X = merged_df.select_dtypes(include=["number"])
        merged_df["Match Score"] = model.predict(X)
        st.dataframe(merged_df)
        st.download_button("Download Predictions", merged_df.to_csv(index=False).encode("utf-8"),
                           file_name="predictions.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
