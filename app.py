
import streamlit as st
import pandas as pd
import joblib

st.title("Kitchain: Подбор зон для фуд-брендов")
st.write("Загрузите файлы Excel с данными о брендах и зонах, чтобы получить прогнозы соответствия брендов и локаций.")

brand_file = st.file_uploader("Excel файл с брендами", type=["xlsx"])
zone_file = st.file_uploader("Excel файл с зонами", type=["xlsx"])

if brand_file is not None and zone_file is not None:
    try:
        brand_df = pd.read_excel(brand_file)
        zone_df = pd.read_excel(zone_file)
    except Exception as e:
        st.error(f"Ошибка чтения файлов: {e}")
    else:
        try:
            model = joblib.load("kitchain_match_model.joblib")
        except Exception as e:
            st.error(f"Ошибка загрузки ML-модели: {e}")
        else:
            results = []
            for _, brand in brand_df.iterrows():
                for _, zone in zone_df.iterrows():
                    rank_score = 11 - int(brand["Aggregator Position"])
                    b_orders = brand["Avg Monthly Orders"]
                    b_AOV = brand["AOV"]
                    z_AOV = zone["AOV"]
                    z_freq = zone["Order Frequency"]
                    z_pop = zone["Population"]
                    z_hh = zone["Households"]
                    match_rank = 0
                    comp_val = 3
                    for ci in [1, 2, 3]:
                        cuisine_col = f"Top{ci} Cuisine"
                        comp_col = f"Comp_top{ci}"
                        if cuisine_col in zone_df.columns and comp_col in zone_df.columns:
                            if str(brand["Cuisine"]).lower() == str(zone[cuisine_col]).lower():
                                match_rank = ci
                                comp_val = zone[comp_col]
                                break
                    features = [
                        b_orders,
                        rank_score,
                        b_AOV,
                        z_pop,
                        z_hh,
                        z_AOV,
                        z_freq,
                        match_rank,
                        comp_val
                    ]
                    pred_score = model.predict([features])[0]
                    pred_score = min(max(pred_score, 1), 10)
                    pred_score = round(pred_score, 1)
                    results.append({
                        "Brand": brand["Name"],
                        "Zone": zone["Name"],
                        "Predicted Score": pred_score
                    })
            res_df = pd.DataFrame(results)
            if res_df.empty:
                st.write("Нет результатов для отображения.")
            else:
                res_df.sort_values(["Brand", "Predicted Score"], ascending=[True, False], inplace=True)
                st.subheader("Результаты прогноза")
                st.dataframe(res_df, use_container_width=True)
                top_n = 3
                summary_data = []
                for brand_name, group in res_df.groupby("Brand"):
                    top_zones = group.nlargest(top_n, "Predicted Score")
                    for _, row in top_zones.iterrows():
                        summary_data.append({
                            "Brand": brand_name,
                            "Top Zone": row["Zone"],
                            "Score": row["Predicted Score"]
                        })
                summary_df = pd.DataFrame(summary_data)
                st.subheader(f"Топ-{top_n} зон для каждого бренда")
                st.dataframe(summary_df, use_container_width=True)
                st.subheader("Визуализация: сравнение зон по брендам")
                brands = res_df["Brand"].unique()
                tabs = st.tabs(list(brands))
                for i, brand_name in enumerate(brands):
                    brand_data = res_df[res_df["Brand"] == brand_name].sort_values("Predicted Score", ascending=False)
                    chart_data = brand_data.set_index("Zone")["Predicted Score"]
                    with tabs[i]:
                        st.write(f"**{brand_name}** — баллы по зонам:")
                        st.bar_chart(chart_data)
                csv_data = res_df.to_csv(index=False).encode('utf-8')
                st.download_button("Скачать результаты в CSV", data=csv_data, file_name="zone_match_results.csv", mime="text/csv")
