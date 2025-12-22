import streamlit as st
import pandas as pd
from trend_demo_api import TrendRepair


st.title("Trend Deviation Repair Demo")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, index_col=0)
    st.write("Uploaded Data: (first 100 rows)")
    st.dataframe(df.head(100))


    group_attr = st.selectbox("Grouping attribute", df.columns)
    agg_attr = st.selectbox("Aggregation attribute", df.columns)
    agg_func = st.selectbox("Aggregation function", ["sum", "avg", "median", "max"])

    tr = TrendRepair(df, agg_func, group_attr, agg_attr)

    if st.button("Run original query"):
        original = tr.run_query()
        st.bar_chart(
            original,
            x=group_attr,
            y=agg_attr
        )
    # TODO: make the bar chart stay! even when clicking on the other "FInd Repair" button

    if st.button("Find heuristic repair"):
        heur_trend_result, heur_removed_per_group, heur_total_removed = tr.run_heuristic()

        st.bar_chart(
            heur_trend_result,
            x=group_attr,
            y=agg_attr
        )
    # TODO: make the bar chart stay, and make it another color, and make the bars more narrow
    # TODO: Also, can we show them on the same graph?

    # TODO: also compute the optimal solution and show it on the same graph.