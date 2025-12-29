import streamlit as st
import pandas as pd
import altair as alt

from trend_demo_api import TrendRepair

st.set_page_config(page_title="Trend Deviation Repair Demo", layout="wide")
st.title("Trend Deviation Repair Demo")


# Helpers for stable colors.
def _hex_to_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

def _rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)

def _lerp(a, b, t):
    return int(round(a + (b - a) * t))

def _interp_hex(c1: str, c2: str, t: float) -> str:
    r1, g1, b1 = _hex_to_rgb(c1)
    r2, g2, b2 = _hex_to_rgb(c2)
    return _rgb_to_hex((_lerp(r1, r2, t), _lerp(g1, g2, t), _lerp(b1, b2, t)))

def _make_step_palette(n: int):
    # Light -> dark green gradient
    if n <= 1:
        return ["#31a354"]
    c_light = "#a1d99b"
    c_dark = "#006d2c"
    return [_interp_hex(c_light, c_dark, i / (n - 1)) for i in range(n)]


# Column layout.
controls_col, output_col = st.columns([1, 3], gap="large")

with controls_col:
    st.header("Controls")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is None:
        st.info("Upload a CSV to begin.")
        st.stop()

    df = pd.read_csv(uploaded_file, index_col=0)

    with st.expander("Preview data (first 100 rows)", expanded=False):
        st.dataframe(df.head(100), use_container_width=True)

    group_attr = st.selectbox("Grouping attribute", df.columns)
    agg_attr = st.selectbox("Aggregation attribute", df.columns)
    agg_func = st.selectbox("Aggregation function", ["sum", "avg", "median", "max"])

    # Reset everything when query definition changes.
    params_key = (uploaded_file.name, group_attr, agg_attr, agg_func)
    if st.session_state.get("params_key") != params_key:
        st.session_state["params_key"] = params_key

        # Persist TrendRepair so step-by-step actually progresses across clicks
        st.session_state["tr_obj"] = TrendRepair(df, agg_func, group_attr, agg_attr)

        # Results
        st.session_state["original_df"] = None
        st.session_state["heur_df"] = None

        # Step-by-step storage
        st.session_state["partial_steps"] = []  # list[DataFrame], step i stored at index i-1

        # Max steps
        st.session_state["max_steps"] = int(df[group_attr].dropna().nunique())

        step_labels = [f"better repair (step {i})" for i in range(1, st.session_state["max_steps"] + 1)]
        step_colors = _make_step_palette(len(step_labels))

        st.session_state["SERIES_DOMAIN"] = ["Original", "Heuristic"] + step_labels
        st.session_state["SERIES_RANGE"] = ["#9ecae1", "#3182bd"] + step_colors  # light blue, dark blue, greens

        st.session_state["cb_show_original"] = True
        st.session_state["cb_show_heuristic"] = True

    tr = st.session_state["tr_obj"]
    max_steps = st.session_state.get("max_steps", 0)
    current_steps = len(st.session_state.get("partial_steps", []))

    st.divider()
    st.subheader("Run")

    if st.button("Run original query", use_container_width=True):
        with st.spinner("Running original query…"):
            st.session_state["original_df"] = tr.run_query()

    if st.button("Find heuristic repair", use_container_width=True):
        with st.spinner("Running heuristic repair…"):
            if st.session_state.get("original_df") is None:
                st.session_state["original_df"] = tr.run_query()

            heur_trend_result, _, _ = tr.run_heuristic()
            st.session_state["heur_df"] = heur_trend_result

    # Limit the button by number of steps
    step_done = (max_steps > 0) and (current_steps >= max_steps)
    if st.button("Find better repairs:", use_container_width=True, disabled=step_done):
        with st.spinner("Computing next partial solution…"):
            if len(st.session_state["partial_steps"]) < st.session_state["max_steps"]:
                intermediate_result = tr.compute_next_partial_solution()
                st.session_state["partial_steps"].append(intermediate_result)

                # Auto-check only newest step; uncheck all previous steps
                new_step = len(st.session_state["partial_steps"])
                for i in range(1, new_step):
                    st.session_state[f"cb_show_step_{i}"] = False
                st.session_state[f"cb_show_step_{new_step}"] = True

    st.caption(f"Better repair steps: {current_steps}/{max_steps}")


with output_col:
    # Live query text
    query_sql = f"SELECT {agg_func.upper()}({agg_attr}) AS value GROUP BY {group_attr}"
    st.markdown("### Query")
    st.info(query_sql)

    st.markdown("### results")
    chart_slot = st.empty()

    st.markdown("#### Display")
    show_original = st.checkbox("Show Original", value=True, key="cb_show_original")
    show_heuristic = st.checkbox("Show Heuristic", value=True, key="cb_show_heuristic")

    partial_steps = st.session_state.get("partial_steps", [])
    show_steps = {}
    for i in range(1, len(partial_steps) + 1):
        key = f"cb_show_step_{i}"
        if key not in st.session_state:
            st.session_state[key] = (i == len(partial_steps))
        show_steps[i] = st.checkbox(f"Show better repair (step {i})", key=key)

    def _to_long(trend_df: pd.DataFrame, series_label: str) -> pd.DataFrame:
        out = trend_df[[group_attr, agg_attr]].copy()
        out.rename(columns={group_attr: "Group", agg_attr: "Value"}, inplace=True)
        out["Group"] = out["Group"].astype(str)
        out["Series"] = series_label
        return out

    frames = []
    original_df = st.session_state.get("original_df")
    heur_df = st.session_state.get("heur_df")

    if show_original and original_df is not None:
        frames.append(_to_long(original_df, "Original"))

    if show_heuristic and heur_df is not None:
        frames.append(_to_long(heur_df, "Heuristic"))

    for i, df_step in enumerate(partial_steps, start=1):
        if show_steps.get(i, False):
            frames.append(_to_long(df_step, f"better repair (step {i})"))

    if not frames:
        chart_slot.info("Run at least one algorithm (and enable it in Display).")
    else:
        data = pd.concat(frames, ignore_index=True)

        if original_df is not None:
            base_groups = original_df[group_attr].astype(str).tolist()
        elif heur_df is not None:
            base_groups = heur_df[group_attr].astype(str).tolist()
        else:
            base_groups = frames[-1]["Group"].tolist()

        group_sort = list(dict.fromkeys(base_groups))
        y_title = f"{agg_func}({agg_attr})"

        SERIES_DOMAIN_ALL = st.session_state["SERIES_DOMAIN"]
        SERIES_RANGE_ALL = st.session_state["SERIES_RANGE"]
        color_map = dict(zip(SERIES_DOMAIN_ALL, SERIES_RANGE_ALL))

        present_set = set(data["Series"].unique().tolist())
        present_series = [s for s in SERIES_DOMAIN_ALL if s in present_set]
        present_colors = [color_map[s] for s in present_series]

        chart = (
            alt.Chart(data)
            .mark_bar()
            .encode(
                x=alt.X("Group:N", sort=group_sort, title=group_attr),
                xOffset=alt.XOffset("Series:N", sort=present_series),
                y=alt.Y("Value:Q", title=y_title),
                color=alt.Color(
                    "Series:N",
                    scale=alt.Scale(domain=present_series, range=present_colors),
                    legend=alt.Legend(orient="top", title=None),
                ),
                tooltip=[
                    alt.Tooltip("Series:N"),
                    alt.Tooltip("Group:N"),
                    alt.Tooltip("Value:Q", format=",.4f"),
                ],
            )
            .properties(height=450)
        )

        chart_slot.altair_chart(chart, use_container_width=True)
