import time

import streamlit as st
import pandas as pd
import altair as alt

from trend_demo_api import TrendRepair

st.set_page_config(page_title="Trend Deviation Repair Demo", layout="wide")
st.title("Trend Deviation Repair Demo")


# -------------------------
# Helpers for stable colors
# -------------------------

def _hex_to_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


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
    if n <= 0:
        return []
    if n == 1:
        return ["#31a354"]
    c_light = "#a1d99b"
    c_dark = "#006d2c"
    return [_interp_hex(c_light, c_dark, i / (n - 1)) for i in range(n)]


def _fmt_sec(x):
    if x is None:
        return "—"
    try:
        return f"{float(x):.2f}s"
    except Exception:
        return "—"


def _fmt_int(x):
    if x is None:
        return "—"
    try:
        return str(int(x))
    except Exception:
        return "—"


def _compute_deleted_for_current_step(tr: TrendRepair):
    """
    Compute 'tuples deleted' for the *current* optimal step (right after computing it).

    Important: call this immediately after compute_next_partial_solution(), before any
    additional DP steps are computed, so the internal DP state still matches the step
    we want to measure.
    """
    try:
        if tr.removed_by_heur is None or tr.heur_trend_result is None or tr.inc_dp is None:
            return None

        # After compute_next_partial_solution, computed_dp_so_far has already been incremented.
        # The step we just computed corresponds to i = computed_dp_so_far - 1.
        step_num = int(tr.computed_dp_so_far)
        i = step_num - 1
        if i < 0:
            return None
        if i + 1 >= len(tr.group_keys):
            # This is exactly where TrendRepair would crash (next_group_key undefined).
            return None

        heur_map = tr.heur_trend_result.set_index(tr.grouping_col).to_dict()[tr.aggregation_col]
        next_group_key = tr.group_keys[i + 1]
        cutoff = heur_map.get(next_group_key)

        # Safe to call again for the same i right after it was computed: it won't advance state.
        removed_tuples_up_to_i = tr.inc_dp.compute_up_to_i(i, cutoff)

        remaining_group_keys = tr.group_keys[i + 1 :]
        removed_tuples_from_i_plus_1 = tr.removed_by_heur[
            tr.removed_by_heur[tr.grouping_col].isin(remaining_group_keys)
        ]

        all_removed_index = removed_tuples_up_to_i.index.append(removed_tuples_from_i_plus_1.index)
        return int(len(all_removed_index))
    except Exception:
        return None


def _to_long(trend_df: pd.DataFrame, group_attr: str, agg_attr: str, series_label: str) -> pd.DataFrame:
    out = trend_df[[group_attr, agg_attr]].copy()
    out.rename(columns={group_attr: "Group", agg_attr: "Value"}, inplace=True)
    out["Group"] = out["Group"].astype(str)
    out["Series"] = series_label
    return out


def _render_chart(chart_slot, group_attr: str, agg_attr: str, agg_func: str):
    original_df = st.session_state.get("original_df")
    heur_df = st.session_state.get("heur_df")
    partial_steps = st.session_state.get("partial_steps", [])

    show_original = bool(st.session_state.get("cb_show_original", True))
    show_heuristic = bool(st.session_state.get("cb_show_heuristic", True))

    frames = []
    if show_original and original_df is not None:
        frames.append(_to_long(original_df, group_attr, agg_attr, "Original"))
    if show_heuristic and heur_df is not None:
        frames.append(_to_long(heur_df, group_attr, agg_attr, "Heuristic"))

    for i, df_step in enumerate(partial_steps, start=1):
        if bool(st.session_state.get(f"cb_show_step_{i}", i == len(partial_steps))):
            frames.append(_to_long(df_step, group_attr, agg_attr, f"better repair (step {i})"))

    if not frames:
        chart_slot.info("Run at least one algorithm (and enable it in Display).")
        return

    data = pd.concat(frames, ignore_index=True)

    # Preserve a stable (and intuitive) group order.
    base_df = (
        original_df
        if original_df is not None
        else (heur_df if heur_df is not None else (partial_steps[-1] if partial_steps else None))
    )
    group_sort = []
    if base_df is not None and group_attr in base_df.columns:
        group_sort = list(dict.fromkeys(base_df[group_attr].astype(str).tolist()))

    y_title = f"{agg_func}({agg_attr})"

    domain_all = st.session_state.get("SERIES_DOMAIN", [])
    range_all = st.session_state.get("SERIES_RANGE", [])
    color_map = dict(zip(domain_all, range_all))

    # Only show legend entries for series that are currently visible (checked).
    present_set = set(data["Series"].unique().tolist())
    present_series = [s for s in domain_all if s in present_set]
    present_colors = [color_map[s] for s in present_series]

    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X(
                "Group:N",
                sort=group_sort if group_sort else None,
                title=group_attr,
                axis=alt.Axis(labelAngle=0),
            ),
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


# -------------------------
# Layout: left controls, right output
# -------------------------

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

        st.session_state["tr_obj"] = TrendRepair(df, agg_func, group_attr, agg_attr)

        # Results
        st.session_state["original_df"] = None
        st.session_state["heur_df"] = None

        # Step-by-step storage
        st.session_state["partial_steps"] = []

        # Metrics
        st.session_state["runtime_original"] = None
        st.session_state["runtime_heuristic"] = None
        st.session_state["runtime_steps"] = []
        st.session_state["deleted_original"] = 0
        st.session_state["deleted_heuristic"] = None
        st.session_state["deleted_steps"] = []

        # Clear old step checkboxes (from previous query runs)
        for k in list(st.session_state.keys()):
            if k.startswith("cb_show_step_"):
                del st.session_state[k]

        # Max steps:
        # TrendRepair.compute_next_partial_solution crashes when computed_dp_so_far+1 >= len(group_keys),
        # so the safe maximum number of steps is (num_groups - 1).
        num_groups = len(st.session_state["tr_obj"].group_keys)
        st.session_state["max_steps"] = max(num_groups - 1, 0)

        step_labels = [f"better repair (step {i})" for i in range(1, st.session_state["max_steps"] + 1)]
        step_colors = _make_step_palette(len(step_labels))
        st.session_state["SERIES_DOMAIN"] = ["Original", "Heuristic"] + step_labels
        st.session_state["SERIES_RANGE"] = ["#9ecae1", "#3182bd"] + step_colors  # light blue, dark blue, greens

        st.session_state["cb_show_original"] = True
        st.session_state["cb_show_heuristic"] = True

        st.session_state["run_optimal_seq"] = False
        st.session_state["opt_steps_to_run"] = 3

    tr = st.session_state["tr_obj"]
    max_steps = int(st.session_state.get("max_steps", 0))
    current_steps = len(st.session_state.get("partial_steps", []))

    st.divider()
    st.subheader("Run")

    if st.button("Original data", use_container_width=True):
        with st.spinner("Running original query…"):
            t0 = time.perf_counter()
            st.session_state["original_df"] = tr.run_query()
            st.session_state["runtime_original"] = time.perf_counter() - t0
            st.session_state["deleted_original"] = 0

    if st.button("Heuristic", use_container_width=True):
        with st.spinner("Running heuristic repair…"):
            if st.session_state.get("original_df") is None:
                t0 = time.perf_counter()
                st.session_state["original_df"] = tr.run_query()
                st.session_state["runtime_original"] = time.perf_counter() - t0
                st.session_state["deleted_original"] = 0

            t0 = time.perf_counter()
            heur_trend_result, _, heur_total_removed = tr.run_heuristic()
            st.session_state["heur_df"] = heur_trend_result
            st.session_state["runtime_heuristic"] = time.perf_counter() - t0
            st.session_state["deleted_heuristic"] = int(heur_total_removed)

    # Optimal is step-by-step; cap by max_steps to avoid TrendRepair's UnboundLocalError.
    step_done = (max_steps <= 0) or (current_steps >= max_steps)
    if st.button("Optimal", use_container_width=True, disabled=step_done):
        # Ensure heuristic exists (TrendRepair depends on it anyway); time it if needed.
        if st.session_state.get("heur_df") is None:
            with st.spinner("Running heuristic repair (required for optimal)…"):
                if st.session_state.get("original_df") is None:
                    t0 = time.perf_counter()
                    st.session_state["original_df"] = tr.run_query()
                    st.session_state["runtime_original"] = time.perf_counter() - t0
                    st.session_state["deleted_original"] = 0

                t0 = time.perf_counter()
                heur_trend_result, _, heur_total_removed = tr.run_heuristic()
                st.session_state["heur_df"] = heur_trend_result
                st.session_state["runtime_heuristic"] = time.perf_counter() - t0
                st.session_state["deleted_heuristic"] = int(heur_total_removed)

        # Trigger the automatic 3-step run (with sleeps) on the output side.
        st.session_state["run_optimal_seq"] = True

    st.caption(f"Optimal steps: {current_steps}/{max_steps}")


with output_col:
    # Big, centered query
    query_sql = f"Query: SELECT {agg_func.upper()}({agg_attr}) AS value GROUP BY {group_attr}"
    st.markdown(
        f"""
        <div style="text-align:center; font-size:28px; font-weight:700; color:white; padding:8px 0 2px 0;">
            {query_sql}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Results")
    chart_slot = st.empty()
    status_slot = st.empty()

    # If the user clicked Optimal, run up to 3 steps automatically, with a 3s pause between steps.
    if bool(st.session_state.get("run_optimal_seq", False)):
        tr = st.session_state["tr_obj"]
        steps_to_run = int(st.session_state.get("opt_steps_to_run", 3))
        max_steps = int(st.session_state.get("max_steps", 0))

        for run_idx in range(steps_to_run):
            partial_steps = st.session_state.get("partial_steps", [])
            if len(partial_steps) >= max_steps:
                break

            step_num = len(partial_steps) + 1
            status_slot.info(f"Computing Optimal step {step_num}…")

            t0 = time.perf_counter()
            intermediate_result = tr.compute_next_partial_solution()
            dt = time.perf_counter() - t0

            st.session_state["partial_steps"].append(intermediate_result)
            st.session_state["runtime_steps"].append(dt)

            # Capture tuples-deleted count *for this step* right now (before any further DP steps).
            st.session_state["deleted_steps"].append(_compute_deleted_for_current_step(tr))

            # Auto-check only newest step; uncheck all previous steps
            for i in range(1, step_num):
                st.session_state[f"cb_show_step_{i}"] = False
            st.session_state[f"cb_show_step_{step_num}"] = True

            # Render the updated chart immediately
            _render_chart(chart_slot, group_attr, agg_attr, agg_func)

            # Sleep between steps (but not after the last one we run).
            if run_idx < steps_to_run - 1 and len(st.session_state.get("partial_steps", [])) < max_steps:
                status_slot.info(f"Computed step {step_num}. Next step in 3 seconds…")
                time.sleep(3)

        st.session_state["run_optimal_seq"] = False
        status_slot.empty()

    # -------------------------
    # Display table (checkboxes + metrics)
    # -------------------------
    st.markdown("")

    # Header row
    hcols = st.columns([3, 1.2, 1.2, 2])
    hcols[0].markdown("**Display**")
    hcols[1].markdown("**Runtime**")
    hcols[2].markdown("**Tuples deleted**")
    hcols[3].markdown("**Explanation**")

    # Rows
    r_original = st.columns([3, 1.2, 1.2, 2])
    r_original[0].checkbox("Original", key="cb_show_original")
    r_original[1].write(_fmt_sec(st.session_state.get("runtime_original")))
    r_original[2].write(_fmt_int(st.session_state.get("deleted_original", 0)))
    r_original[3].write("")

    r_heur = st.columns([3, 1.2, 1.2, 2])
    r_heur[0].checkbox("Heuristic", key="cb_show_heuristic")
    r_heur[1].write(_fmt_sec(st.session_state.get("runtime_heuristic")))
    r_heur[2].write(_fmt_int(st.session_state.get("deleted_heuristic")))
    r_heur[3].write("")

    partial_steps = st.session_state.get("partial_steps", [])
    runtimes_steps = st.session_state.get("runtime_steps", [])
    deleted_steps = st.session_state.get("deleted_steps", [])

    for i in range(1, len(partial_steps) + 1):
        key = f"cb_show_step_{i}"
        if key not in st.session_state:
            st.session_state[key] = (i == len(partial_steps))

        rt = runtimes_steps[i - 1] if i - 1 < len(runtimes_steps) else None
        td = deleted_steps[i - 1] if i - 1 < len(deleted_steps) else None

        r = st.columns([3, 1.2, 1.2, 2])
        r[0].checkbox(f"better repair (step {i})", key=key)
        r[1].write(_fmt_sec(rt))
        r[2].write(_fmt_int(td))
        r[3].write("")

    # Final render based on whatever is currently checked
    _render_chart(chart_slot, group_attr, agg_attr, agg_func)
