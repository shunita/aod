import time
import streamlit as st
import pandas as pd
import altair as alt
import json

from trend_demo_api import TrendRepair

IS_SLEEP = True
USE_LIGHT_BG = False

st.set_page_config(page_title="MonoTune: Analyze Trend Deviations", layout="wide")
st.title("MonoTune: Analyze Trend Deviations")

# Subtle styling for the explanation modal (slightly transparent + blur)
# Global theme (background + text) + dialog styling
if USE_LIGHT_BG:
    st.markdown(
        """
        <style>
          /* ---------- Page backgrounds ---------- */
          div[data-testid="stAppViewContainer"],
          div[data-testid="stHeader"],
          section[data-testid="stSidebar"]{
            background: #ffffff !important;
          }

          /* ---------- Readable text (avoid global div/span forcing) ---------- */
          div[data-testid="stAppViewContainer"] :is(h1,h2,h3,h4,h5,h6,p,li,label){
            color: #111111 !important;
          }

          /* ---------- Inputs / selects (closed control) ---------- */
          div[data-baseweb="select"] > div,
          div[data-baseweb="input"] > div,
          div[data-baseweb="textarea"] > div,
          input, textarea{
            background-color: #ffffff !important;
            color: #111111 !important;
            border-color: rgba(0,0,0,0.20) !important;
          }

          /* ---------- BaseWeb portal layer (opened dropdown menus + popovers content) ---------- */
          div[data-baseweb="layer"]{
            color: #111111 !important;
          }
          div[data-baseweb="layer"] *{
            color: #111111 !important;
          }
          div[data-baseweb="layer"] :is(div,ul,li,section){
            background-color: #ffffff !important;
          }
          div[data-baseweb="layer"] [role="option"]:hover{
            background-color: rgba(0,0,0,0.06) !important;
          }

          /* ---------- Buttons (THIS fixes the Explain popover trigger) ---------- */
          /* Streamlit uses stBaseButton-* wrappers for most buttons & popover triggers */
          div[data-testid^="stBaseButton"] button,
          div[data-testid="stPopover"] button,
          button[aria-haspopup]{
            background-color: #f3f4f6 !important;
            color: #111111 !important;
            border: 1px solid rgba(0,0,0,0.20) !important;
          }
          div[data-testid^="stBaseButton"] button:hover,
          div[data-testid="stPopover"] button:hover,
          button[aria-haspopup]:hover{
            background-color: #e5e7eb !important;
          }

          /* ---------- Keep file uploader DARK (as you wanted) ---------- */
          div[data-testid="stFileUploader"] section{
            background: #111827 !important;
            border-color: rgba(255,255,255,0.22) !important;
          }
          div[data-testid="stFileUploader"] *{
            color: rgba(255,255,255,0.92) !important;
          }
          div[data-testid="stFileUploader"] button{
            background: #0f172a !important;
            color: #ffffff !important;
            border: 1px solid rgba(255,255,255,0.22) !important;
          }

          /* ---------- (Optional) st.dialog in light mode ---------- */
          div[data-testid='stDialog'] > div[role='dialog']{
            background: rgba(255, 255, 255, 0.90);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(0, 0, 0, 0.12);
            border-radius: 16px;
          }
          div[data-testid='stDialog'] :is(h2,h3,p,li){
            color: #111111 !important;
          }

          /* Buttons */
        div.stButton > button,
        div.stDownloadButton > button {
        background-color: #f3f4f6 !important;
        color: #111111 !important;
        border: 1px solid rgba(0,0,0,0.20) !important;
        }
        div.stButton > button:hover,
        div.stDownloadButton > button:hover {
        background-color: #e5e7eb !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <style>
          /* (Optional) keep your dark defaults, plus dialog styling */
          div[data-testid='stDialog'] > div[role='dialog'] {
            background: rgba(20, 20, 20, 0.82);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 16px;
          }
          div[data-testid='stDialog'] h2,
          div[data-testid='stDialog'] h3,
          div[data-testid='stDialog'] p,
          div[data-testid='stDialog'] li {
            color: #ffffff;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )



# Helpers
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
    Compute 'tuples deleted' for the *current* optimal step.
    Call immediately after compute_next_partial_solution().
    """
    try:
        if tr.removed_by_heur is None or tr.heur_trend_result is None or tr.inc_dp is None:
            return None

        step_num = int(tr.computed_dp_so_far)
        i = step_num - 1
        if i < 0:
            return None

        # last step => DP-only deletions (no i+1, no heuristic remainder)
        if i + 1 >= len(tr.group_keys):
            # Last step: DP-only (no next-group cutoff and no heuristic remainder)
            removed_tuples_up_to_i = tr.inc_dp.compute_up_to_i(i, None)
            if removed_tuples_up_to_i is None:
                return None
            return int(len(removed_tuples_up_to_i))

        heur_map = tr.heur_trend_result.set_index(tr.grouping_col).to_dict()[tr.aggregation_col]
        next_group_key = tr.group_keys[i + 1]
        cutoff = heur_map.get(next_group_key)

        removed_tuples_up_to_i = tr.inc_dp.compute_up_to_i(i, cutoff)

        remaining_group_keys = tr.group_keys[i + 1 :]
        removed_tuples_from_i_plus_1 = tr.removed_by_heur[
            tr.removed_by_heur[tr.grouping_col].isin(remaining_group_keys)
        ]

        all_removed_index = removed_tuples_up_to_i.index.append(removed_tuples_from_i_plus_1.index)
        return int(len(all_removed_index))
    except Exception:
        return None


def _to_long(
    trend_df: pd.DataFrame,
    group_attr: str,
    agg_attr: str,
    series_key: str,
    series_label: str = None,
) -> pd.DataFrame:
    out = trend_df[[group_attr, agg_attr]].copy()
    out.rename(columns={group_attr: "Group", agg_attr: "Value"}, inplace=True)
    out["Group"] = out["Group"].astype(str)
    out["SeriesKey"] = series_key
    out["SeriesLabel"] = series_label if series_label is not None else series_key
    return out


def _hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _lerp(a, b, t):
    return int(a + (b - a) * t)


def _make_step_palette(n: int):
    # Soft rainbow-ish but consistent
    start = (255, 159, 67)  # orange
    mid = (46, 204, 113)  # green
    end = (155, 89, 182)  # purple

    if n <= 1:
        return [_hex(start)]

    colors = []
    for i in range(n):
        t = i / max(n - 1, 1)
        if t < 0.5:
            tt = t / 0.5
            rgb = (_lerp(start[0], mid[0], tt), _lerp(start[1], mid[1], tt), _lerp(start[2], mid[2], tt))
        else:
            tt = (t - 0.5) / 0.5
            rgb = (_lerp(mid[0], end[0], tt), _lerp(mid[1], end[1], tt), _lerp(mid[2], end[2], tt))
        colors.append(_hex(rgb))
    return colors


def _render_chart(chart_slot, group_attr: str, agg_attr: str, agg_func: str):
    original_df = st.session_state.get("original_df")
    heur_df = st.session_state.get("heur_df")
    partial_steps = st.session_state.get("partial_steps", [])

    deleted_steps = st.session_state.get("deleted_steps", [])

    show_original = bool(st.session_state.get("cb_show_original", True))
    show_heuristic = bool(st.session_state.get("cb_show_heuristic", True))

    frames = []

    if show_original and original_df is not None:
        frames.append(_to_long(original_df, group_attr, agg_attr, "Original"))

    if show_heuristic and heur_df is not None:
        frames.append(_to_long(heur_df, group_attr, agg_attr, "Heuristic"))

    # Determine which steps to show
    auto_in_progress = bool(st.session_state.get("auto_in_progress", False))
    auto_visible_step = st.session_state.get("auto_visible_step", None)

    for i, step_df in enumerate(partial_steps, start=1):
        key = f"cb_show_step_{i}"
        if auto_in_progress and auto_visible_step is not None:
            show_step = (i == int(auto_visible_step))
        else:
            show_step = bool(st.session_state.get(key, i == len(partial_steps)))

        if show_step and step_df is not None:
            max_steps = int(st.session_state.get("max_steps", 0))
            step_label = "Optimal" if (max_steps > 0 and i == max_steps) else f"Intermediate repair (step {i})"
            deleted_n = deleted_steps[i - 1] if (i - 1) < len(deleted_steps) else None
            legend_label = f"{step_label} - {_fmt_int(deleted_n)} tuples deleted"
            frames.append(_to_long(step_df, group_attr, agg_attr, step_label, legend_label))

    if not frames:
        chart_slot.info("Run Original / Heuristic / Optimal to populate the chart.")
        return

    data = pd.concat(frames, ignore_index=True)

    # x-axis order 
    group_order = None
    base_df = None
    if original_df is not None:
        base_df = original_df
    elif heur_df is not None:
        base_df = heur_df
    elif partial_steps:
        base_df = partial_steps[0]

    # --- Stripe logic for Optimal/intermediate steps ---
    # Ensure we always have a deterministic order to map "first i groups" correctly
    if group_order is None:
        group_order = list(dict.fromkeys(data["Group"].tolist()))  # preserve appearance order

    group_index = {g: idx for idx, g in enumerate(group_order)}
    data["GroupIndex"] = data["Group"].map(group_index).fillna(10**9).astype(int)

    # For "Intermediate repair (step i)": groups [0..i-1] are DP => solid, groups [i..] are heuristic => striped
    # For "Optimal": user requested ALL striped
    step_cutoff = {}
    for s in data["SeriesKey"].unique():
        if s.startswith("Intermediate repair (step "):
            try:
                step_cutoff[s] = int(s.split("step ")[1].split(")")[0])
            except Exception:
                step_cutoff[s] = None
        elif s == "Optimal":
            step_cutoff[s] = None
        else:
            step_cutoff[s] = None

    data["StepCutoff"] = data["SeriesKey"].map(step_cutoff)

    data["IsStriped"] = False
    m = data["StepCutoff"].notna()
    data.loc[m, "IsStriped"] = (data.loc[m, "StepCutoff"] < 0) | (
        data.loc[m, "GroupIndex"] >= data.loc[m, "StepCutoff"]
    )


    if base_df is not None and group_attr in base_df.columns:
        group_order = [str(x) for x in base_df[group_attr].tolist()]


    # Stable domain/range
    domain_all = st.session_state.get("SERIES_DOMAIN", [])
    range_all = st.session_state.get("SERIES_RANGE", [])

    present_series = list(data["SeriesKey"].unique())
    # Keep consistent order based on the full domain definition
    present_series = [s for s in domain_all if s in present_series] + [s for s in present_series if s not in domain_all]

    # If only one series is visible, reserve a second (empty) offset slot
    # so the bar width matches the 2-series look.
    offset_domain = present_series
    if len(present_series) == 1:
        offset_domain = [present_series[0], "__offset_spacer__"]

    color_scale = alt.Scale(domain=domain_all, range=range_all)

    # Legend labels: show tuple-deletion counts for optimal steps, while keeping colors stable via SeriesKey
    label_map = (
        data.drop_duplicates("SeriesKey")[["SeriesKey", "SeriesLabel"]]
        .set_index("SeriesKey")["SeriesLabel"]
        .to_dict()
    )
    legend_label_expr = f"{json.dumps(label_map)}[datum.value] || datum.value"

    # Base bars
    bars = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X(
                "Group:N",
                sort=group_order,
                axis=alt.Axis(
                    labelAngle=0,
                    title=group_attr,
                    labelFontSize=18,
                    titleFontSize=20,
                ),
            ),
            y=alt.Y(
                "Value:Q",
                title=f"{agg_func}({agg_attr})",
                axis=alt.Axis(
                    labelFontSize=18,
                    titleFontSize=20,
                ),
            ),
            color=alt.Color(
                "SeriesKey:N",
                scale=color_scale,
                legend=alt.Legend(
                    title=None,
                    values=present_series,
                    labelExpr=legend_label_expr,
                    labelLimit=0,   # 0 = no ellipsis / no truncation
                    orient="top",
                    labelFontSize=18,
                    titleFontSize=20,
                ),
            ),
            xOffset=alt.XOffset(
                "SeriesKey:N",
                sort=offset_domain,
                scale=alt.Scale(domain=offset_domain),
            ),

            # NEW: make "striped" bars faint (the stripe overlay will sit on top)
            opacity=alt.condition("datum.IsStriped", alt.value(0.35), alt.value(1.0)),

            tooltip=["SeriesLabel:N", "Group:N", alt.Tooltip("Value:Q", format=".4g")],
        )
    )


    # Overlay red arrows/lines for Original adjacent decreases (only if Original is shown and present)
    arrow_layers = []
    if "Original" in present_series:
        # IMPORTANT: use the SAME x field ("Group") + SAME sort list as the bars,
        # otherwise Vega-Lite will re-derive the x-domain and you get lexicographic ordering.
        orig = data[data["SeriesKey"] == "Original"][["Group", "Value"]].copy()

        ordered_groups = group_order if group_order else orig["Group"].tolist()
        val_map = orig.set_index("Group")["Value"].to_dict()

        pts = []
        ends = []
        pair_id = 0

        for g1, g2 in zip(ordered_groups, ordered_groups[1:]):
            v1 = val_map.get(g1)
            v2 = val_map.get(g2)
            if v1 is None or v2 is None:
                continue
            try:
                v1f = float(v1)
                v2f = float(v2)
            except Exception:
                continue

            if v1f > v2f:
                pts.append({"Pair": pair_id, "Group": g1, "Value": v1f, "SeriesKey": "Original"})
                pts.append({"Pair": pair_id, "Group": g2, "Value": v2f, "SeriesKey": "Original"})
                ends.append({"Group": g2, "Value": v2f, "SeriesKey": "Original"})
                pair_id += 1

        if pts:
            dec_pts = pd.DataFrame(pts)
            dec_end = pd.DataFrame(ends)

            dec_line = (
                alt.Chart(dec_pts)
                .mark_line(color="red", strokeWidth=3)
                .encode(
                    x=alt.X("Group:N", sort=ordered_groups),
                    y=alt.Y("Value:Q"),
                    detail="Pair:N",
                    xOffset=alt.XOffset(
                        "SeriesKey:N",
                        sort=offset_domain,
                        scale=alt.Scale(domain=offset_domain),
                    ),
                )
            )

            dec_head = (
                alt.Chart(dec_end)
                .mark_point(shape="triangle-down", size=120, color="red")
                .encode(
                    x=alt.X("Group:N", sort=ordered_groups),
                    y=alt.Y("Value:Q"),
                    xOffset=alt.XOffset(
                        "SeriesKey:N",
                        sort=offset_domain,
                        scale=alt.Scale(domain=offset_domain),
                    ),
                )
            )

            arrow_layers = [dec_line, dec_head]


    chart = alt.layer(bars, *arrow_layers).properties(height=450)
    if USE_LIGHT_BG:
        chart = (
            chart.properties(background="#ffffff")
            .configure_view(fill="#ffffff", strokeOpacity=0)
            .configure_axis(
                labelColor="#111111",
                titleColor="#111111",
                gridColor="rgba(0,0,0,0.10)",
                domainColor="rgba(0,0,0,0.25)",
                tickColor="rgba(0,0,0,0.25)",
            )
            .configure_legend(labelColor="#111111", titleColor="#111111")
            .configure_title(color="#111111")
        )
    chart_slot.altair_chart(chart, use_container_width=True)


def _render_table_header():
    hcols = st.columns([3, 1.2, 1.2, 2])
    hcols[0].markdown("**Display**")
    hcols[1].markdown("**Runtime**")
    hcols[2].markdown("**Tuples deleted**")
    hcols[3].markdown("**Explanation**")


def _render_row(label: str, key: str, runtime_s, deleted_n, runtime_total_s=None):
    r = st.columns([3, 1.2, 1.2, 2])
    r[0].checkbox(label, key=key)

    rt_disp = _fmt_sec(runtime_s) if runtime_total_s is None else f"{_fmt_sec(runtime_s)} ({_fmt_sec(runtime_total_s)})"
    r[1].write(rt_disp)
    r[2].write(_fmt_int(deleted_n))

    # explanation
    with r[3]:
        if hasattr(st, "popover"):
            with st.popover("Explain", type="secondary", width="stretch"):
                # markdown-only (no widgets) so there is no rerun
                st.markdown(f"### {label}")
                st.markdown(f"**Runtime:** {rt_disp}")
                st.markdown(f"**Tuples deleted:** {_fmt_int(deleted_n)}")

                # temporary content
                params_key = st.session_state.get("params_key")
                if isinstance(params_key, tuple) and len(params_key) == 4:
                    _, group_attr, agg_attr, agg_func = params_key
                    # st.markdown(
                    #     f"**Query:** `SELECT {str(agg_func).upper()}({agg_attr}) GROUP BY {group_attr}`"
                    # )
                    st.markdown(
                        f"Trend: expect {str(agg_func).upper()}({agg_attr}) to increase with {group_attr}"
                    )
        else:
            # fallback if popover not available in this Streamlit version
            r[3].write("—")

# temporary
def _build_explanation_md(payload: dict) -> str:
    """Return markdown explanation text for the given row payload."""
    label = payload.get("label", "")
    key = payload.get("key", "")
    runtime_s = payload.get("runtime_s")
    deleted_n = payload.get("deleted_n")

    # Pull current params (best-effort)
    params_key = st.session_state.get("params_key")
    group_attr = agg_attr = agg_func = None
    if isinstance(params_key, tuple) and len(params_key) == 4:
        _, group_attr, agg_attr, agg_func = params_key

    tr = st.session_state.get("tr_obj")

    lines = []
    lines.append(f"### {label}")
    if runtime_s is not None:
        lines.append(f"**Runtime:** {_fmt_sec(runtime_s)}")
    if deleted_n is not None:
        lines.append(f"**Tuples deleted:** {_fmt_int(deleted_n)}")

    if group_attr and agg_attr and agg_func:
        lines.append("")
        # lines.append("**Query:**")
        lines.append(f"Trend: expect {str(agg_func).upper()}({agg_attr}) to increase with {group_attr}")
        # lines.append(f"`SELECT {str(agg_func).upper()}({agg_attr}) GROUP BY {group_attr}`")

    lines.append("")

    if label == "Original":
        lines.append("This is the baseline grouped aggregate computed directly from the uploaded data (no deletions).")
        lines.append(
            "Adjacent decreases in the *Original* series are highlighted with red arrows/lines to make trend violations easy to spot."
        )
    elif label == "Heuristic":
        lines.append(
            "This is the greedy heuristic repair: it deletes tuples to reduce / eliminate trend violations quickly, without guaranteeing global optimality."
        )
        lines.append("The *Tuples deleted* count is the total number of removed tuples returned by the heuristic.")
    elif key.startswith("cb_show_step_"):
        # Optimal (DP) partial / final steps
        try:
            step_num = int(key.split("_")[-1])
        except Exception:
            step_num = None

        if step_num is not None and tr is not None:
            n_groups = len(getattr(tr, "group_keys", []) or [])
            is_final = (n_groups > 0 and step_num >= n_groups)
            lines.append("This series comes from the *optimal* (dynamic programming) repair, shown step-by-step.")
            if is_final:
                lines.append("**Final step:** DP-only solution over *all* groups (no heuristic remainder).")
            else:
                lines.append(
                    "**Intermediate step:** DP fixes a prefix of the groups, then the heuristic is used for the remaining suffix (to keep the demo interactive)."
                )

            # Add the step boundary details when available
            if not is_final and getattr(tr, "heur_trend_result", None) is not None:
                try:
                    i = step_num - 1
                    next_group_key = tr.group_keys[i + 1]
                    heur_map = tr.heur_trend_result.set_index(tr.grouping_col).to_dict()[tr.aggregation_col]
                    cutoff = heur_map.get(next_group_key)
                    lines.append("")
                    lines.append(f"**Step boundary:** next group = `{next_group_key}`")
                    if cutoff is not None:
                        lines.append(f"**Heuristic cutoff (next group's aggregate):** `{cutoff}`")
                except Exception:
                    pass

            lines.append("")
            lines.append("**How to read `Tuples deleted` for steps:**")
            if is_final:
                lines.append("- DP removed tuples only (there is no 'next group', so there is no heuristic remainder to combine).")
            else:
                lines.append("- DP removed tuples for the prefix (up to this step).")
                lines.append("- PLUS heuristic-removed tuples for the remaining groups (step+1 … end).")
        else:
            lines.append("This is an optimal-repair step. Run the heuristic/optimal computation first to unlock a richer explanation.")
    else:
        lines.append("No explanation is available for this row yet.")

    return "\n".join(lines)


if hasattr(st, "dialog"):

    @st.dialog("Explanation")
    def _explanation_dialog():
        payload = st.session_state.get("explain_payload") or {}
        st.markdown(_build_explanation_md(payload))

else:

    # Fallback for older Streamlit versions
    def _explanation_dialog():
        payload = st.session_state.get("explain_payload") or {}
        st.info("Your Streamlit version does not support modal dialogs. Here's the explanation inline:")
        st.markdown(_build_explanation_md(payload))


def _maybe_show_explanation_dialog():
    if not st.session_state.get("explain_open", False):
        return
    _explanation_dialog()
    st.session_state["explain_open"] = False


# Layout: left controls, right output

controls_col, output_col = st.columns([1, 3], gap="large")

with controls_col:
    st.header("Controls")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    has_data = uploaded_file is not None
    if not has_data:
        st.info("Upload a CSV to begin.")
    else:
        df = pd.read_csv(uploaded_file)

        group_attr = st.selectbox("Grouping attribute", df.columns)
        agg_attr = st.selectbox("Aggregation attribute", df.columns)
        agg_func = st.selectbox("Aggregation function", ["sum", "avg", "median", "max"])
        trend_direction = st.radio(
        "Trend direction",
        ["non-decreasing", "non-increasing"],
        index=0,
        )

        params_key = (uploaded_file.name, group_attr, agg_attr, agg_func)

        if st.session_state.get("params_key") != params_key:
            st.session_state["params_key"] = params_key

            st.session_state["tr_obj"] = TrendRepair(df, agg_func, group_attr, agg_attr)

            st.session_state["original_df"] = None
            st.session_state["heur_df"] = None
            st.session_state["partial_steps"] = []

            st.session_state["runtime_original"] = None
            st.session_state["runtime_heuristic"] = None
            st.session_state["runtime_steps"] = []
            st.session_state["deleted_original"] = 0
            st.session_state["deleted_heuristic"] = None
            st.session_state["deleted_steps"] = []

            # clear old step checkboxes from previous query runs
            for k in list(st.session_state.keys()):
                if k.startswith("cb_show_step_"):
                    del st.session_state[k]

            # Safe max steps to avoid TrendRepair's next_group_key bug
            num_groups = len(st.session_state["tr_obj"].group_keys)
            st.session_state["max_steps"] = max(num_groups, 0)

            max_steps = int(st.session_state.get("max_steps", 0))
            step_labels = [
                ("Optimal" if i == max_steps else f"Intermediate repair (step {i})")
                for i in range(1, max_steps + 1)
            ]
            step_colors = _make_step_palette(len(step_labels))
            st.session_state["SERIES_DOMAIN"] = ["Original", "Heuristic"] + step_labels
            st.session_state["SERIES_RANGE"] = ["#9ecae1", "#3182bd"] + step_colors

            st.session_state["cb_show_original"] = True
            st.session_state["cb_show_heuristic"] = True

            st.session_state["run_optimal_seq"] = False
            st.session_state["auto_in_progress"] = False
            st.session_state["auto_visible_step"] = None
            st.session_state["pending_step_checkbox_reset"] = False
            st.session_state["latest_step_for_reset"] = None

            # explanation modal state
            st.session_state["explain_open"] = False
            st.session_state["explain_payload"] = None

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

        step_done = (max_steps <= 0) or (current_steps >= max_steps)
        if st.button("Optimal", use_container_width=True, disabled=step_done):
            # ensure heuristic exists (optimal depends on it)
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

            st.session_state["run_optimal_seq"] = True


with output_col:
    if uploaded_file is None:
        st.markdown("### Welcome 👋")
        st.markdown(
            """
This demo lets you **upload a dataset**, choose the following:
- a **group-by attribute**
- an **aggregation attribute**
- an **aggregation function**
- expected **trend direction**

Then compare:
- The trend based on the **Original data**
- **Heuristic** repair that maintains the trend
- **Optimal (DP)** repair shown step-by-step

Use the controls on the left to upload a dataset to begin.
            """
        )
        st.stop()
    else:
        # Centered query (bigger, pops, white)
        # query_sql = f"SELECT {agg_func.upper()}({agg_attr}) AS value GROUP BY {group_attr}"
        query_sql = f"Trend: expect {str(agg_func).upper()}({agg_attr}) to increase with {group_attr}"

        q_fg = "#111111" if USE_LIGHT_BG else "#ffffff"
        q_bg = "rgba(0,0,0,0.04)" if USE_LIGHT_BG else "rgba(255,255,255,0.06)"
        q_border = "rgba(0,0,0,0.12)" if USE_LIGHT_BG else "rgba(255,255,255,0.10)"

        st.markdown(
            f"""
            <div style="
                text-align:center;
                font-size:28px;
                font-weight:700;
                color:{q_fg};
                padding:14px 16px;
                border-radius:12px;
                background: {q_bg};
                border: 1px solid {q_border};
                margin-bottom: 14px;
            ">
                {query_sql}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Results")
        chart_slot = st.empty()
        status_slot = st.empty()

        if st.session_state.get("pending_step_checkbox_reset", False):
            latest = st.session_state.get("latest_step_for_reset")
            if latest is not None:
                for i in range(1, len(st.session_state.get("partial_steps", [])) + 1):
                    st.session_state[f"cb_show_step_{i}"] = (i == int(latest))
            st.session_state["pending_step_checkbox_reset"] = False
            st.session_state["latest_step_for_reset"] = None

        # Initial chart
        _render_chart(chart_slot, group_attr, agg_attr, agg_func)

        # Display table
        st.markdown("#### Display")
        _render_table_header()

        _render_row(
            "Original",
            "cb_show_original",
            st.session_state.get("runtime_original"),
            st.session_state.get("deleted_original", 0),
        )
        _render_row(
            "Heuristic",
            "cb_show_heuristic",
            st.session_state.get("runtime_heuristic"),
            st.session_state.get("deleted_heuristic"),
        )
        max_steps = int(st.session_state.get("max_steps", 0))
        optimal_row_slot = st.empty()

        partial_steps = st.session_state.get("partial_steps", [])
        runtimes_steps = st.session_state.get("runtime_steps", [])
        deleted_steps = st.session_state.get("deleted_steps", [])

        # Show final DP result as "Optimal" on the 3rd row (if available)
        if max_steps > 0 and len(partial_steps) >= max_steps:
            i = max_steps
            key = f"cb_show_step_{i}"
            if key not in st.session_state:
                st.session_state[key] = True
            rt = runtimes_steps[i - 1] if i - 1 < len(runtimes_steps) else None
            td = deleted_steps[i - 1] if i - 1 < len(deleted_steps) else None
            with optimal_row_slot:
                rt_total = sum(runtimes_steps[:i]) if i <= len(runtimes_steps) else None
                _render_row("Optimal", key, rt, td, runtime_total_s=rt_total)

        # Steps container (append rows during the smooth loop)
        steps_container = st.container()

        with steps_container:
            partial_steps = st.session_state.get("partial_steps", [])
            runtimes_steps = st.session_state.get("runtime_steps", [])
            deleted_steps = st.session_state.get("deleted_steps", [])

            intermediate_upto = min(len(partial_steps), max(0, max_steps - 1))
            for i in range(1, intermediate_upto + 1):
                key = f"cb_show_step_{i}"
                if key not in st.session_state:
                    st.session_state[key] = (i == len(partial_steps))
                rt = runtimes_steps[i - 1] if i - 1 < len(runtimes_steps) else None
                td = deleted_steps[i - 1] if i - 1 < len(deleted_steps) else None
                rt_total = sum(runtimes_steps[:i]) if i <= len(runtimes_steps) else None
                _render_row(f"Intermediate repair (step {i})", key, rt, td, runtime_total_s=rt_total)

        _maybe_show_explanation_dialog()

        # smooth Optimal: compute -> show -> sleep -> next (no reruns during the sequence) 
        if bool(st.session_state.get("run_optimal_seq", False)):
            tr = st.session_state["tr_obj"]
            max_steps = int(st.session_state.get("max_steps", 0))

            steps_to_run = max_steps - len(st.session_state.get("partial_steps", []))
            if steps_to_run <= 0:
                st.session_state["run_optimal_seq"] = False
            else:
                st.session_state["auto_in_progress"] = True
                last_step_num = None

                for k in range(steps_to_run):
                    step_num = len(st.session_state["partial_steps"]) + 1
                    last_step_num = step_num

                    status_slot.info(f"Computing Optimal step {step_num}…")

                    t0 = time.perf_counter()
                    intermediate_result = tr.compute_next_partial_solution()
                    dt = time.perf_counter() - t0

                    st.session_state["partial_steps"].append(intermediate_result)
                    st.session_state["runtime_steps"].append(dt)
                    st.session_state["deleted_steps"].append(_compute_deleted_for_current_step(tr))

                    # Tell chart to show only this newest step (smooth), without touching checkbox keys
                    st.session_state["auto_visible_step"] = step_num

                    # Update chart immediately
                    _render_chart(chart_slot, group_attr, agg_attr, agg_func)

                    # Append the new row immediately to the table
                    row_label = "Optimal" if (max_steps > 0 and step_num >= max_steps) else f"Intermediate repair (step {step_num})"
                    target_container = optimal_row_slot if row_label == "Optimal" else steps_container

                    with target_container:
                        _render_row(
                            row_label,
                            key=f"cb_show_step_{step_num}",
                            runtime_s=dt,
                            deleted_n=st.session_state["deleted_steps"][-1],
                            runtime_total_s=sum(st.session_state["runtime_steps"]),
                        )

                    # Sleep between steps (optional)
                    if IS_SLEEP:
                        if k < steps_to_run - 1:
                            status_slot.info(f"Step {step_num} ready. Next step in 3 seconds…")
                            time.sleep(3)

                # End auto mode
                st.session_state["auto_in_progress"] = False
                st.session_state["auto_visible_step"] = None
                st.session_state["run_optimal_seq"] = False
                status_slot.empty()

                # One rerun AFTER the whole smooth sequence:
                # makes checkbox states become "only newest checked" without flicker per step
                if last_step_num is not None:
                    st.session_state["pending_step_checkbox_reset"] = True
                    st.session_state["latest_step_for_reset"] = int(last_step_num)
                    st.rerun()