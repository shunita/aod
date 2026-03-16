import time
import logging
import streamlit as st
import pandas as pd
import altair as alt
import json
import re

from trend_demo_api import TrendRepair

# Configure logging for silent exceptions
logging.basicConfig(level=logging.WARNING)
_logger = logging.getLogger(__name__)

explain_step = None
explain_steps_batch = None
DEFAULT_MODEL = None
try:
    from llm_explainer import explain_step, explain_steps_batch, DEFAULT_MODEL
except Exception:
    _logger.warning("llm_explainer failed to load")

hardcoded_explanations = None
try:
    import hardcoded_explanations
except Exception:
    _logger.warning("Hardcoded explanations are not available")

IS_SLEEP = True
USE_LIGHT_BG = False
HEURISTIC_COLOR = "#fca5a5"
FIRST_STEP_COLOR = (252, 165, 165)  # light red
OPTIMAL_COLOR = (134, 239, 172)  # light green
NARROW_BARS = False


# Explanation modes:
# 0 = LLM  
# 1 = Stats-only 
# 2 = Hard-coded
EXPLANATION_TYPE = 0

st.set_page_config(page_title="MonoTune: Analyze Trend Deviations", layout="wide")
st.markdown(
    "<h3 style='margin-bottom: 0.5rem;'>MonoTune: Analyze Trend Deviations</h3>",
    unsafe_allow_html=True,
)


# If you change EXPLANATION_TYPE in the code, reset cached explanations automatically.
if st.session_state.get("_explanation_type") != EXPLANATION_TYPE:
    st.session_state["_explanation_type"] = EXPLANATION_TYPE
    st.session_state["llm_explanations"] = {}
    st.session_state["distribution_summaries"] = {}
    st.session_state["pending_optimal_llm_batch"] = False

# global styling
# styling for light background
if USE_LIGHT_BG:
    st.markdown(
        """
        <style>
          /*  Page backgrounds  */
          div[data-testid="stAppViewContainer"],
          div[data-testid="stHeader"],
          section[data-testid="stSidebar"]{
            background: #ffffff !important;
          }

          /*  Readable text (avoid global div/span forcing)  */
          div[data-testid="stAppViewContainer"] :is(h1,h2,h3,h4,h5,h6,p,li,label){
            color: #111111 !important;
          }

          /*  Inputs / selects (closed control)  */
          div[data-baseweb="select"] > div,
          div[data-baseweb="input"] > div,
          div[data-baseweb="textarea"] > div,
          input, textarea{
            background-color: #ffffff !important;
            color: #111111 !important;
            border-color: rgba(0,0,0,0.20) !important;
          }

          /*  Select dropdown arrow (caret) in light mode  */
          div[data-baseweb="select"] svg,
          div[data-baseweb="select"] svg *{
            fill: #111111 !important;
            stroke: #111111 !important;
          }

          /*  BaseWeb portal layer (opened dropdown menus + popovers content)  */
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

          /*  Buttons  */
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


          /*  Expander (st.expander) should stay light even when open  */
          details[data-testid="stExpander"],
          div[data-testid="stExpander"] details{
            background-color: #ffffff !important;
            border: 1px solid rgba(0,0,0,0.15) !important;
            border-radius: 8px !important;
          }
          details[data-testid="stExpander"] > summary,
          div[data-testid="stExpander"] summary{
            background-color: #f3f4f6 !important;
            color: #111111 !important;
            border-radius: 8px !important;
          }
          details[data-testid="stExpander"][open] > summary,
          div[data-testid="stExpander"] details[open] > summary{
            background-color: #f3f4f6 !important;
            color: #111111 !important;
          }
          details[data-testid="stExpander"] > summary span,
          div[data-testid="stExpander"] summary span{
            color: #111111 !important;
          }
          details[data-testid="stExpander"] > summary svg,
          div[data-testid="stExpander"] summary svg{
            fill: #111111 !important;
            stroke: #111111 !important;
          }
          details[data-testid="stExpander"] > div,
          div[data-testid="stExpander"] details > div{
            background-color: #ffffff !important;
            color: #111111 !important;
          }

          /*  File uploader (light mode) */
          div[data-testid="stFileUploader"] section{
            background: #ffffff !important;
            border-color: rgba(0,0,0,0.20) !important;
          }
          div[data-testid="stFileUploader"] *{
            color: #111111 !important;
          }
          div[data-testid="stFileUploader"] svg,
          div[data-testid="stFileUploader"] svg *{
            fill: #111111 !important;
            stroke: #111111 !important;
          }
          div[data-testid="stFileUploader"] button{
            background: #f3f4f6 !important;
            color: #111111 !important;
            border: 1px solid rgba(0,0,0,0.20) !important;
          }

          div[data-testid="stFileUploader"] svg{
          background: transparent !important;
          }
          div[data-testid="stFileUploader"] svg rect{
          fill: none !important;
          }
          div[data-testid="stFileUploader"] svg path,
          div[data-testid="stFileUploader"] svg line,
          div[data-testid="stFileUploader"] svg polyline,
          div[data-testid="stFileUploader"] svg circle{
          stroke: #111111 !important;
          fill: none !important;
          }

          /*  st.dialog in light mode  */
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

        /* kill divider spacing but keep the line */
        div[data-testid="stDivider"]{
        margin-top: -1.6rem !important;
        margin-bottom: -1.1rem !important;
        padding: 0 !important;
        }
        div[data-testid="stDivider"] hr{
        margin: 0 !important;
        }

        /* Run header spacing */
        div[data-testid="stSubheader"]{
        margin-top: -1.8rem !important;
        margin-bottom: -0.4rem !important;
        padding: 0 !important;
        }
        div[data-testid="stSubheader"] h3{
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1.0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # dark style
else:
    st.markdown(
        """
        <style>
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

        /* kill divider spacing but keep the line */
        div[data-testid="stDivider"]{
        margin-top: -1.6rem !important;
        margin-bottom: -1.1rem !important;
        padding: 0 !important;
        }
        div[data-testid="stDivider"] hr{
        margin: 0 !important;
        }

        /* Run header spacing */
        div[data-testid="stSubheader"]{
        margin-top: -1.8rem !important;
        margin-bottom: -0.4rem !important;
        padding: 0 !important;
        }
        div[data-testid="stSubheader"] h3{
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1.0 !important;
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
    
def _hex_to_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def _rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)

def _lerp_hex(c0: str, c1: str, t: float) -> str:
    # t=0 -> c0, t=1 -> c1
    t = max(0.0, min(1.0, float(t)))
    r0, g0, b0 = _hex_to_rgb(c0)
    r1, g1, b1 = _hex_to_rgb(c1)
    r = int(round(r0 + (r1 - r0) * t))
    g = int(round(g0 + (g1 - g0) * t))
    b = int(round(b0 + (b1 - b0) * t))
    return _rgb_to_hex((r, g, b))

def _recolor_intermediate_steps_by_deleted():
    domain = st.session_state.get("SERIES_DOMAIN") or []
    rng = st.session_state.get("SERIES_RANGE") or []
    if not domain or len(domain) != len(rng):
        return

    max_steps = int(st.session_state.get("max_steps", 0))
    deleted_steps = st.session_state.get("deleted_steps") or []
    heur_deleted = st.session_state.get("deleted_heuristic", None)

    # change color after computation emnds
    if max_steps <= 1 or len(deleted_steps) < max_steps or heur_deleted is None:
        return

    try:
        heur_color = rng[domain.index("Heuristic")]
        opt_color  = rng[domain.index("Optimal")]
    except ValueError:
        return

    opt_deleted = deleted_steps[max_steps - 1]

    denom = (heur_deleted - opt_deleted)

    new_rng = list(rng)
    for i in range(1, max_steps): 
        step_label = f"Intermediate repair (step {i})"
        if step_label not in domain:
            continue

        d = deleted_steps[i - 1]

        if denom == 0:
            t = 0  #heuristic == optimal
        else:
            t = (d - opt_deleted) / denom

        new_rng[domain.index(step_label)] = _lerp_hex(opt_color, heur_color, t)

    st.session_state["SERIES_RANGE"] = new_rng


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

        # last step
        if i + 1 >= len(tr.group_keys):
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
    except Exception as e:
        _logger.warning("_compute_deleted_for_current_step failed: %s", e)
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
    """
    Colors for optimal/intermediate steps:
    - The last step ("Optimal") is light green.
    - Earlier steps gradually shift from light red -> light green.
    """
    start = FIRST_STEP_COLOR  
    end = OPTIMAL_COLOR 

    if n <= 1:
        return [_hex(end)]

    colors = []
    for i in range(n):
        t = i / max(n - 1, 1)
        rgb = (
            _lerp(start[0], end[0], t),
            _lerp(start[1], end[1], t),
            _lerp(start[2], end[2], t),
        )
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
        heur_deleted = st.session_state.get("deleted_heuristic", None)
        legend_label = f"Heuristic - {_fmt_int(heur_deleted)} tuples deleted"
        frames.append(_to_long(heur_df, group_attr, agg_attr, "Heuristic", legend_label))

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
            step_label = "Optimal" if (0 < max_steps == i) else f"Intermediate repair (step {i})"
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

    if group_order is None:
        group_order = list(dict.fromkeys(data["Group"].tolist())) 

    group_index = {g: idx for idx, g in enumerate(group_order)}
    data["GroupIndex"] = data["Group"].map(group_index).fillna(10**9).astype(int)

    step_cutoff = {}
    for s in data["SeriesKey"].unique():
        if s.startswith("Intermediate repair (step "):
            try:
                step_cutoff[s] = int(s.split("step ")[1].split(")")[0])
            except Exception as e:
                _logger.warning("Failed to parse step number from '%s': %s", s, e)
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

    # Per-bar tuple stats
    tuple_counts = st.session_state.get("tuple_counts_per_group") or {}
    if tuple_counts:
        # Build a (SeriesKey, Group) -> (deleted,left) lookup and merge it into the chart dataframe.
        pairs = data[["SeriesKey", "Group"]].drop_duplicates().copy()

        heur_deleted = st.session_state.get("heur_deleted_per_group") or {}
        heur_left = st.session_state.get("heur_left_per_group") or {}
        step_deleted_list = st.session_state.get("step_deleted_per_group") or []
        step_left_list = st.session_state.get("step_left_per_group") or []

        cache = {}

        def _get_maps(series_key: str):
            if series_key in cache:
                return cache[series_key]

            if series_key == "Original":
                del_map = {g: 0 for g in tuple_counts}
                left_map = tuple_counts
            elif series_key == "Heuristic":
                del_map = heur_deleted
                left_map = heur_left
            elif series_key == "Optimal":
                idx = len(step_left_list) - 1
                del_map = step_deleted_list[idx] if 0 <= idx < len(step_deleted_list) else {}
                left_map = step_left_list[idx] if 0 <= idx < len(step_left_list) else {}
            else:
                m = re.match(r"Intermediate repair \(step (\d+)\)", str(series_key))
                if m:
                    idx = int(m.group(1)) - 1
                    del_map = step_deleted_list[idx] if 0 <= idx < len(step_deleted_list) else {}
                    left_map = step_left_list[idx] if 0 <= idx < len(step_left_list) else {}
                else:
                    del_map, left_map = {}, {}

            cache[series_key] = (del_map, left_map)
            return cache[series_key]

        pairs["TuplesDeleted"] = pairs.apply(
            lambda r: int(_get_maps(r["SeriesKey"])[0].get(r["Group"], 0)), axis=1
        )
        pairs["TuplesLeft"] = pairs.apply(
            lambda r: int(_get_maps(r["SeriesKey"])[1].get(r["Group"], 0)), axis=1
        )

        data = data.merge(pairs, on=["SeriesKey", "Group"], how="left")
    else:
        data["TuplesDeleted"] = None
        data["TuplesLeft"] = None

    # Stable domain/range
    domain_all = st.session_state.get("SERIES_DOMAIN", [])
    range_all = st.session_state.get("SERIES_RANGE", [])

    present_series = list(data["SeriesKey"].unique())
    # Keep consistent order based on the full domain definition
    present_series = [s for s in domain_all if s in present_series] + [s for s in present_series if s not in domain_all]

    # If only one series is visible, reserve a second (empty) offset slot - narrow bar
    offset_domain = present_series
    if len(present_series) == 1 and NARROW_BARS:
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
                    labelLimit=0,  
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

            # make "heuristic" bars faint
            opacity=alt.condition("datum.IsStriped", alt.value(0.35), alt.value(1.0)),
            
            # tooltip - what you see when you hover
            tooltip=[
                # alt.Tooltip("SeriesLabel:N", title="Series"),
                # alt.Tooltip("Group:N", title="Group"),
                alt.Tooltip("Value:Q", title=f"{str(agg_func).upper()}({agg_attr})", format=".4g"),
                alt.Tooltip("TuplesDeleted:Q", title="Tuples deleted", format="d"),
                alt.Tooltip("TuplesLeft:Q", title="Tuples left", format="d"),
            ],
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
            except Exception as e:
                _logger.warning("Failed to convert values to float: v1=%s, v2=%s: %s", v1, v2, e)
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
    
    # change tooltip font size
    st.markdown(
        """
        <style>
        /* Vega/Altair hover tooltip */
        .vg-tooltip { 
            font-size: 15px !important;
            line-height: 1.25 !important;
        }
        .vg-tooltip table { 
            font-size: 15px !important;
        }
        /* "Title" row (Series / headers) */
        .vg-tooltip table thead th {
            font-size: 17px !important;
            font-weight: 700 !important;
        }
        .vg-tooltip table tbody td {
            font-size: 15px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


    chart_slot.altair_chart(chart, width='stretch')

# Shared layout for the results table under the chart
RESULTS_TABLE_COLS = [3, 1.2, 1.2]
DISPLAY_CELL_COLS = [0.18, 0.82]   # checkbox slot, then text slot
RESULTS_TABLE_GAP = "small"

def _render_table_header():
    cols = st.columns(RESULTS_TABLE_COLS, gap=RESULTS_TABLE_GAP)

    # Match the internal structure of the Display column in the body:
    # small checkbox slot + label text slot
    display_cols = cols[0].columns(DISPLAY_CELL_COLS, gap="small")
    display_cols[1].markdown("**Display**")

    cols[1].markdown("**Runtime**")
    cols[2].markdown("**Tuples deleted**")


def is_heuristic_label(label: str) -> bool:
    return "Heuristic" in label


def _get_distribution_summary(label: str):
    """Get or compute distribution summary for a repair label."""
    cache_key = "distribution_summaries"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = {}

    cache = st.session_state[cache_key]
    if label in cache:
        return cache[label]

    tr = st.session_state.get("tr_obj")
    if tr is None:
        return None

    try:
        if is_heuristic_label(label):
            summary = tr.summarize_distribution_differences("heuristic")
        elif label == "Optimal" or label.startswith("Intermediate repair (step "):
            # Extract step number
            if label == "Optimal":
                step_num = int(st.session_state.get("max_steps", 0))
            else:
                step_num = int(label.split("step ")[1].split(")")[0])
            summary = tr.summarize_distribution_differences(step_num)
        else:
            return None

        if summary:
            cache[label] = "\n".join(summary)
            return cache[label]
    except Exception as e:
        _logger.warning("_get_distribution_summary failed for '%s': %s", label, e)
    return None


def _render_distribution_expander(label: str):
    """Render an expandable panel showing distribution differences for a repair."""
    summary = _get_distribution_summary(label)
    if summary:
        with st.expander("Explanation: Distribution differences (removed vs. remaining)", expanded=False):
            st.markdown(summary)


def _render_row(label: str, key: str, runtime_s, deleted_n, runtime_total_s=None):
    row = st.container()
    cols = row.columns(RESULTS_TABLE_COLS, gap=RESULTS_TABLE_GAP)

    # Split the Display cell into:
    # [checkbox] [label text]
    display_cols = cols[0].columns(DISPLAY_CELL_COLS, gap="small")
    display_cols[0].checkbox("", key=key, label_visibility="collapsed")
    display_cols[1].markdown(label)

    rt_disp = (
        _fmt_sec(runtime_s)
        if runtime_total_s is None
        else f"{_fmt_sec(runtime_s)} ({_fmt_sec(runtime_total_s)})"
    )
    cols[1].write(rt_disp)
    cols[2].write(_fmt_int(deleted_n))

    # Distribution differences expander (only for Heuristic and Optimal/Intermediate steps)
    if label != "Original" and deleted_n is not None and deleted_n > 0:
        with row:
            _render_distribution_expander(label)

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
    llm_txt = (st.session_state.get("llm_explanations") or {}).get(label)
    if llm_txt:
        lines.append(llm_txt)
        lines.append("")
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
    elif is_heuristic_label(label):
        lines.append(
            "This is the greedy heuristic repair: it deletes tuples to reduce / eliminate trend violations quickly, without guaranteeing global optimality."
        )
        lines.append("The *Tuples deleted* count is the total number of removed tuples returned by the heuristic.")
    elif key.startswith("cb_show_step_"):
        # Optimal (DP) partial / final steps
        try:
            step_num = int(key.split("_")[-1])
        except Exception as e:
            _logger.warning("Failed to parse step number from key '%s': %s", key, e)
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
                except Exception as e:
                    _logger.warning("Failed to compute step boundary details: %s", e)

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


# -----------------------------
# LLM explanations (auto mode)
# -----------------------------

def _parse_step_num_from_label(label: str):
    if label == "Optimal":
        return int(st.session_state.get("max_steps", 0)) or None
    m = re.match(r"Intermediate repair \(step (\d+)\)", str(label))
    if m:
        try:
            return int(m.group(1))
        except Exception as e:
            _logger.warning("Failed to parse step number from label '%s': %s", label, e)
            return None
    return None


def _build_llm_payload(label: str, runtime_s=None):
    """
    Build a compact payload for the LLM based on per-group tuple deletion/remaining stats.
    Keeps input small for speed: includes totals + top groups by deletions.
    """
    params_key = st.session_state.get("params_key")
    group_attr = agg_attr = agg_func = None
    if isinstance(params_key, tuple) and len(params_key) == 4:
        _, group_attr, agg_attr, agg_func = params_key

    tr = st.session_state.get("tr_obj")
    group_order = []
    try:
        if tr is not None and getattr(tr, "group_keys", None) is not None:
            group_order = [str(x) for x in tr.group_keys]
    except Exception as e:
        _logger.warning("Failed to get group_order: %s", e)
        group_order = []

    original_counts = st.session_state.get("tuple_counts_per_group") or {}
    original_total = int(sum(original_counts.values())) if isinstance(original_counts, dict) else None

    deleted_map = {}
    left_map = {}
    attributes_with_high_diff = None

    if label == "Original":
        deleted_map = {str(k): 0 for k in original_counts.keys()}
        left_map = {str(k): int(v) for k, v in original_counts.items()}
    elif is_heuristic_label(label):
        deleted_map = st.session_state.get("heur_deleted_per_group") or {}
        left_map = st.session_state.get("heur_left_per_group") or {}
        attributes_with_high_diff = tr.summarize_distribution_differences("heuristic")
    else:
        step_num = _parse_step_num_from_label(label)
        if step_num is not None:
            deleted_steps = st.session_state.get("step_deleted_per_group") or []
            left_steps = st.session_state.get("step_left_per_group") or []
            if 1 <= step_num <= len(deleted_steps):
                deleted_map = deleted_steps[step_num - 1] or {}
            if 1 <= step_num <= len(left_steps):
                left_map = left_steps[step_num - 1] or {}
            attributes_with_high_diff = tr.summarize_distribution_differences(step_num)

    # normalize keys to str/int
    def _get_int(m, k, default=0):
        try:
            return int(m.get(k, default))
        except Exception:
            return int(default)

    # If no group order, fall back to keys from original counts
    if not group_order:
        group_order = [str(k) for k in original_counts.keys()]

    per_group = []
    for g in group_order:
        per_group.append({
            "group": g,
            "deleted_total": _get_int(deleted_map, g, 0),
            "remaining": _get_int(left_map, g, _get_int(original_counts, g, 0)),
        })

    tuples_remaining = int(sum([x["remaining"] for x in per_group])) if per_group else None
    tuples_deleted_total = None
    if original_total is not None and tuples_remaining is not None:
        tuples_deleted_total = int(original_total) - int(tuples_remaining)

    # Top groups by deletions (compact)
    top_groups = sorted(per_group, key=lambda r: r.get("deleted_total", 0), reverse=True)
    top_groups = top_groups[: min(8, len(top_groups))]

    payload = {
        "series_label": label,
        "step_num": _parse_step_num_from_label(label),
        "query": {
            "group_attr": group_attr,
            "agg_attr": agg_attr,
            "agg_func": agg_func,
            "trend_direction": st.session_state.get("trend_direction", "non-decreasing"),
        },
        "totals": {
            "tuples_original": original_total,
            "tuples_remaining": tuples_remaining,
            "tuples_deleted_total": tuples_deleted_total,
        },
        "top_groups_by_deleted": [
            {"group": r["group"], "deleted_total": r["deleted_total"], "remaining": r["remaining"]}
            for r in top_groups
        ],
        "runtime_s": runtime_s,
        "attributes_with_high_diff": "\n".join(attributes_with_high_diff) if attributes_with_high_diff is not None else "",
    }
    return payload



def _baseline_explanation_from_payload(payload: dict) -> str:
    """Fast, deterministic fallback explanation (also used for EXPLANATION_TYPE=1)."""
    if not isinstance(payload, dict):
        return ""
    label = payload.get("series_label") or payload.get("label") or ""
    q = payload.get("query") or {}
    totals = payload.get("totals") or {}
    top = payload.get("top_groups_by_deleted") or []
    # TODO possibly there should be an instance of attrs with high diff for each repair.
    attributes_with_high_diff = None
    attributes_with_high_diff = payload.get("attributes_with_high_diff")

    group_attr = q.get("group_attr")
    agg_attr = q.get("agg_attr")
    agg_func = q.get("agg_func")
    trend_dir = q.get("trend_direction")

    tuples_original = totals.get("tuples_original")
    tuples_remaining = totals.get("tuples_remaining")
    tuples_deleted_total = totals.get("tuples_deleted_total")

    lines = []
    if group_attr and agg_attr and agg_func:
        exp = "increase with" if trend_dir != "non-increasing" else "decrease with"
        lines.append(f"Trend: expect {str(agg_func).upper()}({agg_attr}) to {exp} {group_attr}.")
        lines.append("")

    if tuples_original is not None and tuples_remaining is not None:
        if tuples_deleted_total is None:
            try:
                tuples_deleted_total = int(tuples_original) - int(tuples_remaining)
            except Exception:
                tuples_deleted_total = None
    if tuples_deleted_total is not None and tuples_remaining is not None:
        lines.append(f"**Totals:** deleted **{_fmt_int(tuples_deleted_total)}** tuples; remaining **{_fmt_int(tuples_remaining)}**.")
    elif tuples_deleted_total is not None:
        lines.append(f"**Totals:** deleted **{_fmt_int(tuples_deleted_total)}** tuples.")
    lines.append("")

    # Top groups table (compact)
    if top:
        if group_attr:
            lines.append(f"**Most impacted {group_attr} groups:**")
        else:
            lines.append("**Most impacted groups:**")
        lines.append("")
        lines.append("| Group | Deleted | Remaining |")
        lines.append("|---|---:|---:|")
        for r in top:
            g = r.get("group", "—")
            d = r.get("deleted_total", 0)
            rem = r.get("remaining", 0)
            lines.append(f"| {g} | {int(d) if d is not None else 0} | {int(rem) if rem is not None else 0} |")
        lines.append("")

    # Series-specific note
    if label == "Original":
        lines.append("Baseline aggregate from the uploaded data (no deletions).")
    elif is_heuristic_label(label):
        lines.append("Greedy heuristic repair: deletes tuples to reduce/eliminate trend violations (not globally optimal).")
    elif isinstance(label, str) and (label == "Optimal" or label.startswith("Intermediate repair (step ")):
        lines.append("Optimal (DP) repair shown step-by-step (intermediate steps may combine DP prefix + heuristic suffix).")

    lines.append("Attributes with largest differences:")
    if attributes_with_high_diff is not None:
        lines.append(attributes_with_high_diff)

    return "\n".join([ln for ln in lines if ln is not None])


def _format_stats_explanation(payload: dict) -> str:
    """EXPLANATION_TYPE=1: show the exact statistics we would send to the LLM."""
    # For now, reuse the deterministic baseline, but also include the raw payload as JSON (compact).
    base = _baseline_explanation_from_payload(payload)
    try:
        payload_json = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        _logger.warning("Failed to serialize payload to JSON: %s", e)
        payload_json = None

    if payload_json:
        return base + "\n\n---\n\n**Stats payload (sent to the LLM in mode 0):**\n```json\n" + payload_json + "\n```\n"
    return base


def _hardcoded_explanation_for_label(label: str) -> str | None:
    """EXPLANATION_TYPE=2: pull explanations from hardcoded_explanations.py."""
    try:
        hc = hardcoded_explanations
        if hc is None:
            return None
        step_num = _parse_step_num_from_label(label)
        max_steps = int(st.session_state.get("max_steps", 0) or 0) or None
        if hasattr(hc, "get_explanation"):
            return hc.get_explanation(label=label, step_num=step_num, max_steps=max_steps)
        # Fallback if user only defines constants
        if label == "Original" and hasattr(hc, "ORIGINAL"):
            return getattr(hc, "ORIGINAL")
        if is_heuristic_label(label) and hasattr(hc, "HEURISTIC"):
            return getattr(hc, "HEURISTIC")
        if label == "Optimal" and hasattr(hc, "OPTIMAL"):
            return getattr(hc, "OPTIMAL")
        if step_num is not None and hasattr(hc, "STEP_EXPLANATIONS"):
            steps = getattr(hc, "STEP_EXPLANATIONS") or []
            if 1 <= step_num <= len(steps):
                return steps[step_num - 1]
        return None
    except Exception as e:
        _logger.warning("_hardcoded_explanation_for_label failed for '%s': %s", label, e)
        return None



def _auto_generate_single_explanation(label: str):
    """Generate and store explanation for a single label (Original/Heuristic) once."""
    if st.session_state.get("llm_explanations") is None:
        st.session_state["llm_explanations"] = {}

    if st.session_state["llm_explanations"].get(label):
        return

    runtime_key = {
        "Original": "runtime_original",
        "Heuristic": "runtime_heuristic",
    }.get(label)

    runtime_s = st.session_state.get(runtime_key) if runtime_key else None
    payload = _build_llm_payload(label, runtime_s=runtime_s)

    text = None
    if EXPLANATION_TYPE == 2:
        text = _hardcoded_explanation_for_label(label)
    elif EXPLANATION_TYPE == 1:
        text = _format_stats_explanation(payload)
    else:
        # EXPLANATION_TYPE == 0 (LLM)
        if explain_step is None:
            text = _baseline_explanation_from_payload(payload)
        else:
            try:
                text = explain_step(payload, model=st.session_state.get("llm_model", DEFAULT_MODEL))
            except Exception as e:
                _logger.warning("LLM explain_step failed for '%s': %s", label, e)
                text = _baseline_explanation_from_payload(payload)

    if text:
        st.session_state["llm_explanations"][label] = text



def _auto_generate_optimal_batch_explanations():
    """After ALL steps are done, generate all step explanations in one batch."""
    if st.session_state.get("llm_explanations") is None:
        st.session_state["llm_explanations"] = {}

    max_steps = int(st.session_state.get("max_steps", 0))
    if max_steps <= 0:
        return

    step_labels = [
        ("Optimal" if i == max_steps else f"Intermediate repair (step {i})")
        for i in range(1, max_steps + 1)
    ]

    # Only generate missing ones
    missing = [lab for lab in step_labels if not st.session_state["llm_explanations"].get(lab)]
    if not missing:
        return

    # Mode 2: hard-coded
    if EXPLANATION_TYPE == 2:
        for lab in missing:
            txt = _hardcoded_explanation_for_label(lab)
            if txt and (not st.session_state["llm_explanations"].get(lab)):
                st.session_state["llm_explanations"][lab] = txt
        return

    payloads = []
    for lab in missing:
        # runtime for steps is stored in runtime_steps list (aligned with partial_steps)
        step_num = _parse_step_num_from_label(lab)
        runtime_s = None
        try:
            if step_num is not None:
                rts = st.session_state.get("runtime_steps") or []
                if 1 <= step_num <= len(rts):
                    runtime_s = rts[step_num - 1]
        except Exception as e:
            _logger.warning("Failed to get runtime for step %s: %s", step_num, e)

        payload = _build_llm_payload(lab, runtime_s=runtime_s)

        # Mode 1: deterministic stats-only explanation
        if EXPLANATION_TYPE == 1:
            st.session_state["llm_explanations"][lab] = _format_stats_explanation(payload)
        else:
            payloads.append(payload)

    if EXPLANATION_TYPE == 1:
        return

    # Mode 0: LLM batch
    if explain_steps_batch is None:
        for p in payloads:
            lab = p.get("series_label")
            if lab and (not st.session_state["llm_explanations"].get(lab)):
                st.session_state["llm_explanations"][lab] = _baseline_explanation_from_payload(p)
        return

    try:
        out = explain_steps_batch(payloads, model=st.session_state.get("llm_model", DEFAULT_MODEL))
        if isinstance(out, dict):
            for lab, txt in out.items():
                if txt and (not st.session_state["llm_explanations"].get(lab)):
                    st.session_state["llm_explanations"][lab] = txt
    except Exception as e:
        _logger.warning("LLM explain_steps_batch failed: %s", e)
        # Fallback: deterministic explanations
        for p in payloads:
            lab = p.get("series_label")
            if lab and (not st.session_state["llm_explanations"].get(lab)):
                st.session_state["llm_explanations"][lab] = _baseline_explanation_from_payload(p)



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


# Example datasets available in data/ folder
EXAMPLE_DATASETS = {
    # "Stack Overflow (Binned Salary)": "data/SO/so_clean_for_trend_outliers_binned_salary.csv",
    "Salary by Education (Stack Overflow)": "data/SO/so_concise_for_edlevel_median_USA.csv",
    "Diabetes by Age": "data/diabetes/diabetes_prediction_dataset_binned_age.csv",
    "Loans by Employment Time (German Credit)": "data/german_credit/german_textual.csv",
}
UPLOAD_OPTION = "Upload your own..."

# Helper function to clean loaded dataframe
def _clean_dataframe(dataframe):
    # Drop unnamed index columns
    unnamed_cols = [c for c in dataframe.columns if c.startswith("Unnamed")]
    if unnamed_cols:
        dataframe = dataframe.drop(columns=unnamed_cols)

    # Clean up corrupted Unicode characters (replacement char sequences -> apostrophe)
    corrupted_base = chr(0xef) + chr(0xbf) + chr(0xbd)
    corrupted_with_trailing = corrupted_base + chr(0xef)

    # Only select object columns (candidates for containing text)
    for col in dataframe.select_dtypes(include=["object"]).columns:
        # CHECK: Skip columns that don't contain any actual strings (e.g. Boolean/NaN columns)
        if not dataframe[col].dropna().apply(lambda x: isinstance(x, str)).any():
            continue

        try:
            dataframe[col] = dataframe[col].str.replace(corrupted_base * 3, "'", regex=False)
            dataframe[col] = dataframe[col].str.replace(corrupted_with_trailing, "'", regex=False)
            dataframe[col] = dataframe[col].str.replace(corrupted_base, "'", regex=False)
        except AttributeError:
            # Fallback for any other columns where .str accessor fails
            continue

    return dataframe

# --- Dataset Loading (before layout) ---
# Use a separate session state key to track loaded dataset (not the widget key)
selected_dataset = st.session_state.get("loaded_dataset_name", None)
df = st.session_state.get("loaded_dataset_df", None)
data_key = st.session_state.get("loaded_dataset_key", None)

has_data = df is not None

# --- Welcome Screen (no dataset selected) ---
if not has_data:
    st.markdown("---")
    wcol1, wcol2, wcol3 = st.columns([1, 2, 1])
    with wcol2:
        st.markdown("### Select a Dataset")
        dataset_options = ["-- Select a dataset --"] + list(EXAMPLE_DATASETS.keys()) + [UPLOAD_OPTION]
        selected = st.selectbox(
            "Choose a dataset",
            dataset_options,
            key="welcome_dataset_select",
            label_visibility="collapsed",
        )

        if selected == UPLOAD_OPTION:
            uploaded = st.file_uploader(
                "Drag and drop a CSV file",
                type=["csv"],
                key="welcome_uploader",
            )
            if uploaded is not None:
                try:
                    loaded_df = pd.read_csv(uploaded)
                    loaded_df = _clean_dataframe(loaded_df)
                    st.session_state["loaded_dataset_name"] = uploaded.name
                    st.session_state["loaded_dataset_df"] = loaded_df
                    st.session_state["loaded_dataset_key"] = uploaded.name
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to load CSV: {e}")

        elif selected and selected != "-- Select a dataset --":
            # Load the example dataset
            example_path = EXAMPLE_DATASETS.get(selected)
            if example_path:
                try:
                    loaded_df = None
                    for encoding in ["utf-8", "latin-1", "cp1252"]:
                        try:
                            loaded_df = pd.read_csv(example_path, encoding=encoding)
                            break
                        except UnicodeDecodeError:
                            continue
                    if loaded_df is not None:
                        loaded_df = _clean_dataframe(loaded_df)
                        st.session_state["loaded_dataset_name"] = selected
                        st.session_state["loaded_dataset_df"] = loaded_df
                        st.session_state["loaded_dataset_key"] = example_path
                        st.rerun()
                    else:
                        st.error("Failed to load dataset: encoding error")
                except Exception as e:
                    st.error(f"Failed to load dataset: {e}")

        st.markdown("---")
        st.markdown(
            """
**How it works:**
1. Choose a **grouping attribute** and **aggregation function**
2. View the **original trend** with violations highlighted
3. Compare **heuristic** (fast) vs **optimal** (DP) repairs
4. See which tuples are removed to restore monotonicity
            """
        )
    st.stop()

# --- Main Layout (dataset selected) ---
controls_col, output_col = st.columns([1, 3], gap="large")

with controls_col:
    # "Change dataset" button
    st.divider()
    if st.button("Change dataset", width='stretch'):
        # Clear loaded dataset from session state
        st.session_state.pop("loaded_dataset_name", None)
        st.session_state.pop("loaded_dataset_df", None)
        st.session_state.pop("loaded_dataset_key", None)
        st.session_state.pop("params_key", None)
        st.session_state.pop("tr_obj", None)
        st.session_state.pop("distribution_summaries", None)
        st.rerun()

    st.subheader("Query")

    # Filter columns to exclude non-useful ones for grouping
    available_columns = [c for c in df.columns if not c.startswith("Unnamed")]
    # Only numeric columns can be aggregated
    numeric_columns = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    group_attr = st.selectbox("Grouping attribute", available_columns, key="group_attr_select")
    # Exclude the grouping column from aggregation options (same column causes pandas error)
    agg_columns = [c for c in numeric_columns if c != group_attr]
    if not agg_columns:
        st.warning("No numeric columns available for aggregation (excluding the grouping column).")
        st.stop()
    agg_attr = st.selectbox("Aggregation attribute", agg_columns, key="agg_attr_select")
    agg_func = st.selectbox("Aggregation function", ["sum", "avg", "median", "max"], key="agg_func_select")
    trend_direction = st.radio(
        "Trend direction",
        ["non-decreasing", "non-increasing"],
        index=0,
        horizontal=True,
        key="trend_direction_radio",
    )

    st.session_state["trend_direction"] = trend_direction

    params_key = (data_key, group_attr, agg_attr, agg_func)

    if st.session_state.get("params_key") != params_key:
        st.session_state["params_key"] = params_key
        st.session_state["tr_obj"] = TrendRepair(df, agg_func, group_attr, agg_attr)

        # Per-group tuple counts (for chart hover tooltips)
        _tuple_counts = df.groupby(group_attr, dropna=False).size()
        st.session_state["tuple_counts_per_group"] = {str(k): int(v) for k, v in _tuple_counts.to_dict().items()}

        # Per-group tuple stats for each series (filled as series are computed)
        st.session_state["heur_deleted_per_group"] = {}
        st.session_state["heur_left_per_group"] = {}
        st.session_state["step_deleted_per_group"] = []
        st.session_state["step_left_per_group"] = []

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

        num_groups = len(st.session_state["tr_obj"].group_keys)
        st.session_state["max_steps"] = max(num_groups, 0)

        max_steps = int(st.session_state.get("max_steps", 0))
        step_labels = [
            ("Optimal" if i == max_steps else f"Intermediate repair (step {i})")
            for i in range(1, max_steps + 1)
        ]
        step_colors = _make_step_palette(len(step_labels))
        st.session_state["SERIES_DOMAIN"] = ["Original", "Heuristic"] + step_labels
        st.session_state["SERIES_RANGE"] = ["#9ecae1", HEURISTIC_COLOR] + step_colors

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

        # LLM explanations (auto-generated)
        st.session_state["llm_explanations"] = {}
        st.session_state["distribution_summaries"] = {}
        st.session_state["pending_optimal_llm_batch"] = False
        st.session_state["llm_model"] = DEFAULT_MODEL

# Store references for Run buttons in output_col
tr = st.session_state.get("tr_obj")
max_steps = int(st.session_state.get("max_steps", 0))
current_steps = len(st.session_state.get("partial_steps", []))

with output_col:
    # Query description banner
    query_sql = f"Trend: expect {str(agg_func).upper()}({agg_attr}) to increase with {group_attr}"

    q_fg = "#111111" if USE_LIGHT_BG else "#ffffff"
    q_bg = "rgba(0,0,0,0.04)" if USE_LIGHT_BG else "rgba(255,255,255,0.06)"
    q_border = "rgba(0,0,0,0.12)" if USE_LIGHT_BG else "rgba(255,255,255,0.10)"

    st.markdown(
        f"""
        <div style="
            text-align:center;
            font-size:24px;
            font-weight:600;
            color:{q_fg};
            padding:10px 14px;
            border-radius:10px;
            background: {q_bg};
            border: 1px solid {q_border};
            margin-bottom: 10px;
        ">
            {query_sql}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Run Buttons (horizontal) ---
    run_col1, run_col2, run_col3 = st.columns(3)

    with run_col1:
        if st.button("Original", width='stretch'):
            with st.spinner("Running..."):
                t0 = time.perf_counter()
                st.session_state["original_df"] = tr.run_query()
                st.session_state["runtime_original"] = time.perf_counter() - t0
                st.session_state["deleted_original"] = 0
                _auto_generate_single_explanation("Original")

    # Unified progress slot (below buttons, above chart)
    progress_slot = st.empty()

    with run_col2:
        if st.button("Heuristic", width='stretch'):
            def heur_progress(iteration, smvi, removed):
                progress_slot.info(f"Heuristic: iteration {iteration}, removed {removed} tuples")

            if st.session_state.get("original_df") is None:
                progress_slot.info("Computing original...")
                t0 = time.perf_counter()
                st.session_state["original_df"] = tr.run_query()
                st.session_state["runtime_original"] = time.perf_counter() - t0
                st.session_state["deleted_original"] = 0

            progress_slot.info("Running heuristic...")
            t0 = time.perf_counter()
            heur_trend_result, _, heur_total_removed = tr.run_heuristic(progress_callback=heur_progress)
            progress_slot.empty()
            st.session_state["heur_df"] = heur_trend_result
            st.session_state["runtime_heuristic"] = time.perf_counter() - t0
            st.session_state["deleted_heuristic"] = int(heur_total_removed)

            if getattr(tr, "heur_deleted_per_group", None) is not None:
                st.session_state["heur_deleted_per_group"] = {
                    str(k): int(v) for k, v in tr.heur_deleted_per_group.to_dict().items()
                }
            if getattr(tr, "heur_left_per_group", None) is not None:
                st.session_state["heur_left_per_group"] = {
                    str(k): int(v) for k, v in tr.heur_left_per_group.to_dict().items()
                }
            st.session_state.get("llm_explanations", {}).pop("Heuristic", None)
            _auto_generate_single_explanation("Heuristic")

    with run_col3:
        step_done = (max_steps <= 0) or (current_steps >= max_steps)
        if st.button("Optimal", width='stretch', disabled=step_done):
            if st.session_state.get("heur_df") is None:
                def opt_heur_progress(iteration, smvi, removed):
                    progress_slot.info(f"Heuristic: iteration {iteration}, removed {removed} tuples")

                if st.session_state.get("original_df") is None:
                    progress_slot.info("Computing original...")
                    t0 = time.perf_counter()
                    st.session_state["original_df"] = tr.run_query()
                    st.session_state["runtime_original"] = time.perf_counter() - t0
                    st.session_state["deleted_original"] = 0

                progress_slot.info("Running heuristic (required)...")
                t0 = time.perf_counter()
                heur_trend_result, _, heur_total_removed = tr.run_heuristic(progress_callback=opt_heur_progress)
                progress_slot.empty()
                st.session_state["heur_df"] = heur_trend_result
                st.session_state["runtime_heuristic"] = time.perf_counter() - t0
                st.session_state["deleted_heuristic"] = int(heur_total_removed)
                _auto_generate_single_explanation("Heuristic")

                if getattr(tr, "heur_deleted_per_group", None) is not None:
                    st.session_state["heur_deleted_per_group"] = {
                        str(k): int(v) for k, v in tr.heur_deleted_per_group.to_dict().items()
                    }
                if getattr(tr, "heur_left_per_group", None) is not None:
                    st.session_state["heur_left_per_group"] = {
                        str(k): int(v) for k, v in tr.heur_left_per_group.to_dict().items()
                    }

            st.session_state["run_optimal_seq"] = True

    # Auto-generate LLM explanations for Optimal steps (batch) AFTER the smooth auto-run finishes.
    if (
        st.session_state.get("pending_optimal_llm_batch")
        and not st.session_state.get("auto_in_progress", False)
        and 0 < int(st.session_state.get("max_steps", 0) or 0) <= len(st.session_state.get("partial_steps", []))
    ):
        with st.spinner("Generating explanations for Optimal steps…"):
            _auto_generate_optimal_batch_explanations()
        st.session_state["pending_optimal_llm_batch"] = False

    # st.markdown("### Results")
    chart_slot = st.empty()

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
    # st.markdown("#### Display")
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

    # Main table shows ONLY the latest computed step (intermediate OR final optimal).
    latest_row_slot = st.empty()
    with st.expander("Show additional steps", expanded=False):
        additional_steps_slot = st.empty()

    partial_steps = st.session_state.get("partial_steps", [])
    runtimes_steps = st.session_state.get("runtime_steps", [])
    deleted_steps = st.session_state.get("deleted_steps", [])

    current_steps = len(partial_steps)

    latest_step_num = None
    latest_label = None
    if max_steps > 0 and current_steps >= max_steps:
        latest_step_num = max_steps
        latest_label = "Optimal"
    elif current_steps > 0:
        latest_step_num = current_steps
        latest_label = f"Intermediate repair (step {latest_step_num})"

    # Latest step row (replaces previous step each time)
    latest_row_slot.empty()
    if latest_step_num is not None:
        i = int(latest_step_num)
        key = f"cb_show_step_{i}"
        if key not in st.session_state:
            st.session_state[key] = True
        rt = runtimes_steps[i - 1] if i - 1 < len(runtimes_steps) else None
        td = deleted_steps[i - 1] if i - 1 < len(deleted_steps) else None
        rt_total = sum(runtimes_steps[:i]) if i <= len(runtimes_steps) else None
        with latest_row_slot.container():
            _render_row(latest_label, key, rt, td, runtime_total_s=rt_total)

    # Older steps - under the expander
    additional_steps_slot.empty()
    with additional_steps_slot.container():
        if latest_step_num is not None:
            for i in range(1, int(latest_step_num)):
                key = f"cb_show_step_{i}"
                if key not in st.session_state:
                    st.session_state[key] = False
                rt = runtimes_steps[i - 1] if i - 1 < len(runtimes_steps) else None
                td = deleted_steps[i - 1] if i - 1 < len(deleted_steps) else None
                rt_total = sum(runtimes_steps[:i]) if i <= len(runtimes_steps) else None
                _render_row(
                    f"Intermediate repair (step {i})",
                    key,
                    rt,
                    td,
                    runtime_total_s=rt_total,
                )

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

                progress_slot.info(f"Computing Optimal step {step_num}…")

                t0 = time.perf_counter()
                intermediate_result = tr.compute_next_partial_solution()
                dt = time.perf_counter() - t0

                st.session_state["partial_steps"].append(intermediate_result)
                st.session_state["runtime_steps"].append(dt)
                st.session_state["deleted_steps"].append(_compute_deleted_for_current_step(tr))

                # Per-group tuple stats for this step (for chart hover tooltips)
                if getattr(tr, "last_step_deleted_per_group", None) is not None:
                    st.session_state["step_deleted_per_group"].append(
                        {str(kk): int(vv) for kk, vv in tr.last_step_deleted_per_group.to_dict().items()}
                    )
                else:
                    st.session_state["step_deleted_per_group"].append({})
                if getattr(tr, "last_step_left_per_group", None) is not None:
                    st.session_state["step_left_per_group"].append(
                        {str(kk): int(vv) for kk, vv in tr.last_step_left_per_group.to_dict().items()}
                    )
                else:
                    st.session_state["step_left_per_group"].append({})

                # Tell chart to show only this newest step (smooth), without touching checkbox keys
                st.session_state["auto_visible_step"] = step_num

                # Update chart immediately
                _render_chart(chart_slot, group_attr, agg_attr, agg_func)

                # Update the display table immediately:
                # - main table shows ONLY the newest step
                # - older steps move into the expander
                row_label = "Optimal" if (max_steps > 0 and step_num >= max_steps) else f"Intermediate repair (step {step_num})"

                # Latest step row
                latest_row_slot.empty()
                with latest_row_slot.container():
                    _render_row(
                        row_label,
                        key=f"cb_show_step_{step_num}",
                        runtime_s=dt,
                        deleted_n=st.session_state["deleted_steps"][-1],
                        runtime_total_s=sum(st.session_state["runtime_steps"]),
                    )

                # Additional steps stay hidden during the smooth run (avoid duplicate checkbox keys).
                additional_steps_slot.empty()

                # Sleep
                if IS_SLEEP:
                    if k < steps_to_run - 1:
                        progress_slot.info(f"Step {step_num} ready. Next step in 3 seconds…")
                        time.sleep(0.5)

            # End auto mode
            st.session_state["auto_in_progress"] = False
            st.session_state["auto_visible_step"] = None
            st.session_state["run_optimal_seq"] = False
            progress_slot.empty()

            _recolor_intermediate_steps_by_deleted()

            # One rerun AFTER the whole smooth sequence:
            # makes checkbox states become "only newest checked" without flicker per step
            if last_step_num is not None:
                st.session_state["pending_optimal_llm_batch"] = True
                st.session_state["pending_step_checkbox_reset"] = True
                st.session_state["latest_step_for_reset"] = int(last_step_num)
                st.rerun()