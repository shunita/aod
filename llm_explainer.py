from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from openai import OpenAI

_logger = logging.getLogger(__name__)

DEFAULT_ENV_PATH = "connection.env"
DEFAULT_MODEL = "gpt-5-mini"

# Only cache successful generations: (model, digest) -> text
_SUCCESS_CACHE: Dict[Tuple[str, str], str] = {}


def _stable_digest(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(s).hexdigest()


def _resolve_env_path(env_path: str) -> Path:
    """
    Resolve env file location robustly:
    1) exact path as given (absolute)
    2) relative to current working directory
    3) relative to this file's directory
    """
    p = Path(env_path)
    if p.is_absolute() and p.exists():
        return p
    cwd_p = Path.cwd() / env_path
    if cwd_p.exists():
        return cwd_p
    here_p = Path(__file__).resolve().parent / env_path
    if here_p.exists():
        return here_p
    # fall back to cwd
    return cwd_p


def _get_client(env_path: str = DEFAULT_ENV_PATH) -> OpenAI:
    p = _resolve_env_path(env_path)
    load_dotenv(p)  # no-op if missing; validate below
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            f"OPENAI_API_KEY not found. Create '{p.name}' in your project folder "
            f"(same folder you run Streamlit from) and add:\n"
            f'OPENAI_API_KEY="YOUR_API_KEY"\n'
            f"Also add {p.name} to .gitignore."
        )
    return OpenAI(api_key=api_key)


def _system_prompt() -> str:
    # Keep it short and strict
    return (
        "You write short UI explanations for a trend-repair demo that removes tuples to fix monotonicity violations.\n\n"
        "Rules:\n"
        "- Use ONLY the provided statistics; do NOT invent facts.\n"
        "- If 'attributes_with_high_diff' is provided, briefly mention which attribute values are over/under-represented in removed tuples.\n"
        "- Keep numbers and group names exactly as provided.\n"
        "- Be concise (max 80 words).\n"
        "- Output plain text (no markdown, no bullets).\n"
    )


def _baseline_from_payload(p: Dict[str, Any]) -> str:
    series = p.get("series_label", "Result")
    q = p.get("query") or {}
    totals = p.get("totals") or {}
    trend = p.get("trend_status") or {}

    group_attr = q.get("group_attr", "?")
    agg_attr = q.get("agg_attr", "?")
    agg_func = str(q.get("agg_func", "")).upper()

    deleted_total = totals.get("tuples_deleted_total")
    deleted_step = totals.get("tuples_deleted_this_step")
    remaining = totals.get("tuples_remaining")
    dec_cnt = trend.get("adjacent_decreases_count")

    top = []
    for row in (p.get("top_groups_by_deleted") or [])[:3]:
        g = row.get("group")
        ds = row.get("deleted_this_step")
        if g is not None and ds is not None:
            top.append(f"{g} (+{ds})")
    top_str = ", ".join(top) if top else "no dominant group"

    parts = [f"{series}."]
    if deleted_step is not None and deleted_total is not None:
        parts.append(f"Deleted {deleted_step} additional tuples ({deleted_total} total).")
    if remaining is not None:
        parts.append(f"{remaining} tuples remain.")
    parts.append(f"Most deletions this step: {top_str}.")
    if dec_cnt is not None:
        parts.append(f"Adjacent decreases remaining in {agg_func}({agg_attr}) over {group_attr}: {dec_cnt}.")
    return " ".join(parts)


def explain_step(step_payload: Dict[str, Any], model: str = DEFAULT_MODEL, env_path: str = DEFAULT_ENV_PATH) -> str:
    """
    Generate a single explanation (fast). Returns baseline on failure.
    """
    d = _stable_digest(step_payload)
    cache_key = (model, d)
    if cache_key in _SUCCESS_CACHE:
        return _SUCCESS_CACHE[cache_key]

    baseline = _baseline_from_payload(step_payload)

    try:
        client = _get_client(env_path)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _system_prompt()},
                {
                    "role": "user",
                    "content": (
                        "Rewrite the baseline explanation into a clear, user-friendly sentence or two.\n\n"
                        f"Baseline:\n{baseline}\n\n"
                        "Stats JSON (ground truth):\n"
                        f"{json.dumps(step_payload, ensure_ascii=False)}\n"
                    ),
                },
            ],
            # IMPORTANT for GPT-5 / reasoning models:
            # - max_tokens is deprecated/not supported; use max_completion_tokens
            # - temperature may be unsupported for some GPT-5 family models; omit it
            max_completion_tokens=160,
        )
        text = (resp.choices[0].message.content or "").strip()
        out = text if text else baseline
        _SUCCESS_CACHE[cache_key] = out
        return out
    except Exception as e:
        _logger.warning("LLM explain_step failed: %s", e)
        return baseline


def explain_steps_batch(
    step_payloads: List[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
    env_path: str = DEFAULT_ENV_PATH,
) -> Dict[str, str]:
    """
    Generate explanations for multiple steps in ONE API call.
    Returns: series_label -> explanation text.
    Falls back to baselines per label if anything fails.
    """
    # Deduplicate by label
    by_label: Dict[str, Dict[str, Any]] = {}
    for p in step_payloads:
        label = str(p.get("series_label") or "")
        if not label:
            continue
        by_label[label] = p

    labels = list(by_label.keys())
    if not labels:
        return {}

    # Build compact items list (small tokens, fast response)
    items = []
    baselines: Dict[str, str] = {}
    for label in labels:
        p = by_label[label]
        base = _baseline_from_payload(p)
        baselines[label] = base
        items.append(
            {
                "series_label": label,
                "baseline": base,
                "totals": p.get("totals") or {},
                "top_groups_by_deleted": p.get("top_groups_by_deleted") or [],
                "query": p.get("query") or {},
                "trend_status": (p.get("trend_status") or {}),
            }
        )

    # If we already cached all, return from cache
    all_cached = True
    out_cached: Dict[str, str] = {}
    for label in labels:
        d = _stable_digest(by_label[label])
        k = (model, d)
        if k in _SUCCESS_CACHE:
            out_cached[label] = _SUCCESS_CACHE[k]
        else:
            all_cached = False
    if all_cached:
        return out_cached

    try:
        client = _get_client(env_path)
        prompt = (
            "For each item, write a short UI explanation (max 60 words) based ONLY on the provided stats.\n"
            "Keep numbers and group names exactly. Don't invent causes.\n"
            "Return ONLY a JSON object mapping series_label -> explanation string.\n\n"
            f"Items:\n{json.dumps(items, ensure_ascii=False)}\n"
        )

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _system_prompt()},
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=900,
        )
        raw = (resp.choices[0].message.content or "").strip()

        # Parse JSON (strict, then best-effort extraction)
        parsed: Any
        try:
            parsed = json.loads(raw)
        except Exception:
            # Extract first {...} block
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    parsed = json.loads(raw[start : end + 1])
                except Exception:
                    parsed = {}
            else:
                parsed = {}

        if not isinstance(parsed, dict):
            parsed = {}

        final: Dict[str, str] = {}
        for label in labels:
            txt = parsed.get(label)
            if isinstance(txt, str) and txt.strip():
                final[label] = txt.strip()
            else:
                final[label] = baselines.get(label, "")

            # cache success for each payload digest
            d = _stable_digest(by_label[label])
            _SUCCESS_CACHE[(model, d)] = final[label]

        return final
    except Exception as e:
        _logger.warning("LLM explain_steps_batch failed: %s", e)
        # Hard fallback: deterministic baselines
        return baselines
