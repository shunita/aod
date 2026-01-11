from __future__ import annotations
from typing import Optional


# Main series explanations
ORIGINAL = (
    "**Original (baseline):** This is the grouped aggregate computed directly from the uploaded data (no deletions).\n\n"
    "Red arrows/lines mark adjacent decreases in the *Original* bars to highlight trend violations."
)

HEURISTIC = (
    "**Heuristic repair:** A greedy deletion strategy that removes tuples to quickly reduce or eliminate trend violations.\n\n"
    "It is fast, but it does not guarantee a globally optimal (minimum-deletions) solution."
)

OPTIMAL = (
    "**Optimal repair (DP):** The final step is the globally optimal deletion plan over all groups (dynamic programming).\n\n"
    "Earlier intermediate steps may combine a DP-fixed prefix with a heuristic-fixed suffix to keep the demo interactive."
)


# Optimal step explanations (step 1..N). You can leave this list empty.
# Examples:
# STEP_EXPLANATIONS = [
#   "Step 1: DP fixes the first group(s), heuristic handles the rest.",
#   "Step 2: DP prefix expanded; fewer violations remain in later groups.",
#   "Step 3: DP over all groups (true optimum).",
# ]
STEP_EXPLANATIONS = [
  "Step 1: DP fixes the first group(s), heuristic handles the rest.",
  "Step 2: DP prefix expanded; fewer violations remain in later groups.",
  "Final step: DP over all groups (true optimum).",
]


def get_explanation(label: str, step_num: Optional[int], max_steps: Optional[int] = None) -> Optional[str]:
    """Return a hard-coded Markdown explanation for a given UI label.

    Parameters
    - label: one of "Original", "Heuristic", "Optimal", or "Intermediate repair (step i)"
    - step_num: parsed step number if label is a step label
    - max_steps: number of groups/steps in the current run (optional)

    Returns
    - Markdown string, or None if no hard-coded explanation is available.
    """
    if label == "Original":
        return ORIGINAL
    if label == "Heuristic":
        return HEURISTIC
    if label == "Optimal":
        # fall back to that when OPTIMAL is empty.
        if OPTIMAL:
            return OPTIMAL
        if step_num is not None and 1 <= step_num <= len(STEP_EXPLANATIONS):
            return STEP_EXPLANATIONS[step_num - 1]
        if max_steps is not None and 1 <= int(max_steps) <= len(STEP_EXPLANATIONS):
            return STEP_EXPLANATIONS[int(max_steps) - 1]
        return None

    # Intermediate steps
    if step_num is not None and 1 <= step_num <= len(STEP_EXPLANATIONS):
        return STEP_EXPLANATIONS[step_num - 1]

    return None
