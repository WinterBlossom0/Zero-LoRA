"""Turn evaluated deletions into the paper's metrics: target damage, collateral damage, and damage at fixed collateral.

Each evaluated deletion gives NLL changes (deleted model minus base model) per evaluation set. With FLORES codes:
  target damage     mean change over the target codes (e.g. zho_Hans, zho_Hant);
  collateral damage mean change over every other FLORES code that is evaluated, except the codes listed as related
                    to the target (e.g. yue_Hant, Cantonese in Traditional script, for a Chinese target), which are
                    reported but count on neither side.
Primary metric: damage at collateral c = the largest target damage among a method's deletions whose collateral
damage is at most c (the best operating point the method offers at that side-effect budget); default c = 0.05 nats.
"""

import itertools
import math

COLLATERAL_LEVEL = 0.05  # nats of mean NLL rise on non-target languages accepted as "selective"


def damage_and_collateral(delta: dict[str, float], target_codes: list[str], related_codes: list[str]
                          ) -> tuple[float, float]:
    """Return (target damage, collateral damage) from per-set NLL changes.

    A target code is a FLORES code (key "flores_<code>") or a full evaluation key such as "wiki_corruption" (a
    planted target, scored on its own test rows). Collateral: the FLORES codes that are neither targets nor related.
    """
    target_keys = [code if code in delta else f"flores_{code}" for code in target_codes]
    missing = [key for key in target_keys if key not in delta]
    assert not missing, f"target codes without an evaluation: {missing}"
    excluded = set(target_keys) | {f"flores_{code}" for code in related_codes}
    others = [value for key, value in delta.items() if key.startswith("flores_") and key not in excluded]
    assert others, "no collateral languages evaluated"
    target = sum(delta[key] for key in target_keys) / len(target_keys)
    return target, sum(others) / len(others)


def damage_at_collateral(points: list[tuple[float, float]], level: float = COLLATERAL_LEVEL) -> float:
    """Return the largest target damage among (target damage, collateral damage) points with collateral <= level;
    0 if none qualifies (no deletion, i.e. the base model, qualifies with damage 0)."""
    qualifying = [damage for damage, collateral in points if collateral <= level]
    return max(qualifying, default=0.0)


def frontier_area(points: list[tuple[float, float]], max_collateral: float) -> float:
    """Return the area under the best-damage-so-far curve over collateral in [0, max_collateral] (nats^2).

    The curve is a step function: at collateral c it is damage_at_collateral(points, c); a summary over all
    side-effect budgets, like the faithfulness areas of MIB (Mueller et al. 2025).
    """
    steps = sorted({0.0, max_collateral} | {collateral for _, collateral in points
                                            if 0.0 <= collateral <= max_collateral})
    area = 0.0
    for left, right in itertools.pairwise(steps):
        area += damage_at_collateral(points, left) * (right - left)
    assert math.isfinite(area)
    return area
