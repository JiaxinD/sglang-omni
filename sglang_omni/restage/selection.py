"""Select among candidates measured with one workload, SLO and load grid."""

from collections.abc import Mapping
from dataclasses import dataclass

from sglang_omni.restage.search import RateSearchResult


@dataclass(frozen=True)
class CandidateRank:
    key: str
    best_tested_rate: float | None
    passing_prefix_rate: float | None
    median_goodput_qps: float | None
    upper_limit_passed: bool
    nonmonotonic: bool
    reason: str


@dataclass(frozen=True)
class Selection:
    recommended: str | None
    ranking: tuple[CandidateRank, ...]
    baseline: str
    baseline_rate: float | None
    rate_gain_over_baseline: float | None
    scope: str = "best among measured candidates at tested rates; not global optimality"


def select_candidate(
    results: Mapping[str, RateSearchResult], *, baseline: str
) -> Selection:
    """Prefer the highest passing prefix of the tested grid, then median goodput.

    The caller must hold model, workload, SLO and hardware budget constant.
    Equal scores retain the baseline; other exact ties use the candidate key.
    No confidence interval or untested capacity is inferred from point estimates.
    """
    if baseline not in results:
        raise ValueError("The measured baseline must be included")
    grid = [(p.rate, len(p.evaluations)) for p in results[baseline].points]
    if not grid or any(
        [(p.rate, len(p.evaluations)) for p in result.points] != grid
        for result in results.values()
    ):
        raise ValueError("Candidates must use the same rates and repeat counts")
    ranking = []
    for key, result in results.items():
        passing = [point for point in result.points if point.feasible]
        highest = max(passing, key=lambda point: point.rate) if passing else None
        best = None
        for point in result.points:
            if not point.feasible:
                break
            best = point
        reason = (
            f"All {len(best.evaluations)} repeats passed at every tested rate "
            f"up to {best.rate:g} requests/s"
            if best is not None
            else "No recommendation: no tested rate in a passing prefix"
        )
        if result.upper_limit_passed:
            reason += "; upper limit passed, capacity boundary remains unmeasured"
        if result.nonmonotonic:
            reason += "; passing above a failed lower rate, requires confirmation"
        ranking.append(
            CandidateRank(
                key=key,
                best_tested_rate=highest.rate if highest else None,
                passing_prefix_rate=best.rate if best else None,
                median_goodput_qps=best.median_goodput_qps if best else None,
                upper_limit_passed=result.upper_limit_passed,
                nonmonotonic=result.nonmonotonic,
                reason=reason,
            )
        )
    ranking.sort(
        key=lambda row: (
            -(row.passing_prefix_rate or 0),
            -(row.median_goodput_qps or 0),
            row.key != baseline,
            row.key,
        )
    )
    winner = ranking[0] if ranking[0].passing_prefix_rate is not None else None
    baseline_rate = next(
        row.passing_prefix_rate for row in ranking if row.key == baseline
    )
    return Selection(
        recommended=winner.key if winner else None,
        ranking=tuple(ranking),
        baseline=baseline,
        baseline_rate=baseline_rate,
        rate_gain_over_baseline=(
            winner.passing_prefix_rate / baseline_rate
            if winner is not None and baseline_rate is not None
            else None
        ),
    )
