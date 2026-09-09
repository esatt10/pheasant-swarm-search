"""Reports.

Every report here answers the same question in a different amount of detail:
*what does this run's evidence support, and what does it not?* The health
vector at the top of ``summary.md`` is deliberately a vector rather than a
score - five different questions were asked, and collapsing them into one
number would hide exactly the disagreements between them that matter.
"""

from .query_detail import write_query_detail
from .refinements import RefinementCandidate, build_candidates, write_refinements
from .regressions import write_errors_report, write_regressions
from .summary import health_vector, write_arm_comparison, write_collection_report, write_summary

__all__ = [
    "RefinementCandidate",
    "build_candidates",
    "health_vector",
    "write_arm_comparison",
    "write_collection_report",
    "write_errors_report",
    "write_query_detail",
    "write_refinements",
    "write_regressions",
    "write_summary",
]
