"""팀원 C 파트 — 이상탐지(SPC 관리도) · 성능평가 · XAI 연계 모듈.

반도체 PE/CS 직무의 SPC(통계적공정관리) / FDC(Fault Detection & Classification)
구조를 그대로 따른다.

  control_charts.py : Shewhart 3σ, EWMA, CUSUM, Isolation Forest
  evaluation.py     : 합성 이상 주입 기반 오탐/미탐 트레이드오프 정량화
"""

from detect.control_charts import (
    shewhart_3sigma,
    ewma_chart,
    cusum_chart,
    isolation_forest_flags,
    run_all_detectors,
    residualize,
    weekday_residual,
    NOMINAL_ARL0,
)
from detect.evaluation import (
    inject_faults,
    confusion_counts,
    detector_metrics,
    threshold_sweep,
    average_run_length,
    benchmark_detectors,
)

__all__ = [
    "shewhart_3sigma", "ewma_chart", "cusum_chart",
    "isolation_forest_flags", "run_all_detectors",
    "residualize", "weekday_residual", "NOMINAL_ARL0",
    "inject_faults", "confusion_counts", "detector_metrics",
    "threshold_sweep", "average_run_length", "benchmark_detectors",
]
