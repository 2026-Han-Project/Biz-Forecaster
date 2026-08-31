"""이상탐지 성능 분석 도구 (개발자용 · 팀원 C 파트).

앱 화면에서는 뺀 기능이다. 소상공인에게는 3시그마·CUSUM·오탐률 같은 말이
아무 도움이 안 되지만, 관리도를 설계하고 문턱값을 정하려면 반드시 필요한 수치다.
그래서 화면 대신 이 스크립트로 옮겼다.

  사용법
    .venv\\Scripts\\python.exe analyze_detectors.py
    .venv\\Scripts\\python.exe analyze_detectors.py --csv data/integrated_dataset.csv
    .venv\\Scripts\\python.exe analyze_detectors.py --item 크루아상 --magnitude 2.0
    .venv\\Scripts\\python.exe analyze_detectors.py --save        결과를 CSV와 monitoring.db 로 저장

  하는 일
    1. 관리도별 탐지 성능 비교 (spike / step / drift 세 시나리오)
    2. 문턱값 스윕 — 오탐률과 미탐률이 어떻게 맞바뀌는지
    3. 요일 계절성 제거(잔차) 전후 비교
    4. --save 를 주면 monitoring.db 의 detector_performance 테이블에 기록

정답 라벨이 없는 문제를 어떻게 채점하는지는 detect/evaluation.py 의 설명을 참고.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from detect.control_charts import run_all_detectors            # noqa: E402
from detect.evaluation import (                                # noqa: E402
    DETECTOR_KNOBS, FAULT_TYPES, benchmark_detectors, threshold_sweep,
)
from detect.monitoring import (                                # noqa: E402
    MONITORING_DB_PATH, export_detector_performance,
)

DEFAULT_CSV = Path(__file__).resolve().parent / "data" / "integrated_dataset.csv"


def load_series(csv_path, item=None):
    """CSV에서 일별 판매량 계열을 만든다. item을 주면 그 품목만, 없으면 전체 매출."""
    df = pd.read_csv(csv_path, parse_dates=["date"])
    missing = {"date", "item", "sales_qty"} - set(df.columns)
    if missing:
        raise SystemExit(f"필수 컬럼이 없습니다: {', '.join(sorted(missing))}")

    total = df.groupby("date")["sales_qty"].sum().asfreq("D").fillna(0.0)
    if item is None:
        return total, "(전체 매출)"

    if item not in set(df["item"]):
        raise SystemExit(
            f"'{item}' 품목이 없습니다.\n사용 가능: {', '.join(sorted(df['item'].unique()))}"
        )
    s = df[df["item"] == item].groupby("date")["sales_qty"].sum()
    return s.reindex(total.index, fill_value=0.0), item


def section(title):
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser(description="이상탐지 성능 분석 (팀원 C 파트)")
    ap.add_argument("--csv", default=str(DEFAULT_CSV), help="분석할 CSV 경로")
    ap.add_argument("--item", default=None, help="품목명 (생략하면 전체 매출)")
    ap.add_argument("--magnitude", type=float, default=2.5, help="주입할 이상 크기 (표준편차 배수)")
    ap.add_argument("--n-faults", type=int, default=5, help="주입할 이상 개수")
    ap.add_argument("--duration", type=int, default=10, help="이상 지속일 (spike는 무시)")
    ap.add_argument("--tolerance", type=int, default=1, help="탐지 허용 지연(일)")
    ap.add_argument("--save", action="store_true", help="결과를 CSV와 monitoring.db 로 저장")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"파일을 찾을 수 없습니다: {csv_path}")

    series, label = load_series(csv_path, args.item)
    print(f"\n대상   : {label}")
    print(f"기간   : {series.index.min().date()} ~ {series.index.max().date()} ({len(series)}일)")
    print(f"평균   : {series.mean():,.1f}개/일   표준편차 {series.std():,.1f}")

    if len(series) < 80:
        raise SystemExit("성능 분석에는 최소 80일치 데이터가 필요합니다.")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    # ---------- 1. 잔차화 전후 ----------
    section("1. 요일 계절성 제거(잔차) 전후 — 탐지된 이상일 수")
    rows = []
    for mode in ("none", "weekday"):
        _, combined = run_all_detectors(series, residual_mode=mode)
        rows.append({
            "관리도 대상": "원계열" if mode == "none" else "요일 잔차",
            "3시그마": int(combined["3시그마"].sum()),
            "EWMA": int(combined["EWMA"].sum()),
            "CUSUM": int(combined["CUSUM"].sum()),
            "IsolationForest": int(combined["IsolationForest"].sum()),
            "합의(2개+)": int(combined["이상여부"].sum()),
        })
    resid_df = pd.DataFrame(rows)
    print(resid_df.to_string(index=False))
    print("\n  관리도는 '정상일 때 값이 하나의 분포를 따른다'를 전제한다.")
    print("  요일 주기가 남아 있으면 표준편차가 부풀어 관리한계가 넓어지고, 탐지력이 떨어진다.")

    # ---------- 2. 탐지기 비교 ----------
    perf_all = []
    for ft in FAULT_TYPES:
        section(f"2-{FAULT_TYPES.index(ft) + 1}. 탐지기 성능 — {ft} 이상 "
                f"{args.n_faults}건 ({args.magnitude}σ) 주입")
        perf, _, labels = benchmark_detectors(
            series, fault_type=ft, n_faults=args.n_faults,
            magnitude=args.magnitude, duration=args.duration, tolerance=args.tolerance,
        )
        cols = ["탐지기", "오탐(FP)", "미탐(FN)", "오탐률(FAR)", "미탐률(MDR)",
                "F1", "이벤트탐지율", "평균탐지지연(일)", "실측ARL0"]
        print(f"  정답 이상일 {int(labels.sum())}일 / 전체 {len(labels)}일\n")
        print(perf[cols].to_string(index=False))
        best = perf.loc[perf["F1"].idxmax(), "탐지기"]
        print(f"\n  → F1 최고: {best}")
        # export_detector_performance() 가 fault_type 열을 직접 넣으므로
        # 여기서는 붙이지 않고 유형을 따로 들고 간다.
        perf_all.append((ft, perf))

    print("\n  이상 유형마다 승자가 바뀐다. 모든 유형에서 이기는 관리도는 없고,")
    print("  그래서 여러 개를 함께 돌린 뒤 다수결로 합의시킨다.")

    # ---------- 3. 문턱값 스윕 ----------
    section("3. 문턱값 스윕 — 오탐과 미탐은 반대로 움직인다")
    _, faulty, labels = benchmark_detectors(
        series, fault_type="spike", n_faults=args.n_faults,
        magnitude=args.magnitude, duration=args.duration, tolerance=args.tolerance,
    )
    sweeps = []
    for det in DETECTOR_KNOBS:
        sw = threshold_sweep(faulty, labels, detector=det, tolerance=args.tolerance)
        print(f"\n[{det}]")
        print(sw.to_string(index=False))
        sweeps.append(sw)

    print("\n  요점은 '최적값 하나'가 아니라, 어떤 값을 골라도 둘 다 좋아지지는 않는다는 사실이다.")
    print("  어디를 고를지는 통계가 아니라 비용이 정한다 —")
    print("  헛경보 한 번의 비용과 놓친 이상 하나의 비용 중 무엇이 비싼가.")

    # ---------- 4. 저장 ----------
    if args.save:
        section("4. 결과 저장")
        out_dir = Path(__file__).resolve().parent / "analysis_output"
        out_dir.mkdir(exist_ok=True)

        perf_df = pd.concat(
            [d.assign(fault_type=ft) for ft, d in perf_all], ignore_index=True
        )
        sweep_df = pd.concat(sweeps, ignore_index=True)
        safe = label.replace("(", "").replace(")", "").replace(" ", "_")

        for name, d in (("performance", perf_df), ("sweep", sweep_df), ("residual", resid_df)):
            path = out_dir / f"{safe}_{name}.csv"
            d.to_csv(path, index=False, encoding="utf-8-sig")   # 엑셀에서 한글이 깨지지 않게
            print(f"  {path}")

        # 이상 유형별로 따로 기록해야 나중에 "drift에서 무엇이 강했나"를 되짚을 수 있다.
        total = 0
        for ft, d in perf_all:
            total += export_detector_performance(label, d, ft, args.magnitude)
        print(f"  {MONITORING_DB_PATH}  (detector_performance 테이블에 {total}행 추가)")

    print()


if __name__ == "__main__":
    main()
