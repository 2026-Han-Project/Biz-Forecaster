"""이상탐지 결과를 SQLite로 내보내 Grafana에서 보게 하는 계층.

원래 app.py 안에 있었는데, 화면(소상공인용 알림)과 분석(팀원 C용 검증)을 갈라내면서
분석 쪽으로 옮겼다. 이제 analyze_detectors.py 가 이 함수들을 부른다.

Grafana 는 grafana-sqlite-datasource 플러그인으로 이 DB 파일을 직접 읽는다.
(analyze_detectors.py -> monitoring.db -> Grafana, 별도 API 서버 없이 파일 기반)
"""

import sqlite3
from pathlib import Path

import pandas as pd

MONITORING_DB_PATH = Path(__file__).resolve().parent.parent / "monitoring.db"


def export_monitoring_metrics(target, sigma_df, ewma_df, iso_flags, combined, scope_df=None,
                               db_path=MONITORING_DB_PATH, cusum_df=None):
    """이상탐지 탭에서 계산된 결과를 monitoring.db의 daily_metrics 테이블로 내보낸다."""
    scope_map = {}
    if scope_df is not None and not scope_df.empty:
        scope_map = {
            pd.Timestamp(row['날짜']): row['구분'] for _, row in scope_df.iterrows()
        }

    rows = []
    for d in combined.index:
        rows.append({
            'date': d.strftime('%Y-%m-%d'),
            'target': target,
            'value': float(sigma_df.loc[d, 'value']),
            'center': None if pd.isna(sigma_df.loc[d, 'center']) else float(sigma_df.loc[d, 'center']),
            'ucl': None if pd.isna(sigma_df.loc[d, 'ucl']) else float(sigma_df.loc[d, 'ucl']),
            'lcl': None if pd.isna(sigma_df.loc[d, 'lcl']) else float(sigma_df.loc[d, 'lcl']),
            'ewma': None if pd.isna(ewma_df.loc[d, 'ewma']) else float(ewma_df.loc[d, 'ewma']),
            'cusum_pos': None if cusum_df is None else float(cusum_df.loc[d, 'cusum_pos']),
            'cusum_neg': None if cusum_df is None else float(cusum_df.loc[d, 'cusum_neg']),
            'is_anomaly_3sigma': int(bool(combined.loc[d, '3시그마'])),
            'is_anomaly_ewma': int(bool(combined.loc[d, 'EWMA'])),
            'is_anomaly_cusum': int(bool(combined.loc[d, 'CUSUM'])) if 'CUSUM' in combined else 0,
            'is_anomaly_iso': int(bool(combined.loc[d, 'IsolationForest'])),
            'detection_count': int(combined.loc[d, '탐지방법수']),
            'is_anomaly': int(bool(combined.loc[d, '이상여부'])),
            'scope': scope_map.get(pd.Timestamp(d), ''),
            'exported_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        })
    out_df = pd.DataFrame(rows)

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_metrics (
                date TEXT NOT NULL,
                target TEXT NOT NULL,
                value REAL,
                center REAL,
                ucl REAL,
                lcl REAL,
                ewma REAL,
                cusum_pos REAL,
                cusum_neg REAL,
                is_anomaly_3sigma INTEGER,
                is_anomaly_ewma INTEGER,
                is_anomaly_cusum INTEGER,
                is_anomaly_iso INTEGER,
                detection_count INTEGER,
                is_anomaly INTEGER,
                scope TEXT,
                exported_at TEXT,
                PRIMARY KEY (date, target)
            )
        """)
        conn.execute("DELETE FROM daily_metrics WHERE target = ?", (target,))
        out_df.to_sql('daily_metrics', conn, if_exists='append', index=False)
        conn.commit()
    finally:
        conn.close()

    return len(out_df)


def export_detector_performance(target, perf_df, fault_type, magnitude,
                                 db_path=MONITORING_DB_PATH):
    """탭 8에서 계산한 탐지기별 오탐/미탐 성능을 monitoring.db에 남긴다.

    파라미터를 바꿔가며 돌린 결과가 쌓이므로, 나중에 "왜 h=5로 정했는지"를
    기록으로 되짚을 수 있다. 반도체에서 관리한계 변경 이력을 남기는 것과 같은 취지다.
    """
    df = perf_df.copy()
    df.insert(0, 'target', target)
    df.insert(1, 'fault_type', fault_type)
    df.insert(2, 'magnitude', magnitude)
    df['evaluated_at'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

    conn = sqlite3.connect(db_path)
    try:
        df.to_sql('detector_performance', conn, if_exists='append', index=False)
        conn.commit()
    finally:
        conn.close()
    return len(df)
