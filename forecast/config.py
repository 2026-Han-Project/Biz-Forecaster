"""환경 변수(Config) - 매장마다 바꿔 쓰는 값.

기본값은 설계 문서의 '환경 변수 (Config)' 표를 그대로 옮긴 것이다.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

WEEKDAY_NAMES = ["월", "화", "수", "목", "금", "토", "일"]


def default_hourly_profile() -> dict[int, float]:
    """시간대별 판매 비중 - 아침(8시), 하교(15~17시) 피크.

    18시까지 누적 비중이 0.78이 되도록 잡았다(문서 예시: 18시 품절 30개 -> 30/0.78=38개).
    키는 '그 시각부터 한 시간' 구간이다(8 = 8:00~8:59).
    """
    return {
        8: 0.10, 9: 0.09, 10: 0.06, 11: 0.07, 12: 0.08,
        13: 0.06, 14: 0.06, 15: 0.08, 16: 0.09, 17: 0.09,
        18: 0.10, 19: 0.07, 20: 0.05,
    }


@dataclass
class Config:
    # 영업시간과 정기 휴무 요일 (0=월 ... 6=일)
    open_hour: int = 8
    close_hour: int = 21
    closed_weekdays: list[int] = field(default_factory=list)

    # 시간대별 판매 비중. 품절 보정 강도를 결정하므로 매장에 맞게 조정
    hourly_profile: dict[int, float] = field(default_factory=default_hourly_profile)

    # 사장님의 요일 감각(예: 토요일 1.2). 초기에 특히 효과가 큼 (월~일 7개)
    weekday_prior: list[float] = field(default_factory=lambda: [1.0] * 7)

    # 높으면 최근 변화에 빨리 반응 (대신 노이즈도 커짐)
    alpha: float = 0.2
    # 관측을 믿기까지 필요한 입력 일수. 낮을수록 빨리 전환
    k_cred: float = 5
    # 요일 계수를 보수적으로 잡는 정도. 요일 편차가 과해 보이면 높임
    k_weekday: float = 4

    # 품절 보정 상한 / 마감 직전 품절 시 보정
    max_uplift: float = 2.0
    close_uplift: float = 1.05

    # 반죽 손실, 실패, 시식으로 인한 재료 로스
    loss_rate: float = 0.05
    # 초기 단계에서 생산분 중 안 팔린 비율 가정
    prior_waste: float = 0.10

    # 역레시피 계산 기간 / 평소 생산 구성비를 믿는 정도
    bom_window_days: int = 28
    bom_lambda: float = 0.3

    # 데이터 부족 시 수요 변동계수 / 목표 서비스수준 범위
    default_cv: float = 0.25
    sl_bounds: tuple[float, float] = (0.50, 0.95)

    # 이벤트 배수 / 시즌 보정 상한, 하한
    # 'event'(행사, 축제 등)는 문서에 값이 없어 1.3으로 둔 가정값이다.
    event_factors: dict[str, float] = field(
        default_factory=lambda: {"holiday": 1.2, "rain": 0.9, "event": 1.3}
    )
    season_bounds: tuple[float, float] = (0.75, 1.3)

    # ------------------------------------------------------------------
    def __post_init__(self):
        self.hourly_profile = {int(k): float(v) for k, v in self.hourly_profile.items()}
        self.closed_weekdays = sorted({int(d) for d in self.closed_weekdays})
        self.weekday_prior = [float(v) for v in self.weekday_prior]
        self.sl_bounds = tuple(float(v) for v in self.sl_bounds)
        self.season_bounds = tuple(float(v) for v in self.season_bounds)
        self.event_factors = {str(k): float(v) for k, v in self.event_factors.items()}

    def validate(self) -> list[str]:
        """설정값의 명백한 오류를 사람이 읽을 문장으로 돌려준다."""
        errs = []
        if not 0 <= self.open_hour < self.close_hour <= 24:
            errs.append("영업 시작 시각은 마감 시각보다 빨라야 합니다.")
        if len(self.weekday_prior) != 7:
            errs.append("weekday_prior 는 월~일 7개 값이어야 합니다.")
        if len(self.closed_weekdays) >= 7:
            errs.append("모든 요일을 휴무로 둘 수는 없습니다.")
        if not 0 < self.alpha <= 1:
            errs.append("alpha 는 0보다 크고 1 이하여야 합니다.")
        lo, hi = self.sl_bounds
        if not 0 < lo <= hi < 1:
            errs.append("목표 서비스수준 범위는 0~1 사이여야 합니다.")
        lo, hi = self.season_bounds
        if not 0 < lo <= 1 <= hi:
            errs.append("시즌 보정 범위는 하한 <= 1 <= 상한이어야 합니다.")
        if sum(self.open_profile().values()) <= 0:
            errs.append("영업시간 안의 시간대별 판매 비중이 모두 0입니다.")
        return errs

    def open_weekdays(self) -> list[int]:
        return [d for d in range(7) if d not in self.closed_weekdays]

    def open_profile(self) -> dict[int, float]:
        """영업시간 안의 시간대 비중만 추려 합이 1이 되게 정규화한다."""
        prof = {h: max(self.hourly_profile.get(h, 0.0), 0.0)
                for h in range(self.open_hour, self.close_hour)}
        total = sum(prof.values())
        return {h: v / total for h, v in prof.items()} if total > 0 else prof

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        d["hourly_profile"] = {str(k): v for k, v in self.hourly_profile.items()}
        d["sl_bounds"] = list(self.sl_bounds)
        d["season_bounds"] = list(self.season_bounds)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def save(self, path: Path):
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
                              encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Config":
        try:
            return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            return cls()
