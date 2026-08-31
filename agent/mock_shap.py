import hashlib

# 실제 SHAP도 동일 형식으로 받을 예정: (feature, contribution)
_FEATURES = ["기온", "강수량", "공휴일", "주말효과", "전주판매량", "프로모션"]


def get_contributions(item: str, top_n: int = 3) -> list[dict]:
    seed = int(hashlib.md5(item.encode("utf-8")).hexdigest(), 16)

    result = []
    for i, feat in enumerate(_FEATURES):
        raw = (seed >> (i * 5)) % 200 - 100   
        result.append({"feature": feat, "contribution": round(raw / 10, 1)})

    # 영향력 큰 순으로 정렬
    result.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    return result[:top_n]


def format_contributions(contribs: list[dict]) -> str:
    parts = []
    for c in contribs:
        sign = "+" if c["contribution"] >= 0 else ""
        parts.append(f"{c['feature']} {sign}{c['contribution']}개")
    return ", ".join(parts)