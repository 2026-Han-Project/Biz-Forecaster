"""입력 파일 5종을 읽어 표준 컬럼으로 맞춘다.

| 파일            | 내용                                   | 입력 주기      |
|-----------------|----------------------------------------|----------------|
| products.csv    | 제품명, 판매가, 원가, 회수가, 평소 생산량 | 1회            |
| recipe.csv      | 제품 1개당 주요 재료량                   | 1회            |
| purchases.csv   | 날짜, 재료, 수량                         | 과거분 일괄    |
| daily_log.csv   | 날짜, 제품, 결과, (생산량)               | 매일 마감 시   |
| events.csv      | 날짜, 태그                               | 선택           |

사장님이 엑셀에서 한글 머리글로 저장해도 읽히도록 한글 별칭을 함께 받는다.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# 표준 컬럼명 -> 받아들이는 별칭 (공백, 대소문자 무시)
_ALIASES = {
    "products": {
        "product": ["product", "제품명", "제품", "품목", "item"],
        "price": ["price", "판매가", "가격"],
        "cost": ["cost", "원가"],
        "salvage": ["salvage", "회수가", "할인가"],
        "usual_qty": ["usual_qty", "평소생산량", "평소생산", "usual"],
    },
    "recipe": {
        "product": ["product", "제품명", "제품", "품목", "item"],
        "ingredient": ["ingredient", "재료", "재료명"],
        "amount": ["amount", "재료량", "1개당재료량", "수량", "qty"],
    },
    "purchases": {
        "date": ["date", "날짜", "매입일", "일자"],
        "ingredient": ["ingredient", "재료", "재료명"],
        "qty": ["qty", "수량", "매입량", "amount"],
    },
    "daily_log": {
        "date": ["date", "날짜", "일자"],
        "product": ["product", "제품명", "제품", "품목", "item"],
        "result": ["result", "결과"],
        "production": ["production", "생산량"],
    },
    "events": {
        "date": ["date", "날짜", "일자"],
        "tag": ["tag", "태그", "이벤트"],
    },
}

REQUIRED = {
    "products": ["product", "price", "cost", "usual_qty"],
    "recipe": ["product", "ingredient", "amount"],
    "purchases": ["date", "ingredient", "qty"],
    "daily_log": ["date", "product", "result"],
    "events": ["date", "tag"],
}

FILE_NAMES = {
    "products": "products.csv",
    "recipe": "recipe.csv",
    "purchases": "purchases.csv",
    "daily_log": "daily_log.csv",
    "events": "events.csv",
}


class InputError(ValueError):
    pass


def _norm(s: str) -> str:
    return str(s).strip().lstrip("\ufeff").replace(" ", "").replace("_", "").lower()


def _read_csv(src) -> pd.DataFrame:
    """UTF-8(BOM 포함)로 읽고, 실패하면 한글 엑셀 기본값인 CP949로 다시 읽는다."""
    raw = src.getvalue() if hasattr(src, "getvalue") else Path(src).read_bytes()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc, dtype=str,
                               keep_default_na=False, skipinitialspace=True)
        except UnicodeDecodeError:
            continue
    raise InputError("파일 인코딩을 읽을 수 없습니다. UTF-8 또는 CP949로 저장해 주세요.")


def _standardize(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    lookup = {}
    for std, aliases in _ALIASES[kind].items():
        for a in aliases:
            lookup[_norm(a)] = std
    rename = {}
    for col in df.columns:
        std = lookup.get(_norm(col))
        if std and std not in rename.values():
            rename[col] = std
    df = df.rename(columns=rename)[list(rename.values())]
    missing = [c for c in REQUIRED[kind] if c not in df.columns]
    if missing:
        raise InputError(f"{FILE_NAMES[kind]}: 필수 컬럼이 없습니다 - {', '.join(missing)}")
    return df


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.strip(), errors="coerce")


def _to_date(s: pd.Series, kind: str) -> pd.Series:
    d = pd.to_datetime(s.astype(str).str.strip(), errors="coerce")
    if d.isna().any():
        bad = s[d.isna()].head(3).tolist()
        raise InputError(f"{FILE_NAMES[kind]}: 날짜를 읽을 수 없는 행이 있습니다 - {bad}")
    return d.dt.normalize()


def load_products(src) -> pd.DataFrame:
    df = _standardize(_read_csv(src), "products")
    df["product"] = df["product"].str.strip()
    df = df[df["product"] != ""]
    if "salvage" not in df.columns:
        df["salvage"] = "0"
    for c in ("price", "cost", "salvage", "usual_qty"):
        df[c] = _to_num(df[c])
    df["salvage"] = df["salvage"].fillna(0.0)          # 회수가: 없으면 0
    bad = df[df[["price", "cost", "usual_qty"]].isna().any(axis=1)]
    if len(bad):
        raise InputError(f"products.csv: 숫자가 비어 있는 제품이 있습니다 - {bad['product'].tolist()}")
    if df["product"].duplicated().any():
        raise InputError("products.csv: 같은 제품명이 두 번 들어 있습니다.")
    return df.set_index("product")[["price", "cost", "salvage", "usual_qty"]]


def load_recipe(src) -> pd.DataFrame:
    """제품 x 재료 행렬(제품 1개당 재료량)로 돌려준다."""
    df = _standardize(_read_csv(src), "recipe")
    df["product"] = df["product"].str.strip()
    df["ingredient"] = df["ingredient"].str.strip()
    df["amount"] = _to_num(df["amount"]).fillna(0.0)
    df = df[(df["product"] != "") & (df["ingredient"] != "")]
    return df.pivot_table(index="product", columns="ingredient", values="amount",
                          aggfunc="sum", fill_value=0.0)


def load_purchases(src) -> pd.DataFrame:
    df = _standardize(_read_csv(src), "purchases")
    df = df[df["date"].str.strip() != ""]
    df["date"] = _to_date(df["date"], "purchases")
    df["ingredient"] = df["ingredient"].str.strip()
    df["qty"] = _to_num(df["qty"]).fillna(0.0)
    return df[df["qty"] > 0][["date", "ingredient", "qty"]].reset_index(drop=True)


def load_daily_log(src) -> pd.DataFrame:
    df = _standardize(_read_csv(src), "daily_log")
    df = df[df["date"].str.strip() != ""]
    df["date"] = _to_date(df["date"], "daily_log")
    df["product"] = df["product"].str.strip()
    df["result"] = df["result"].astype(str).str.strip()
    if "production" not in df.columns:
        df["production"] = ""
    df["production"] = _to_num(df["production"])        # 빈칸 = 평소대로
    return df[["date", "product", "result", "production"]].reset_index(drop=True)


def load_events(src) -> pd.DataFrame:
    df = _standardize(_read_csv(src), "events")
    df = df[df["date"].str.strip() != ""]
    df["date"] = _to_date(df["date"], "events")
    df["tag"] = df["tag"].str.strip().str.lower()
    return df[df["tag"] != ""][["date", "tag"]].reset_index(drop=True)


def empty_daily_log() -> pd.DataFrame:
    return pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "product": pd.Series(dtype=str),
                         "result": pd.Series(dtype=str), "production": pd.Series(dtype=float)})


def empty_events() -> pd.DataFrame:
    return pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "tag": pd.Series(dtype=str)})


@dataclass
class Inputs:
    products: pd.DataFrame
    recipe: pd.DataFrame
    purchases: pd.DataFrame
    daily_log: pd.DataFrame
    events: pd.DataFrame


def load_inputs(sources: dict) -> Inputs:
    """sources: {'products': 경로|파일객체, ...}. daily_log, events, purchases, recipe 는 없어도 된다.

    products.csv 만 있어도 '평소 생산량' 기반 사전값으로 예측이 돌아간다.
    """
    if sources.get("products") is None:
        raise InputError("products.csv 는 꼭 필요합니다.")
    products = load_products(sources["products"])
    recipe = (load_recipe(sources["recipe"]) if sources.get("recipe") is not None
              else pd.DataFrame(index=pd.Index([], name="product")))
    purchases = (load_purchases(sources["purchases"]) if sources.get("purchases") is not None
                 else pd.DataFrame(columns=["date", "ingredient", "qty"]))
    daily_log = (load_daily_log(sources["daily_log"]) if sources.get("daily_log") is not None
                 else empty_daily_log())
    events = (load_events(sources["events"]) if sources.get("events") is not None
              else empty_events())
    return Inputs(products, recipe, purchases, daily_log, events)
