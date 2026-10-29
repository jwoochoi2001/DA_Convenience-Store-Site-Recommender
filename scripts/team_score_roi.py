# -*- coding: utf-8 -*-
"""팀 발표 자료의 입지 점수·ROI를 재현한다.

입지 점수  = Σ ( 가중치 x 후보 안 백분위 )   (경쟁밀도는 가중치가 음수라 백분위를 그대로 뺀다 = 감점)
ROI        = 월 순이익 x 12 ÷ 필요현금
월 순이익  = 점포당 월매출 x 25% x 경영주 배분율 − 알바 인건비 − 월세(점주 임차형만)
가맹 타입  = 예산 안에서 ROI가 가장 큰 24시간 타입 (CU GREEN 1·2는 월 임차부담금 불명, 이마트24 NH 1은 전대보증금 금액 불명이라 제외)

결과: eda_output/team/{score_A,score_B,roi_A,roi_B}.csv, validation.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import team_spec as T
from data_paths import ROOT, load_franchise_terms, load_rent

OUT = ROOT / "eda_output" / "team"
EXCLUDE_TYPES = {("CU", "GREEN 1 (점주투자안정형 I)"), ("CU", "GREEN 2 (점주투자안정형 II)")}   # 월 임차부담금 금액 불명


# ------------------------------------------------------------------ 입지 점수
def team_score(pool: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """가중치 x 백분위의 합. 지표가 결측이면 그 지표를 빼고 가중치 합으로 나눠 다시 맞춘다."""
    df = pool.copy()
    total = pd.Series(0.0, index=df.index)
    wsum = pd.Series(0.0, index=df.index)
    for col, w in weights.items():
        pct = df[col].rank(pct=True)
        df[f"{col}_pct"] = pct
        total += (w * pct).fillna(0)
        wsum += np.where(pct.notna(), abs(w), 0)
    full = sum(abs(w) for w in weights.values())
    df["score"] = total * (full / wsum.where(wsum > 0))
    df["rank_loc"] = df["score"].rank(ascending=False, method="first").astype(int)
    return df.sort_values("score", ascending=False)


def pools(f: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    a = f[(f["구"].isin(["강남구", "송파구"])) & (f["트랙"] == "실측")]
    b = f[(f["dist_to_dogok_m"] <= T.B_RADIUS_M) & (f["트랙"] == "실측")]
    return a, b


# ------------------------------------------------------------------ 임대료(구 단위)와 가맹 타입
def gu_rent_per_m2() -> dict:
    """R-ONE 최신 분기 1층 임대료(천원/㎡·월)를 구 단위로 모은다. 강남=서울>강남, 송파=3개 권역 평균, 서초=2개 권역 평균."""
    r = load_rent()
    r["DTA_VAL"] = pd.to_numeric(r["DTA_VAL"], errors="coerce")
    r = r[r["WRTTIME_IDTFR_ID"] == r["WRTTIME_IDTFR_ID"].max()]
    v = dict(zip(r["CLS_FULLNM"], r["DTA_VAL"]))
    return {
        "강남구": v["서울>강남"],
        "송파구": float(np.mean([v["서울>기타>잠실/송파"], v["서울>기타>잠실새내역"], v["서울>기타>가락시장"]])),
        "서초구": float(np.mean([v["서울>강남>남부터미널"], v["서울>강남>교대역"]])),
    }


def franchise_types() -> pd.DataFrame:
    fr = load_franchise_terms()
    for c in ["개점투자_소계_만원", "담보_만원", "경영주_최종배분율", "최저수입보장_월_만원"]:
        fr[c] = pd.to_numeric(fr[c], errors="coerce").fillna(0)
    # 24시간 타입: 운영시간에 '24'가 들어가고 '미영업'이 없는 행. 같은 (브랜드, 가맹타입)이 여러 행이면 파일의 첫 행만 쓴다.
    # (이마트24는 '24H 미운영' 행이 먼저 나와 배분율이 낮은 쪽이 선택된다. 발표 자료의 가맹 타입 16개와 같은 규칙이다.)
    fr = fr[fr["운영시간"].str.contains("24") & ~fr["운영시간"].str.contains("미영업")].copy()
    fr = fr.drop_duplicates(["브랜드", "가맹타입"], keep="first")
    fr["cash"] = fr["개점투자_소계_만원"] + fr["담보_만원"]
    fr["share"] = fr["경영주_최종배분율"] / 100.0
    fr["owner_rent"] = fr["점포임차_부담"].isin(["가맹점주", "경영주"])      # 점주가 월세를 내는 타입
    fr["type"] = fr["브랜드"] + " " + fr["가맹타입"]
    mask = [(b, t) not in EXCLUDE_TYPES for b, t in zip(fr["브랜드"], fr["가맹타입"])]
    return fr[mask].reset_index(drop=True)


def evaluate(pool: pd.DataFrame, persona: str, hours: float | None = None, gm: float = T.GROSS_MARGIN,
             owner_labor_man: float = 0.0, extra_cash: dict | None = None, types: pd.DataFrame | None = None,
             rents: dict | None = None) -> pd.DataFrame:
    """상권마다 예산 안 ROI 최대 타입을 골라 월 순이익·ROI·손익분기를 계산한다.

    extra_cash: {'GS25 GS2': 3000} 처럼 필요현금에 더할 금액(전대보증금 시나리오)
    owner_labor_man: 점주 본인 노동을 비용으로 넣을 때 월 환산액(만원)
    """
    hours = T.HOURS[persona] if hours is None else hours
    wage = T.monthly_wage_man(hours)
    types = franchise_types() if types is None else types
    rents = gu_rent_per_m2() if rents is None else rents
    extra_cash = extra_cash or {}
    budget = T.BUDGET_MAN[persona]
    rows = []
    for code, r in pool.iterrows():
        sales = r["spq_month"] / 1e4
        rent = rents[r["구"]] * T.STORE_AREA_M2 / 10
        best = None
        for _, t in types.iterrows():
            cash = t["cash"] + extra_cash.get(t["type"], 0.0)
            if cash > budget:
                continue
            rent_t = rent if t["owner_rent"] else 0.0
            fixed = wage + rent_t + owner_labor_man
            net = sales * gm * t["share"] - fixed
            roi = net * 12 / cash
            if best is None or roi > best["roi"]:
                best = {"type": t["type"], "share": t["share"], "cash": cash, "rent": rent_t, "net": net, "roi": roi,
                        "bep": fixed / (gm * t["share"])}
        if best is None:
            best = {"type": "예산 안 타입 없음", "share": np.nan, "cash": np.nan, "rent": np.nan, "net": np.nan, "roi": np.nan, "bep": np.nan}
        rows.append({"TRDAR_CD": code, "sales_man": sales, **best})
    out = pd.DataFrame(rows).set_index("TRDAR_CD")
    out["payback_months"] = np.where(out["net"] > 0, out["cash"] / out["net"], np.nan)
    out["safety_margin"] = out["sales_man"] / out["bep"] - 1
    out["profit"] = out["net"] > 0
    return out


def roi_table(pool: pd.DataFrame, persona: str, **kw) -> pd.DataFrame:
    sc = team_score(pool, T.WEIGHTS_A_TEAM if persona == "A" else T.WEIGHTS_B_TEAM)
    ev = evaluate(sc, persona, **kw)
    df = sc[["상권명", "구", "상권구분", "score", "rank_loc", "cvs_store_avg", "night_sales_ratio", "outlier", "small_n"]].join(ev)
    df["rank_roi"] = df["roi"].rank(ascending=False, method="first").astype(int)       # 이상값 포함, 적자는 뒤
    pick = df[df["profit"] & ~df["outlier"]].sort_values("roi", ascending=False)
    df["rec_rank"] = pick["roi"].rank(ascending=False, method="first").reindex(df.index)
    return df.sort_values("rank_loc")


# ------------------------------------------------------------------ 민감도: 가중치 ±30%
def top3_frequency(scored: pd.DataFrame, weights: dict, n: int = 3000, spread: float = 0.3, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    cols = list(weights)
    P = np.column_stack([scored[f"{c}_pct"].fillna(scored[f"{c}_pct"].median()).values for c in cols])
    base = np.array([weights[c] for c in cols])
    cnt = np.zeros(len(scored))
    for _ in range(n):
        w = base * rng.uniform(1 - spread, 1 + spread, size=len(cols))
        cnt[np.argsort(-(P @ w))[:3]] += 1
    return pd.Series(cnt / n, index=scored.index)


# ------------------------------------------------------------------ 검증
def validate(a_tbl: pd.DataFrame, b_tbl: pd.DataFrame, f: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, v in T.DECK["A_input_top5"].items():
        r = a_tbl[a_tbl["상권명"] == name].iloc[0]
        rows.append(("A 입지 점수", name, v, round(r["score"], 3), int(r["rank_loc"])))
    for name, v in T.DECK["B_input_top5"].items():
        r = b_tbl[b_tbl["상권명"] == name].iloc[0]
        rows.append(("B 입지 점수", name, v, round(r["score"], 3), int(r["rank_loc"])))
    for name, v in T.DECK["B_apt_households"].items():
        r = f[f["상권명"] == name].iloc[0]
        rows.append(("B 아파트 세대수(경계 400m)", name, v, int(r["apt_households"]), None))
    for name, v in T.DECK["A_net_top5"].items():
        r = a_tbl[a_tbl["상권명"] == name].iloc[0]
        rows.append(("A 입지 TOP5 월 순이익(만원)", name, v, round(r["net"]), None))
    for name, v in T.DECK["B_net_top5"].items():
        r = b_tbl[b_tbl["상권명"] == name].iloc[0]
        rows.append(("B 입지 TOP5 월 순이익(만원)", name, v, round(r["net"]), None))
    for name, v in T.DECK["A_roi_top3"].items():
        r = a_tbl[a_tbl["상권명"] == name].iloc[0]
        rows.append(("A ROI(%)", name, v, round(r["roi"] * 100), int(r["rank_roi"])))
    for name, v in T.DECK["B_roi_top3"].items():
        r = b_tbl[b_tbl["상권명"] == name].iloc[0]
        rows.append(("B ROI(%)", name, v, round(r["roi"] * 100), int(r["rank_roi"])))
    rows.append(("A 손익분기(만원)", "GS25 GS2·18h", T.DECK["A_bep_man"], round(a_tbl["bep"].iloc[0]), None))
    rows.append(("B 손익분기(만원)", "GS25 GS2·12h", T.DECK["B_bep_man"], round(b_tbl.loc[b_tbl["profit"], "bep"].iloc[0]), None))
    rows.append(("A 흑자 상권 수", "90곳 중", T.DECK["A_profit_count"], int(a_tbl["profit"].sum()), None))
    rows.append(("B 흑자 상권 수", "10곳 중", T.DECK["B_profit_count"], int(b_tbl["profit"].sum()), None))
    v = pd.DataFrame(rows, columns=["항목", "대상", "발표 자료", "이 코드", "이 코드 순위"])
    v["차이"] = v["이 코드"] - v["발표 자료"]
    return v


def main() -> None:
    f = pd.read_csv(OUT / "feature_team.csv", dtype={"TRDAR_CD": str}).set_index("TRDAR_CD")
    pa, pb = pools(f)
    a_tbl = roi_table(pa, "A")
    b_tbl = roi_table(pb, "B")
    sa = team_score(pa, T.WEIGHTS_A_TEAM)
    sb = team_score(pb, T.WEIGHTS_B_TEAM)
    sa["top3_freq"] = top3_frequency(sa, T.WEIGHTS_A_TEAM)
    sb["top3_freq"] = top3_frequency(sb, T.WEIGHTS_B_TEAM)
    a_tbl["top3_freq"] = sa["top3_freq"]
    b_tbl["top3_freq"] = sb["top3_freq"]
    a_tbl.to_csv(OUT / "roi_A.csv", encoding="utf-8-sig")
    b_tbl.to_csv(OUT / "roi_B.csv", encoding="utf-8-sig")
    sa.to_csv(OUT / "score_A.csv", encoding="utf-8-sig")
    sb.to_csv(OUT / "score_B.csv", encoding="utf-8-sig")
    v = validate(a_tbl, b_tbl, f)
    v.to_csv(OUT / "validation.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(gu_rent_per_m2())
    print(v.to_string(index=False))
    print(a_tbl.sort_values("rank_loc").head(5)[["상권명", "score", "net", "roi", "type"]].round(3))
    print(b_tbl.sort_values("rank_loc")[["상권명", "score", "net", "roi", "type", "profit"]].round(3))


if __name__ == "__main__":
    main()
