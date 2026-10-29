# -*- coding: utf-8 -*-
"""분기 패널 만들기: 편의점 매출·점포·개폐업을 상권 x 분기(2021Q1~2026Q2)로 정리한다.

입력: eda_output/analysis/feature_extended.csv (후보 상권 184개 목록과 기본 열)
출력: eda_output/team/quarterly_panel.csv

팀 발표 자료의 점포당 매출 정의를 그대로 쓴다.
    점포당 분기매출 = 같은 분기 매출 ÷ 같은 분기 SIMILR_INDUTY_STOR_CO(전체 점포 수)
    -> 최근 4분기 평균, 월 환산은 ÷3
(recommend.py는 '매출 평균 ÷ 점포 수 평균'이라 값이 최대 약 1% 다르다.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data_paths import ROOT, filter_convenience, load_sales, load_stores

OUT = ROOT / "eda_output" / "team"
OUT.mkdir(parents=True, exist_ok=True)

NIGHT_COLS = ["TMZON_00_06_SELNG_AMT", "TMZON_21_24_SELNG_AMT"]
DAY_COLS = ["TMZON_06_11_SELNG_AMT", "TMZON_11_14_SELNG_AMT", "TMZON_14_17_SELNG_AMT"]


def build_panel(codes: set[str]) -> pd.DataFrame:
    s = filter_convenience(load_sales())
    s = s[s["TRDAR_CD"].isin(codes)].copy()
    num = ["THSMON_SELNG_AMT", "THSMON_SELNG_CO"] + NIGHT_COLS + DAY_COLS
    for c in num:
        s[c] = pd.to_numeric(s[c], errors="coerce")
    s["night_amt"] = s[NIGHT_COLS].sum(axis=1)
    s["day_amt"] = s[DAY_COLS].sum(axis=1)
    s = s[["TRDAR_CD", "STDR_YYQU_CD", "THSMON_SELNG_AMT", "THSMON_SELNG_CO", "night_amt", "day_amt"]]

    t = filter_convenience(load_stores())
    t = t[t["TRDAR_CD"].isin(codes)].copy()
    for c in ["SIMILR_INDUTY_STOR_CO", "STOR_CO", "FRC_STOR_CO", "OPBIZ_RT", "CLSBIZ_RT"]:
        t[c] = pd.to_numeric(t[c], errors="coerce")
    t = t[["TRDAR_CD", "STDR_YYQU_CD", "SIMILR_INDUTY_STOR_CO", "STOR_CO", "FRC_STOR_CO", "OPBIZ_RT", "CLSBIZ_RT"]]

    p = t.merge(s, on=["TRDAR_CD", "STDR_YYQU_CD"], how="outer")
    p["quarter"] = p["STDR_YYQU_CD"].str[:4] + "Q" + p["STDR_YYQU_CD"].str[4:]
    store = p["SIMILR_INDUTY_STOR_CO"].where(p["SIMILR_INDUTY_STOR_CO"] > 0)
    p["sales_per_store_q"] = p["THSMON_SELNG_AMT"] / store          # 분기 합계 / 점포 수
    p["night_per_store_q"] = p["night_amt"] / store
    p["tx_per_store_day"] = p["THSMON_SELNG_CO"] / store / 90        # 점포 1곳의 하루 결제 건수(분기 90일로 환산)
    return p.sort_values(["TRDAR_CD", "STDR_YYQU_CD"])


def main() -> None:
    e = pd.read_csv(ROOT / "eda_output" / "analysis" / "feature_extended.csv", dtype={"TRDAR_CD": str})
    codes = set(e["TRDAR_CD"])
    p = build_panel(codes)
    p.to_csv(OUT / "quarterly_panel.csv", index=False, encoding="utf-8-sig")
    print("panel", p.shape, "quarters", p["quarter"].nunique(), "->", OUT / "quarterly_panel.csv")


if __name__ == "__main__":
    main()
