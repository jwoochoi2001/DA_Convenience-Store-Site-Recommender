# -*- coding: utf-8 -*-
"""팀 발표 자료의 지표 정의로 상권 피처를 다시 만든다.

recommend.py와 다른 점
- 점포당 매출: 분기별 (매출 ÷ 점포 수)를 4분기 평균(team_panel.py)
- 경쟁밀도: 상권 편의점 수 ÷ 분기 유동인구(10만명). 최신 분기 점포 수.
- 상권 안정성: 03번 상권변화지표의 평균 영업개월(최신 분기)
- 심야 유동 비중: (00~06시 + 21~24시) 유동 ÷ 전체 유동
- 아파트 세대수: 상권 경계(폴리곤)에서 400m 이내 단지 세대수 합. 서초 추가 수집분 포함.
- 접근성: 상권 중심점 기준 지하철(400m)x2 + 버스(300m)  -> 팀 값과 B 후보 10곳 모두 일치

출력: eda_output/team/feature_team.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Polygon
from shapely.ops import unary_union

from data_paths import ROOT, load_apt_basic, load_apt_geo, load_street_pop, load_trade_area_boundary
from geo_utils import wgs84_arr_to_5181
from team_spec import HOURLY_WAGE, NIGHT_STAFF_HOURS, DAYS_PER_MONTH

OUT = ROOT / "eda_output" / "team"
APT_DIR = ROOT / "data" / "data_go_kr" / "12_13_18_공동주택"


def _read(p):
    return pd.read_csv(p, dtype=str, encoding="utf-8-sig")


def load_apartments() -> pd.DataFrame:
    """강남·송파 + 서초 추가 수집분을 합치고 단지코드로 중복을 뺀다."""
    basic = [load_apt_basic()]
    geo = [load_apt_geo()]
    sc_b, sc_g = APT_DIR / "공동주택_서초_기본정보.csv", APT_DIR / "공동주택_서초_좌표변환.csv"
    if sc_b.exists() and sc_g.exists():
        basic.append(_read(sc_b))
        geo.append(_read(sc_g))
    basic = pd.concat(basic, ignore_index=True).drop_duplicates("kaptCode")
    geo = pd.concat(geo, ignore_index=True).drop_duplicates("kaptCode")
    geo = geo[geo["status"] == "OK"][["kaptCode", "x", "y"]].copy()
    geo["x"], geo["y"] = pd.to_numeric(geo["x"]), pd.to_numeric(geo["y"])
    # 세대수 = kaptdaCnt. 관리정보시스템에 세대수가 0으로 올라온 단지(청년주택·오피스텔형 등)는 호수(hoCnt)로 채운다.
    da = pd.to_numeric(basic["kaptdaCnt"], errors="coerce").fillna(0)
    ho = pd.to_numeric(basic["hoCnt"], errors="coerce").fillna(0)
    basic["households"] = np.where(da > 0, da, ho)
    apt = geo.merge(basic[["kaptCode", "households"]], on="kaptCode", how="left").fillna({"households": 0})
    apt["X"], apt["Y"] = wgs84_arr_to_5181(apt["x"].values, apt["y"].values)
    return apt


def boundary_polygons(codes) -> dict:
    bdf, shapes = load_trade_area_boundary()
    idx = {c: i for i, c in enumerate(bdf["TRDAR_CD"])}
    out = {}
    for c in codes:
        shp = shapes[idx[c]]
        parts = list(shp.parts) + [len(shp.points)]
        polys = [Polygon(shp.points[parts[p]:parts[p + 1]]).buffer(0) for p in range(len(parts) - 1) if len(shp.points[parts[p]:parts[p + 1]]) >= 4]
        out[c] = unary_union(polys)
    return out


def build() -> pd.DataFrame:
    e = pd.read_csv(ROOT / "eda_output" / "analysis" / "feature_extended.csv", dtype={"TRDAR_CD": str}).set_index("TRDAR_CD")
    panel = pd.read_csv(OUT / "quarterly_panel.csv", dtype={"TRDAR_CD": str, "STDR_YYQU_CD": str})
    qs = sorted(panel["STDR_YYQU_CD"].unique())
    last4, latest = qs[-4:], qs[-1]

    # 점포당 매출(월, 원): 분기별 (매출/점포 수) 4분기 평균 ÷ 3. 4분기 모두 있어야 한다.
    p4 = panel[panel["STDR_YYQU_CD"].isin(last4)]
    g = p4.groupby("TRDAR_CD")
    n_ok = g["sales_per_store_q"].count()
    f = pd.DataFrame(index=e.index)
    f["spq_month"] = (g["sales_per_store_q"].mean() / 3).where(n_ok == 4).reindex(f.index)
    f["night_per_store_month"] = (g["night_per_store_q"].mean() / 3).where(n_ok == 4).reindex(f.index)
    f["tx_per_store_day"] = g["tx_per_store_day"].mean().where(n_ok == 4).reindex(f.index)
    f["stores_latest"] = panel[panel["STDR_YYQU_CD"] == latest].set_index("TRDAR_CD")["SIMILR_INDUTY_STOR_CO"].reindex(f.index)
    f["stores_avg4"] = g["SIMILR_INDUTY_STOR_CO"].mean().reindex(f.index)
    f["night_labor_coverage"] = f["night_per_store_month"] / (HOURLY_WAGE * NIGHT_STAFF_HOURS * DAYS_PER_MONTH)

    # 유동인구: 심야 비중(00~06 + 21~24)은 최신 분기, 경쟁밀도의 분모는 최근 4분기 평균
    st_all = load_street_pop()
    st_all = st_all[st_all["TRDAR_CD"].isin(set(e.index))].copy()
    for c in ["TOT_FLPOP_CO", "TMZON_00_06_FLPOP_CO", "TMZON_21_24_FLPOP_CO"]:
        st_all[c] = pd.to_numeric(st_all[c])
    fl_q = sorted(st_all["STDR_YYQU_CD"].unique())
    st = st_all[st_all["STDR_YYQU_CD"] == fl_q[-1]].set_index("TRDAR_CD")
    f["flpop"] = st["TOT_FLPOP_CO"].reindex(f.index)
    f["flpop_avg4"] = st_all[st_all["STDR_YYQU_CD"].isin(fl_q[-4:])].groupby("TRDAR_CD")["TOT_FLPOP_CO"].mean().reindex(f.index)
    f["night_flpop_ratio"] = ((st["TMZON_00_06_FLPOP_CO"] + st["TMZON_21_24_FLPOP_CO"]) / st["TOT_FLPOP_CO"]).reindex(f.index)
    f["competition"] = f["stores_avg4"] / (f["flpop_avg4"] / 1e5)      # 유동 10만명당 편의점 수(4분기 평균)

    # 아파트 세대수(상권 경계 400m)
    apt = load_apartments()
    polys = boundary_polygons(f.index)
    f["apt_households"] = [float(apt["households"].values[shapely.contains_xy(polys[c].buffer(400), apt["X"].values, apt["Y"].values)].sum()) for c in f.index]

    # 재사용 열
    keep = ["상권명", "구", "행정동", "상권구분", "트랙", "XCNTS_VALU", "YDNTS_VALU", "면적_ha", "dist_to_dogok_m",
            "night_sales_ratio", "daytime_sales_ratio", "population_total", "TOT_REPOP_CO", "TOT_WRC_POPLTN_CO",
            "accessibility", "opr_months", "cvs_store_avg", "이상값_점포당매출", "rent_per_m2_kwon", "rent_area",
            "age20_ratio", "age30_ratio", "sales_yoy", "open_rate", "close_rate", "매출분기수"]
    out = e[keep].join(f)
    out = out.rename(columns={"opr_months": "stability"})
    # 팀 정의로 다시 만든 매출이 없으면(4분기 미만) 실측에서 뺀다
    out.loc[out["spq_month"].isna() & (out["트랙"] == "실측"), "트랙"] = "매출불명"
    out["outlier"] = False
    m = out["트랙"] == "실측"
    q1, q3 = out.loc[m, "spq_month"].quantile([0.25, 0.75])
    out.loc[m, "outlier"] = out.loc[m, "spq_month"] > q3 + 1.5 * (q3 - q1)
    out["small_n"] = out["cvs_store_avg"] < 4.0
    return out


def main() -> None:
    out = build()
    out.to_csv(OUT / "feature_team.csv", encoding="utf-8-sig")
    print("feature_team", out.shape, "->", OUT / "feature_team.csv")
    print(out["트랙"].value_counts().to_dict(), "outliers", int(out["outlier"].sum()))


if __name__ == "__main__":
    main()
