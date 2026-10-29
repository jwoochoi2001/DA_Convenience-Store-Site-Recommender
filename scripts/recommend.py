# -*- coding: utf-8 -*-
"""페르소나 기반 편의점 입지 추천 — 전처리_기준서.md(v3) 반영판.

후보 단위는 상권(TRDAR_CD)이다.

페르소나 A (29세, 청년창업, 자본 2천~5천만원, 지역 무관, 인건비/심야매출 중요):
    - 강남+송파 전체 173개 상권 중 "실측" 트랙(편의점 매출 행이 있는 상권)만 점수화한다.
페르소나 B (59세, 은퇴부부, 자본 2.5억, 강남구 도곡동 인근, 확장 염두):
    - 도곡동 상권 중심점 기준 반경 1.5km 내 상권(자치구 무관, 서초구 포함)을 대상으로
      "실측" 트랙만 점수화한다. (기준서 13-2: 서초구 상권 포함 결정)

트랙 분리, 백분위 정규화, 면적 보정, 반경 400m(지하철) 등은 전처리_기준서.md 2·4·6·8~11·13절을 따른다.
"""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon

from data_paths import (
    ROOT,
    filter_convenience,
    gangnam_songpa_trdar_codes,
    load_apt_basic,
    load_apt_geo,
    load_bus_stops,
    load_change_index,
    load_convenience_stores,
    load_facilities,
    load_franchise_terms,
    load_opex_baseline,
    load_resident_pop,
    load_sales,
    load_street_pop,
    load_stores,
    load_subway,
    load_trade_area_boundary,
    load_worker_pop,
)
from geo_utils import (
    count_within_radius,
    epsg5181_to_wgs84,
    iqr_outlier_flag,
    percentile_normalize,
    wgs84_arr_to_5181,
)

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

FIG_DIR = ROOT / "eda_output" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = ROOT / "eda_output"

# ---- 반경 상수 (m) — 전처리_기준서.md 13-4: 지하철 400m로 통일 ----
APT_RADIUS_M = 400       # 아파트 세대수 합산 반경
SUBWAY_RADIUS_M = 400    # 지하철역 접근성 반경 (기존 600m에서 변경)
BUS_RADIUS_M = 300       # 버스정류소 접근성 반경
CVS_RADIUS_M = 300       # 기존 편의점 경쟁 밀도 반경
B_NEARBY_RADIUS_M = 1500 # 페르소나B "도곡동 인근" 반경 (자치구 무관, 13-2)

# 실측 트랙 최소 월매출(원). 이보다 작으면 극소액 노이즈로 보고 실측에서 뺀다.
# 실측 중 하위 2위가 월 1,530만원, 최하위(양재천카페거리)가 월 5.2만원이라 그 사이 간극에 둔 값.
MIN_MEASURED_SALES_MONTH = 1_000_000

# 실측 트랙 최소 매출 분기 수(최근 4분기 중). 4면 4분기 모두 매출이 있어야 실측.
# 일부 분기만 공표된 상권은 분기 평균이 계절성(1분기 낮고 3분기 높음)에 끌려 비교가 어렵다.
# 현재 데이터는 분기 수가 1·2·4로만 나뉘어 3으로 해도 결과가 같다.
MIN_MEASURED_QUARTERS = 4

# ---- 페르소나별 가중치 (0~1 정규화 후 가중합, 상대비교용) ----
# 심야 "금액" 기준으로 조정(시뮬레이션 S1): night_labor_coverage = 점포당 심야금액 ÷ 상수라 같은 지표.
# 비율 지표(night_sales_ratio, night_flpop_ratio)는 심야 금액과 상관이 약해(+0.27, +0.05) 비중을 낮췄다.
WEIGHTS_A = {
    "night_sales_ratio": 0.10,
    "night_flpop_ratio": 0.05,
    "sales_per_store": 0.15,
    "night_labor_coverage": 0.40,
    "population_total": 0.10,
    "chnge_stability": 0.05,
    "competition_density": -0.15,
}

WEIGHTS_B = {
    "chnge_stability": 0.20,
    "apt_households": 0.20,
    "daytime_sales_ratio": 0.15,
    "population_total": 0.15,
    "accessibility": 0.15,
    "sales_per_store": 0.10,
    "competition_density": -0.10,
}


def _latest_n_quarters(df: pd.DataFrame, n: int, q_col: str = "STDR_YYQU_CD") -> pd.DataFrame:
    quarters = sorted(df[q_col].unique())[-n:]
    return df[df[q_col].isin(quarters)]


def _latest_quarter(df: pd.DataFrame, q_col: str = "STDR_YYQU_CD") -> pd.DataFrame:
    latest = df[q_col].max()
    return df[df[q_col] == latest]


def build_feature_table(trdar_codes: set[str], boundary_df: pd.DataFrame) -> pd.DataFrame:
    """상권 코드 집합에 대해 피처 테이블(행=TRDAR_CD)을 만든다. 자치구 무관(서초구 포함 가능)."""

    base = boundary_df[boundary_df["TRDAR_CD"].isin(trdar_codes)].copy()
    base = base.drop_duplicates(subset="TRDAR_CD")
    base["lon"], base["lat"] = epsg5181_to_wgs84(
        base["XCNTS_VALU"].astype(float).values, base["YDNTS_VALU"].astype(float).values
    )
    feat = base.set_index("TRDAR_CD")[
        ["TRDAR_CD_N", "TRDAR_SE_1", "SIGNGU_CD_", "ADSTRD_CD_", "XCNTS_VALU", "YDNTS_VALU", "RELM_AR", "lon", "lat"]
    ].copy()
    feat.rename(
        columns={"TRDAR_CD_N": "상권명", "TRDAR_SE_1": "상권구분", "SIGNGU_CD_": "구", "ADSTRD_CD_": "행정동"},
        inplace=True,
    )
    feat["XCNTS_VALU"] = feat["XCNTS_VALU"].astype(float)
    feat["YDNTS_VALU"] = feat["YDNTS_VALU"].astype(float)
    feat["RELM_AR"] = feat["RELM_AR"].astype(float)
    feat["면적_ha"] = feat["RELM_AR"] / 10000

    # ---- 01/02: 편의점 매출·점포 (최근 4분기 평균, 월 단위) ----
    # THSMON_SELNG_AMT는 분기 합계라 /3으로 월평균 환산한다(기준서 5절).
    # 주간 = 06~17시(TMZON_06_11+11_14+14_17), 심야 = 21~06시 — 기준서 2절.
    sales = filter_convenience(load_sales())
    sales = sales[sales["TRDAR_CD"].isin(trdar_codes)].copy()
    for c in ["THSMON_SELNG_AMT", "THSMON_SELNG_CO", "TMZON_21_24_SELNG_AMT", "TMZON_00_06_SELNG_AMT",
              "TMZON_06_11_SELNG_AMT", "TMZON_11_14_SELNG_AMT", "TMZON_14_17_SELNG_AMT"]:
        sales[c] = pd.to_numeric(sales[c])
    sales["night_amt"] = sales["TMZON_21_24_SELNG_AMT"] + sales["TMZON_00_06_SELNG_AMT"]
    sales["day_amt"] = sales["TMZON_06_11_SELNG_AMT"] + sales["TMZON_11_14_SELNG_AMT"] + sales["TMZON_14_17_SELNG_AMT"]
    sales4 = _latest_n_quarters(sales, 4)
    sales_agg = sales4.groupby("TRDAR_CD").agg(
        cvs_sales_avg=("THSMON_SELNG_AMT", "mean"),
        night_amt_avg=("night_amt", "mean"),
        day_amt_avg=("day_amt", "mean"),
    )
    sales_agg = sales_agg / 3  # 분기 합계 평균 -> 월평균
    feat = feat.join(sales_agg, how="left")
    # 최근 4분기 중 매출 행이 실제로 있는 분기 수. 평균이 몇 개 분기로 만들어졌는지 보여준다.
    n_q = sales4.groupby("TRDAR_CD")["STDR_YYQU_CD"].nunique().rename("매출분기수")
    feat = feat.join(n_q, how="left")
    feat["매출분기수"] = feat["매출분기수"].fillna(0).astype(int)
    # 분모(매출)가 0/결측이면 비율도 결측 — 0으로 채우지 않는다(기준서 6절).
    feat["night_sales_ratio"] = feat["night_amt_avg"] / feat["cvs_sales_avg"].where(feat["cvs_sales_avg"] > 0)
    feat["daytime_sales_ratio"] = feat["day_amt_avg"] / feat["cvs_sales_avg"].where(feat["cvs_sales_avg"] > 0)

    # 점포수 분모는 SIMILR_INDUTY_STOR_CO(전체 점포=일반+프랜차이즈) — 기준서 4절.
    stores = filter_convenience(load_stores())
    stores = stores[stores["TRDAR_CD"].isin(trdar_codes)].copy()
    stores["SIMILR_INDUTY_STOR_CO"] = pd.to_numeric(stores["SIMILR_INDUTY_STOR_CO"])
    stores4 = _latest_n_quarters(stores, 4)
    store_agg = stores4.groupby("TRDAR_CD").agg(cvs_store_avg=("SIMILR_INDUTY_STOR_CO", "mean"))
    feat = feat.join(store_agg, how="left")
    feat["cvs_store_avg"] = feat["cvs_store_avg"].fillna(0)
    feat["cvs_sales_avg"] = feat["cvs_sales_avg"].fillna(0)
    feat["night_amt_avg"] = feat["night_amt_avg"].fillna(0)

    # 점포당 지표도 분모(점포수) 0이면 결측 — 0으로 나누지 않는다(기준서 4절 "통합 추가" 적용).
    store_denom = feat["cvs_store_avg"].where(feat["cvs_store_avg"] > 0)
    feat["sales_per_store"] = feat["cvs_sales_avg"] / store_denom
    feat["night_amt_per_store"] = feat["night_amt_avg"] / store_denom

    # ---- 03: 상권변화지표 (최신 분기) ----
    change = _latest_quarter(load_change_index())
    change = change[change["TRDAR_CD"].isin(trdar_codes)][["TRDAR_CD", "TRDAR_CHNGE_IX_NM"]]
    stability_map = {"상권확장": 1.0, "정체": 0.6, "다이나믹": 0.3, "상권축소": 0.0}
    change["chnge_stability"] = change["TRDAR_CHNGE_IX_NM"].map(stability_map)
    feat = feat.join(change.set_index("TRDAR_CD")[["TRDAR_CHNGE_IX_NM", "chnge_stability"]], how="left")
    feat["chnge_stability"] = feat["chnge_stability"].fillna(0.5)

    # ---- 04: 길단위(유동)인구 (최신 분기) ----
    street = _latest_quarter(load_street_pop())
    street = street[street["TRDAR_CD"].isin(trdar_codes)].copy()
    street["TOT_FLPOP_CO"] = pd.to_numeric(street["TOT_FLPOP_CO"])
    street["TMZON_00_06_FLPOP_CO"] = pd.to_numeric(street["TMZON_00_06_FLPOP_CO"])
    street["night_flpop_ratio"] = street["TMZON_00_06_FLPOP_CO"] / street["TOT_FLPOP_CO"].replace(0, np.nan)
    feat = feat.join(street.set_index("TRDAR_CD")[["TOT_FLPOP_CO", "night_flpop_ratio"]], how="left")
    feat["TOT_FLPOP_CO"] = feat["TOT_FLPOP_CO"].fillna(0)
    feat["night_flpop_ratio"] = feat["night_flpop_ratio"].fillna(0)

    # ---- 05/06: 상주인구·직장인구 (최신 분기 스냅샷, 기준시점 고정값 — 기준서 9절) ----
    resident = _latest_quarter(load_resident_pop())
    resident = resident[resident["TRDAR_CD"].isin(trdar_codes)].copy()
    resident["TOT_REPOP_CO"] = pd.to_numeric(resident["TOT_REPOP_CO"])
    feat = feat.join(resident.set_index("TRDAR_CD")[["TOT_REPOP_CO"]], how="left")

    worker = _latest_quarter(load_worker_pop())
    worker = worker[worker["TRDAR_CD"].isin(trdar_codes)].copy()
    worker["TOT_WRC_POPLTN_CO"] = pd.to_numeric(worker["TOT_WRC_POPLTN_CO"])
    feat = feat.join(worker.set_index("TRDAR_CD")[["TOT_WRC_POPLTN_CO"]], how="left")

    feat["TOT_REPOP_CO"] = feat["TOT_REPOP_CO"].fillna(0)
    feat["TOT_WRC_POPLTN_CO"] = feat["TOT_WRC_POPLTN_CO"].fillna(0)
    feat["population_total"] = feat["TOT_REPOP_CO"] + feat["TOT_WRC_POPLTN_CO"]
    # 면적당(ha) 지표 — 상권 면적이 최대 899배 차이나 절대량만 쓰면 큰 상권이 유리함(기준서 10절).
    feat["population_total_ha"] = feat["population_total"] / feat["면적_ha"].replace(0, np.nan)

    # ---- 07: 집객시설 (최신 분기) ----
    fac = _latest_quarter(load_facilities())
    fac = fac[fac["TRDAR_CD"].isin(trdar_codes)].copy()
    fac["VIATR_FCLTY_CO"] = pd.to_numeric(fac["VIATR_FCLTY_CO"])
    feat = feat.join(fac.set_index("TRDAR_CD")[["VIATR_FCLTY_CO"]], how="left")
    feat["VIATR_FCLTY_CO"] = feat["VIATR_FCLTY_CO"].fillna(0)

    # ---- 공간 조인: 아파트 세대수, 지하철, 버스, 기존 편의점(14번) ----
    # 14번·12/13/18번은 강남·송파만 있는 데이터라, 이번에 포함되는 서초구 상권(13-2)은
    # 이 구간이 과소집계될 수 있다 — 화면에 캐비어트로 표시한다(main() 참고).
    apt_basic = load_apt_basic()[["kaptCode", "kaptdaCnt"]].copy()
    apt_basic["kaptdaCnt"] = pd.to_numeric(apt_basic["kaptdaCnt"], errors="coerce").fillna(0)
    apt_geo = load_apt_geo()
    apt_geo = apt_geo[apt_geo["status"] == "OK"][["kaptCode", "x", "y"]].copy()
    apt_geo["x"] = pd.to_numeric(apt_geo["x"])
    apt_geo["y"] = pd.to_numeric(apt_geo["y"])
    apt = apt_geo.merge(apt_basic, on="kaptCode", how="left")
    apt_x5181, apt_y5181 = wgs84_arr_to_5181(apt["x"].values, apt["y"].values)

    subway = load_subway()
    subway["LAT"] = pd.to_numeric(subway["LAT"])
    subway["LOT"] = pd.to_numeric(subway["LOT"])
    subway_x5181, subway_y5181 = wgs84_arr_to_5181(subway["LOT"].values, subway["LAT"].values)

    bus = load_bus_stops()
    bus["XCRD"] = pd.to_numeric(bus["XCRD"])
    bus["YCRD"] = pd.to_numeric(bus["YCRD"])
    bus_x5181, bus_y5181 = wgs84_arr_to_5181(bus["XCRD"].values, bus["YCRD"].values)

    cvs14 = load_convenience_stores()
    cvs14["lon"] = pd.to_numeric(cvs14["lon"])
    cvs14["lat"] = pd.to_numeric(cvs14["lat"])
    cvs14_x5181, cvs14_y5181 = wgs84_arr_to_5181(cvs14["lon"].values, cvs14["lat"].values)

    apt_hh, subway_cnt, bus_cnt, cvs_cnt = [], [], [], []
    for cx, cy in zip(feat["XCNTS_VALU"], feat["YDNTS_VALU"]):
        apt_hh.append(count_within_radius(cx, cy, apt_x5181, apt_y5181, APT_RADIUS_M, weights=apt["kaptdaCnt"].values))
        subway_cnt.append(count_within_radius(cx, cy, subway_x5181, subway_y5181, SUBWAY_RADIUS_M))
        bus_cnt.append(count_within_radius(cx, cy, bus_x5181, bus_y5181, BUS_RADIUS_M))
        cvs_cnt.append(count_within_radius(cx, cy, cvs14_x5181, cvs14_y5181, CVS_RADIUS_M))

    feat["apt_households"] = apt_hh
    feat["subway_count"] = subway_cnt
    feat["bus_count"] = bus_cnt
    feat["competition_density"] = cvs_cnt
    feat["accessibility"] = feat["subway_count"] * 2 + feat["bus_count"]

    # ---- 인건비 커버리지 (페르소나 A용, 점포 1개 기준, 월 단위) ----
    opex = load_opex_baseline()
    hourly_wage = float(opex[(opex["항목"] == "최저임금_시간급") & (opex["적용연도"] == "2026")]["값"].iloc[0])
    night_hours_per_month = 6 * 30
    night_labor_cost_month = hourly_wage * night_hours_per_month
    feat["night_labor_coverage"] = feat["night_amt_per_store"] / night_labor_cost_month

    # ---- 트랙 분리 (기준서 8절) ----
    # 실측: 01번 매출 행 존재. 매출불명: 매출은 없지만 02번 점포수 또는 14번 반경 내 편의점 존재.
    # 후보제외: 편의점 존재 증거가 전혀 없음.
    has_sales = (feat["cvs_sales_avg"] >= MIN_MEASURED_SALES_MONTH) & (feat["매출분기수"] >= MIN_MEASURED_QUARTERS)
    has_store_evidence = (feat["cvs_store_avg"] > 0) | (feat["competition_density"] > 0)
    feat["트랙"] = np.select([has_sales, has_store_evidence], ["실측", "매출불명"], default="후보제외")

    # 매출불명/후보제외 트랙은 매출 관련 지표를 결측으로 둔다 — 숫자를 지어내지 않는다.
    sales_cols = ["sales_per_store", "night_amt_per_store", "night_sales_ratio", "daytime_sales_ratio", "night_labor_coverage"]
    non_measured = feat["트랙"] != "실측"
    feat.loc[non_measured, sales_cols] = np.nan

    # 이상값 표시 (삭제하지 않고 표시만 — 기준서 11절). 실측 트랙 안에서만 계산.
    feat["이상값_점포당매출"] = ""
    measured = feat["트랙"] == "실측"
    feat.loc[measured, "이상값_점포당매출"] = iqr_outlier_flag(feat.loc[measured, "sales_per_store"])

    return feat


def score_candidates(feat: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """실측 트랙만 대상으로 백분위 순위 정규화 후 가중합 점수를 매긴다(기준서 9절).

    가중치에 들어가는 열 중 하나라도 결측이면 그 상권은 점수를 매기지 않는다(합이 NaN이 되어 자동 제외).
    """
    scored = feat[feat["트랙"] == "실측"].copy()
    total = pd.Series(0.0, index=scored.index)
    for col, w in weights.items():
        invert = w < 0
        norm = percentile_normalize(scored[col], invert=invert)
        scored[f"{col}_norm"] = norm
        total += abs(w) * norm
    scored["score"] = total
    scored = scored.dropna(subset=["score"])
    return scored.sort_values("score", ascending=False)


def reference_table(feat: pd.DataFrame) -> pd.DataFrame:
    """매출불명 트랙 참고표 — 매출과 무관한 수요 지표만. 점수는 매기지 않는다(기준서 8절)."""
    ref = feat[feat["트랙"] == "매출불명"].copy()
    for col in ["population_total", "apt_households", "competition_density", "accessibility"]:
        ref[f"{col}_norm"] = percentile_normalize(ref[col], invert=(col == "competition_density"))
    return ref.sort_values("population_total", ascending=False)


def plot_score_map(boundary_df, shapes, scored: pd.DataFrame, title: str, out_name: str, highlight_codes=None):
    code_to_idx = {code: i for i, code in enumerate(boundary_df["TRDAR_CD"])}
    fig, ax = plt.subplots(figsize=(8, 9))

    vmin, vmax = scored["score"].min(), scored["score"].max()
    cmap = plt.get_cmap("YlGnBu")

    patches = []
    colors = []
    for trdar_cd, row in scored.iterrows():
        idx = code_to_idx.get(trdar_cd)
        if idx is None:
            continue
        shp = shapes[idx]
        parts = list(shp.parts) + [len(shp.points)]
        for p in range(len(parts) - 1):
            pts = shp.points[parts[p]:parts[p + 1]]
            patches.append(MplPolygon(pts, closed=True))
            colors.append(row["score"])

    pc = PatchCollection(patches, cmap=cmap, edgecolor="#333333", linewidth=0.3)
    pc.set_array(np.array(colors))
    pc.set_clim(vmin, vmax)
    ax.add_collection(pc)

    if highlight_codes:
        for trdar_cd in highlight_codes:
            idx = code_to_idx.get(trdar_cd)
            if idx is None:
                continue
            shp = shapes[idx]
            parts = list(shp.parts) + [len(shp.points)]
            for p in range(len(parts) - 1):
                pts = shp.points[parts[p]:parts[p + 1]]
                ax.add_patch(MplPolygon(pts, closed=True, fill=False, edgecolor="red", linewidth=2))

    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(pc, ax=ax, label="추천 점수 (백분위 가중합, 상대비교)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / out_name, dpi=150)
    plt.close(fig)


def franchise_fit(budget_manwon: float) -> pd.DataFrame:
    """예산 매칭 — 1호점 필요현금 기준만 본다(기준서 13-6 결정: 2호점 예산 검증은 하지 않음)."""
    fr = load_franchise_terms()
    for c in ["개점투자_소계_만원", "담보_만원", "경영주_최종배분율", "최저수입보장_월_만원"]:
        fr[c] = pd.to_numeric(fr[c], errors="coerce").fillna(0)
    fr["필요현금_추정_만원"] = fr["개점투자_소계_만원"] + fr["담보_만원"]
    feasible = fr[fr["필요현금_추정_만원"] <= budget_manwon].copy()
    return feasible.sort_values(["경영주_최종배분율", "최저수입보장_월_만원"], ascending=False)


def format_report_table(df: pd.DataFrame, n: int) -> list[str]:
    top = df.head(n)
    lines = [
        "| 순위 | 상권명 | 구/행정동 | 상권구분 | 점수 | 점포당 추정매출(월) | 이상값 | 심야매출비중 | 심야인건비대비배율 | 전체점포수(SIMILR_INDUTY_STOR_CO) | 반경300m 내 편의점 | 상주+직장인구 | 아파트세대수(반경400m) |",
        "|---:|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, (trdar_cd, row) in enumerate(top.iterrows(), start=1):
        lines.append(
            f"| {rank} | {row['상권명']} | {row['구']}/{row['행정동']} | {row['상권구분']} | {row['score']:.3f} "
            f"| {row['sales_per_store']:,.0f}원 | {row['이상값_점포당매출'] or '-'} | {row['night_sales_ratio']:.1%} | {row['night_labor_coverage']:.2f}배 "
            f"| {row['cvs_store_avg']:.1f}곳 | {row['competition_density']:.0f}곳 | {row['population_total']:,.0f}명 | {row['apt_households']:,.0f}세대 |"
        )
    return lines


def format_reference_table(df: pd.DataFrame, n: int) -> list[str]:
    top = df.head(n)
    lines = [
        "| 상권명 | 구/행정동 | 상권구분 | 상주+직장인구 | 아파트세대수(반경400m) | 반경300m 내 편의점 | 접근성(지하철×2+버스) |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for _, row in top.iterrows():
        lines.append(
            f"| {row['상권명']} | {row['구']}/{row['행정동']} | {row['상권구분']} "
            f"| {row['population_total']:,.0f}명 | {row['apt_households']:,.0f}세대 | {row['competition_density']:.0f}곳 | {row['accessibility']:.0f} |"
        )
    return lines


def main() -> None:
    boundary_df, shapes = load_trade_area_boundary()
    gs_codes = gangnam_songpa_trdar_codes()

    # 도곡동 중심점 (도곡동 소재 상권은 강남구라 gs_codes 안에 있음)
    dogok = boundary_df[boundary_df["ADSTRD_CD_"].str.contains("도곡", na=False)]
    ref_x, ref_y = dogok["XCNTS_VALU"].astype(float).mean(), dogok["YDNTS_VALU"].astype(float).mean()

    # 기준서 13-2: 서초구 상권 포함. 자치구 무관하게 도곡동 반경 1.5km 내 전체 상권을 후보군에 넣는다.
    all_df = boundary_df.copy()
    all_df["XCNTS_VALU"] = all_df["XCNTS_VALU"].astype(float)
    all_df["YDNTS_VALU"] = all_df["YDNTS_VALU"].astype(float)
    all_df["dist_to_dogok_m"] = np.sqrt((all_df["XCNTS_VALU"] - ref_x) ** 2 + (all_df["YDNTS_VALU"] - ref_y) ** 2)
    nearby_codes = set(all_df.loc[all_df["dist_to_dogok_m"] <= B_NEARBY_RADIUS_M, "TRDAR_CD"])

    candidate_codes = gs_codes | nearby_codes

    print("[recommend] building feature table ...")
    feat = build_feature_table(candidate_codes, boundary_df)
    feat["dist_to_dogok_m"] = np.sqrt((feat["XCNTS_VALU"] - ref_x) ** 2 + (feat["YDNTS_VALU"] - ref_y) ** 2)
    feat.to_csv(OUT_DIR / "feature_table_gangnam_songpa.csv", encoding="utf-8-sig")

    track_counts = feat.loc[feat.index.isin(gs_codes), "트랙"].value_counts()

    # ---- 페르소나 A: 강남+송파 전체, 실측 트랙만 점수화 ----
    pool_a = feat[feat.index.isin(gs_codes)]
    scored_a = score_candidates(pool_a, WEIGHTS_A)
    ref_a = reference_table(pool_a)
    plot_score_map(boundary_df, shapes, scored_a, "페르소나 A 추천 점수 (강남·송파, 실측 트랙)", "05_recommend_A_map.png",
                    highlight_codes=list(scored_a.head(5).index))
    print("[recommend] persona A scored & mapped")

    # ---- 페르소나 B: 도곡동 반경 1.5km, 자치구 무관(서초구 포함), 실측 트랙만 점수화 ----
    pool_b = feat[feat["dist_to_dogok_m"] <= B_NEARBY_RADIUS_M]
    scored_b = score_candidates(pool_b, WEIGHTS_B)
    ref_b = reference_table(pool_b)
    seocho_in_b = int((pool_b["구"] == "서초구").sum())
    plot_score_map(boundary_df, shapes, scored_b, "페르소나 B 추천 점수 (도곡동 반경 1.5km, 서초구 포함)", "06_recommend_B_map.png",
                    highlight_codes=list(scored_b.head(5).index))
    print("[recommend] persona B scored & mapped")

    # ---- 프랜차이즈 예산 적합성 (1호점 기준만 — 기준서 13-6) ----
    fr_a = franchise_fit(5000)
    fr_b = franchise_fit(25000)

    lines = ["# 페르소나별 편의점 입지 추천 리포트 (전처리_기준서.md v3 반영)", ""]

    lines.append("## 트랙 분리 현황 (강남·송파 173개 기준)")
    lines.append("")
    lines.append(f"- 실측: {track_counts.get('실측', 0)}개 / 매출불명: {track_counts.get('매출불명', 0)}개 / 후보제외: {track_counts.get('후보제외', 0)}개")
    lines.append("- 점수 계산은 실측 트랙만 대상으로 한다. 매출불명 트랙은 매출과 무관한 수요 지표만 별도 표로 보여주고 순위에는 넣지 않는다(기준서 8절).")
    lines.append("")

    lines.append("## 페르소나 A: 29세 청년창업 (자본 2천~5천만원, 강남·송파 전역, 인건비/심야매출 중요)")
    lines.append("")
    lines.append(f"- 점수화 대상(실측 트랙): {len(scored_a)}개 / 매출불명 참고: {len(ref_a)}개")
    lines.append("")
    lines.extend(format_report_table(scored_a, 10))
    lines.append("")
    lines.append("### 매출불명 참고 (점수 없음, 수요 지표만 — 상위 5)")
    lines.append("")
    lines.extend(format_reference_table(ref_a, 5))
    lines.append("")
    lines.append("### 예산(5,000만원) 내 프랜차이즈 옵션 (경영주 최종배분율순)")
    lines.append("")
    lines.append("| 브랜드 | 가맹타입 | 운영시간 | 필요현금 추정(만원) | 경영주 최종배분율 | 최저수입보장(월,만원) |")
    lines.append("|---|---|---|---:|---:|---:|")
    for _, r in fr_a.head(8).iterrows():
        lines.append(f"| {r['브랜드']} | {r['가맹타입']} | {r['운영시간']} | {r['필요현금_추정_만원']:,.0f} | {r['경영주_최종배분율']:.0f}% | {r['최저수입보장_월_만원']:,.0f} |")

    lines.append("")
    lines.append("## 페르소나 B: 59세 은퇴부부 (자본 2.5억, 강남구 도곡동 반경 1.5km, 확장 고려)")
    lines.append("")
    lines.append(
        f"- 후보 범위: 도곡동 소재 상권들의 중심점 평균을 기준점으로 잡고, EPSG:5181 평면거리로 "
        f"반경 {B_NEARBY_RADIUS_M}m 이내 상권을 전부 포함했다(자치구 무관, 기준서 13-2). "
        f"점수화 대상(실측 트랙): {len(scored_b)}개 / 매출불명 참고: {len(ref_b)}개 / 이 중 서초구 상권: {seocho_in_b}개"
    )
    lines.append(
        "- 주의: 아파트 세대수(12·13·18번)·경쟁밀도(14번 상가정보)는 강남·송파에만 있는 원천이라, "
        "이번에 포함된 서초구 상권은 이 두 지표가 과소집계될 수 있다."
    )
    lines.append("")
    lines.extend(format_report_table(scored_b, min(5, len(scored_b))))
    lines.append("")
    lines.append("### 매출불명 참고 (점수 없음, 수요 지표만)")
    lines.append("")
    lines.extend(format_reference_table(ref_b, 5))
    lines.append("")
    lines.append("### 예산(2.5억원) 내 프랜차이즈 옵션 (경영주 최종배분율순, 상위 8개)")
    lines.append("")
    lines.append(
        "**참고(기준서 13-6)**: 아래는 1호점 기준 예산 매칭만이다. B는 2호점 확장을 고려하지만, "
        "1호점 필요현금 × 2가 가용 예산(2.5억) 이내인지는 이번 전처리에서 검증하지 않기로 팀에서 정했다."
    )
    lines.append("")
    lines.append("| 브랜드 | 가맹타입 | 운영시간 | 필요현금 추정(만원) | 경영주 최종배분율 | 최저수입보장(월,만원) |")
    lines.append("|---|---|---|---:|---:|---:|")
    for _, r in fr_b.head(8).iterrows():
        lines.append(f"| {r['브랜드']} | {r['가맹타입']} | {r['운영시간']} | {r['필요현금_추정_만원']:,.0f} | {r['경영주_최종배분율']:.0f}% | {r['최저수입보장_월_만원']:,.0f} |")

    lines.append("")
    lines.append("## 가중치 및 가정")
    lines.append("")
    lines.append(f"- 페르소나 A 가중치: {WEIGHTS_A}")
    lines.append(f"- 페르소나 B 가중치: {WEIGHTS_B}")
    lines.append(
        "- **A·B 점수는 서로 비교할 수 없다(기준서 13-5)**: 모수가 다르고(A는 강남·송파 실측 상권, "
        "B는 도곡동 반경 내 실측 상권), 가중치도 다르다. 같은 상권이 A·B에서 다른 점수를 받는 것이 정상이다."
    )
    lines.append(f"- 반경: 아파트 {APT_RADIUS_M}m, 지하철 {SUBWAY_RADIUS_M}m(기준서 13-4로 600m에서 변경), 버스 {BUS_RADIUS_M}m, 경쟁점포 {CVS_RADIUS_M}m, B 인근범위 {B_NEARBY_RADIUS_M}m")
    lines.append("- 정규화: 백분위 순위(rank(pct=True), 기준서 9절). 가중치에 들어가는 지표 중 하나라도 결측이면 점수를 매기지 않는다.")
    lines.append("- 프랜차이즈 필요현금 = 개점투자_소계_만원 + 담보_만원 (보수적 추정, 실제로는 대출/보증보험으로 담보 일부 대체 가능)")
    lines.append("- 심야 인건비 = 2026년 최저시급 x 6시간(22시~04시 가정) x 30일")
    lines.append("- 주간 시간대 = 06~17시(TMZON_06_11+11_14+14_17), 심야 = 21~06시 (기준서 2절)")
    lines.append("- 담배소매업 이격거리 데이터는 이번 원천에 없어 반영하지 않았다(기준서 13-3).")

    (OUT_DIR / "recommend_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("[recommend] report written ->", OUT_DIR / "recommend_report.md")


if __name__ == "__main__":
    main()
