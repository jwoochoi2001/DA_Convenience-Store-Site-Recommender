# -*- coding: utf-8 -*-
"""상권 확장 분석: 현재 가진 데이터로 상권 단위에서 가공할 수 있는 지표를 모두 붙인다.

입력: eda_output/feature_table_gangnam_songpa.csv (recommend.py가 만든 기본 피처, 강남·송파 173 + 도곡 반경 서초 11 = 184개 상권)
출력: eda_output/analysis/feature_extended.csv (기본 피처 + 확장 지표 + 임대료·공실 + 군집)

확장 지표의 출처
- 01 추정매출: 시간대·요일·성별·연령대 매출 구성, 객단가, 점포당 거래건수, 전년 대비 성장, 분기 변동성 (실측 트랙만)
- 02 점포: 개업률·폐업률, 점포 수 증감, 프랜차이즈 비중
- 03 상권변화지표: 운영·폐업 영업개월 평균
- 04/05/06 인구: 20·30대 비중, 주말·저녁 비중, 상주 60+ 비중, 가구 수
- 07 집객시설: 대학·병원·학교·약국·은행 등
- 08 생활인구(행정동): 심야 생활인구 비중
- 14 상가정보(반경 300m): 음식점·주점·숙박·교육·의료·사무 업소 수
- 15/17 R-ONE: 1층 임대료·공실률 (상권→R-ONE 권역 매핑은 아래 규칙, 가정)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data_paths import (
    ROOT,
    filter_convenience,
    load_change_index,
    load_facilities,
    load_living_pop,
    load_rent,
    load_resident_pop,
    load_sales,
    load_stores,
    load_stores_detail,
    load_street_pop,
    load_vacancy,
    load_worker_pop,
)
from geo_utils import count_within_radius, wgs84_arr_to_5181

OUT = ROOT / "eda_output" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

STORE_RADIUS_M = 300  # 14번 상가 업종 집계 반경(경쟁 편의점 반경과 동일)


def _q_sorted(df: pd.DataFrame) -> list[str]:
    return sorted(df["STDR_YYQU_CD"].unique())


def _num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ------------------------------------------------------------------ 01 매출 구성
def sales_profile(codes: set[str]) -> pd.DataFrame:
    s = filter_convenience(load_sales())
    s = s[s["TRDAR_CD"].isin(codes)].copy()
    amt_cols = ["THSMON_SELNG_AMT", "THSMON_SELNG_CO", "MDWK_SELNG_AMT", "WKEND_SELNG_AMT",
                "FRI_SELNG_AMT", "SAT_SELNG_AMT",
                "TMZON_00_06_SELNG_AMT", "TMZON_06_11_SELNG_AMT", "TMZON_11_14_SELNG_AMT",
                "TMZON_14_17_SELNG_AMT", "TMZON_17_21_SELNG_AMT", "TMZON_21_24_SELNG_AMT",
                "ML_SELNG_AMT", "FML_SELNG_AMT",
                "AGRDE_10_SELNG_AMT", "AGRDE_20_SELNG_AMT", "AGRDE_30_SELNG_AMT",
                "AGRDE_40_SELNG_AMT", "AGRDE_50_SELNG_AMT", "AGRDE_60_ABOVE_SELNG_AMT"]
    s = _num(s, amt_cols)
    qs = _q_sorted(s)
    last4, prev4 = qs[-4:], qs[-8:-4]

    cur = s[s["STDR_YYQU_CD"].isin(last4)]
    g = cur.groupby("TRDAR_CD")[amt_cols].sum()
    tot = g["THSMON_SELNG_AMT"].where(g["THSMON_SELNG_AMT"] > 0)
    out = pd.DataFrame(index=g.index)
    out["weekend_ratio"] = g["WKEND_SELNG_AMT"] / (g["MDWK_SELNG_AMT"] + g["WKEND_SELNG_AMT"]).where(lambda x: x > 0)
    out["friSat_ratio"] = (g["FRI_SELNG_AMT"] + g["SAT_SELNG_AMT"]) / tot
    out["h00_06_ratio"] = g["TMZON_00_06_SELNG_AMT"] / tot
    out["h06_11_ratio"] = g["TMZON_06_11_SELNG_AMT"] / tot
    out["h11_14_ratio"] = g["TMZON_11_14_SELNG_AMT"] / tot
    out["h14_17_ratio"] = g["TMZON_14_17_SELNG_AMT"] / tot
    out["h17_21_ratio"] = g["TMZON_17_21_SELNG_AMT"] / tot
    out["h21_24_ratio"] = g["TMZON_21_24_SELNG_AMT"] / tot
    out["male_sales_ratio"] = g["ML_SELNG_AMT"] / (g["ML_SELNG_AMT"] + g["FML_SELNG_AMT"]).where(lambda x: x > 0)
    age_tot = g[[c for c in amt_cols if c.startswith("AGRDE")]].sum(axis=1).where(lambda x: x > 0)
    out["age10_ratio"] = g["AGRDE_10_SELNG_AMT"] / age_tot
    out["age20_ratio"] = g["AGRDE_20_SELNG_AMT"] / age_tot
    out["age30_ratio"] = g["AGRDE_30_SELNG_AMT"] / age_tot
    out["age40_ratio"] = g["AGRDE_40_SELNG_AMT"] / age_tot
    out["age50_ratio"] = g["AGRDE_50_SELNG_AMT"] / age_tot
    out["age60p_ratio"] = g["AGRDE_60_ABOVE_SELNG_AMT"] / age_tot
    out["avg_ticket_won"] = g["THSMON_SELNG_AMT"] / g["THSMON_SELNG_CO"].where(g["THSMON_SELNG_CO"] > 0)
    n_q = cur.groupby("TRDAR_CD")["STDR_YYQU_CD"].nunique()
    out["tx_per_month"] = g["THSMON_SELNG_CO"] / n_q / 3  # 월 거래건수(상권 전체)

    # 분기 변동성(최근 4분기 분기매출의 변동계수)
    qamt = cur.groupby(["TRDAR_CD", "STDR_YYQU_CD"])["THSMON_SELNG_AMT"].sum().unstack()
    out["sales_cv"] = qamt.std(axis=1, ddof=0) / qamt.mean(axis=1)

    # 전년 대비 성장: 최근 4분기 평균 vs 직전 4분기 평균 (양쪽 모두 4분기 있어야 계산)
    prev = s[s["STDR_YYQU_CD"].isin(prev4)].groupby(["TRDAR_CD", "STDR_YYQU_CD"])["THSMON_SELNG_AMT"].sum().unstack()
    both = (qamt.notna().sum(axis=1) == 4) & (prev.reindex(qamt.index).notna().sum(axis=1) == 4)
    yoy = qamt.mean(axis=1) / prev.reindex(qamt.index).mean(axis=1) - 1
    out["sales_yoy"] = yoy.where(both)
    return out


# ------------------------------------------------------------------ 02 점포 동태
def store_dynamics(codes: set[str]) -> pd.DataFrame:
    s = filter_convenience(load_stores())
    s = s[s["TRDAR_CD"].isin(codes)].copy()
    s = _num(s, ["SIMILR_INDUTY_STOR_CO", "STOR_CO", "FRC_STOR_CO", "OPBIZ_RT", "CLSBIZ_RT",
                 "OPBIZ_STOR_CO", "CLSBIZ_STOR_CO"])
    qs = _q_sorted(s)
    last4, prev4 = qs[-4:], qs[-8:-4]
    cur = s[s["STDR_YYQU_CD"].isin(last4)]
    out = cur.groupby("TRDAR_CD").agg(
        open_rate=("OPBIZ_RT", "mean"),
        close_rate=("CLSBIZ_RT", "mean"),
        close_cnt_4q=("CLSBIZ_STOR_CO", "sum"),
        open_cnt_4q=("OPBIZ_STOR_CO", "sum"),
        frc=("FRC_STOR_CO", "mean"),
        allst=("SIMILR_INDUTY_STOR_CO", "mean"),
    )
    out["franchise_ratio"] = out["frc"] / out["allst"].where(out["allst"] > 0)
    prev = s[s["STDR_YYQU_CD"].isin(prev4)].groupby("TRDAR_CD")["SIMILR_INDUTY_STOR_CO"].mean()
    out["store_growth"] = (out["allst"] - prev.reindex(out.index)) / prev.reindex(out.index).where(lambda x: x > 0)
    return out.drop(columns=["frc", "allst"])


# ------------------------------------------------------------------ 03 상권변화지표
def change_months(codes: set[str]) -> pd.DataFrame:
    c = load_change_index()
    c = c[c["TRDAR_CD"].isin(codes)].copy()
    c = _num(c, ["OPR_SALE_MT_AVRG", "CLS_SALE_MT_AVRG"])
    latest = c["STDR_YYQU_CD"].max()
    c = c[c["STDR_YYQU_CD"] == latest].set_index("TRDAR_CD")
    return c[["OPR_SALE_MT_AVRG", "CLS_SALE_MT_AVRG"]].rename(
        columns={"OPR_SALE_MT_AVRG": "opr_months", "CLS_SALE_MT_AVRG": "cls_months"})


# ------------------------------------------------------------------ 04/05/06 인구 구성
def population_profile(codes: set[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=sorted(codes))

    st = load_street_pop()
    st = st[st["TRDAR_CD"].isin(codes)]
    st = st[st["STDR_YYQU_CD"] == st["STDR_YYQU_CD"].max()].copy()
    st = _num(st, ["TOT_FLPOP_CO", "AGRDE_20_FLPOP_CO", "AGRDE_30_FLPOP_CO", "SAT_FLPOP_CO", "SUN_FLPOP_CO",
                   "TMZON_17_21_FLPOP_CO", "TMZON_21_24_FLPOP_CO", "TMZON_00_06_FLPOP_CO"])
    st = st.set_index("TRDAR_CD")
    tot = st["TOT_FLPOP_CO"].where(st["TOT_FLPOP_CO"] > 0)
    out["fl_age2030_ratio"] = ((st["AGRDE_20_FLPOP_CO"] + st["AGRDE_30_FLPOP_CO"]) / tot).reindex(out.index)
    # (토+일)/전체 유동인구. 요일 균등이면 2/7≈0.286이라 이보다 크면 주말 쏠림.
    out["fl_weekend_ratio"] = ((st["SAT_FLPOP_CO"] + st["SUN_FLPOP_CO"]) / tot).reindex(out.index)
    out["fl_evening_ratio"] = ((st["TMZON_17_21_FLPOP_CO"] + st["TMZON_21_24_FLPOP_CO"]) / tot).reindex(out.index)

    rp = load_resident_pop()
    rp = rp[rp["TRDAR_CD"].isin(codes)]
    rp = rp[rp["STDR_YYQU_CD"] == rp["STDR_YYQU_CD"].max()].copy()
    rp = _num(rp, ["TOT_REPOP_CO", "AGRDE_20_REPOP_CO", "AGRDE_30_REPOP_CO", "AGRDE_60_ABOVE_REPOP_CO", "TOT_HSHLD_CO"])
    rp = rp.set_index("TRDAR_CD")
    rt = rp["TOT_REPOP_CO"].where(rp["TOT_REPOP_CO"] > 0)
    out["res_age2030_ratio"] = ((rp["AGRDE_20_REPOP_CO"] + rp["AGRDE_30_REPOP_CO"]) / rt).reindex(out.index)
    out["res_age60p_ratio"] = (rp["AGRDE_60_ABOVE_REPOP_CO"] / rt).reindex(out.index)
    out["households"] = rp["TOT_HSHLD_CO"].reindex(out.index)

    wp = load_worker_pop()
    wp = wp[wp["TRDAR_CD"].isin(codes)]
    wp = wp[wp["STDR_YYQU_CD"] == wp["STDR_YYQU_CD"].max()].copy()
    wp = _num(wp, ["TOT_WRC_POPLTN_CO", "AGRDE_20_WRC_POPLTN_CO", "AGRDE_30_WRC_POPLTN_CO"])
    wp = wp.set_index("TRDAR_CD")
    wt = wp["TOT_WRC_POPLTN_CO"].where(wp["TOT_WRC_POPLTN_CO"] > 0)
    out["wrk_age2030_ratio"] = ((wp["AGRDE_20_WRC_POPLTN_CO"] + wp["AGRDE_30_WRC_POPLTN_CO"]) / wt).reindex(out.index)
    return out


# ------------------------------------------------------------------ 07 집객시설
def facility_profile(codes: set[str]) -> pd.DataFrame:
    f = load_facilities()
    f = f[f["TRDAR_CD"].isin(codes)]
    f = f[f["STDR_YYQU_CD"] == f["STDR_YYQU_CD"].max()].copy()
    cols = ["UNIV_CO", "GNRL_HSPTL_CO", "GEHSPT_CO", "PARMACY_CO", "BANK_CO", "KNDRGR_CO", "ELESCH_CO",
            "MSKUL_CO", "HGSCHL_CO", "SUPMK_CO", "THEAT_CO", "STAYNG_FCLTY_CO", "PBLOFC_CO"]
    f = _num(f, cols).set_index("TRDAR_CD")
    out = pd.DataFrame(index=f.index)
    out["fac_university"] = f["UNIV_CO"]
    out["fac_hospital"] = f["GNRL_HSPTL_CO"] + f["GEHSPT_CO"]
    out["fac_pharmacy"] = f["PARMACY_CO"]
    out["fac_bank"] = f["BANK_CO"]
    out["fac_school"] = f["KNDRGR_CO"] + f["ELESCH_CO"] + f["MSKUL_CO"] + f["HGSCHL_CO"]
    out["fac_supermarket"] = f["SUPMK_CO"]
    out["fac_stay"] = f["STAYNG_FCLTY_CO"]
    out["fac_public"] = f["PBLOFC_CO"]
    return out


# ------------------------------------------------------------------ 08 생활인구(행정동)
def living_pop_night(feat: pd.DataFrame) -> pd.Series:
    """상권의 (구, 행정동) -> 행정동코드(14번 상가정보의 adongCd로 매핑) -> 심야(21~06시) 생활인구 비중.

    14번이 강남·송파만 있어서 서초구 상권은 결측이다.
    """
    sd = load_stores_detail()[["signguNm", "adongNm", "adongCd"]].drop_duplicates()
    key = {(r.signguNm, r.adongNm): r.adongCd for r in sd.itertuples()}
    lv = load_living_pop()
    lv["TMZON_PD_SE"] = pd.to_numeric(lv["TMZON_PD_SE"])
    lv["TOT_LVPOP_CO"] = pd.to_numeric(lv["TOT_LVPOP_CO"])
    night_h = {21, 22, 23, 0, 1, 2, 3, 4, 5}
    lv["night"] = lv["TMZON_PD_SE"].isin(night_h)
    g_all = lv.groupby("ADSTRD_CODE_SE")["TOT_LVPOP_CO"].sum()
    g_night = lv[lv["night"]].groupby("ADSTRD_CODE_SE")["TOT_LVPOP_CO"].sum()
    ratio = (g_night / g_all)
    # 시간대 한 칸당 평균 생활인구(상권이 속한 동의 규모 참고용)
    vals = []
    for _, r in feat.iterrows():
        code = key.get((r["구"], r["행정동"]))
        vals.append(ratio.get(str(code)) if code else np.nan)
    return pd.Series(vals, index=feat.index, name="lv_night_ratio")


# ------------------------------------------------------------------ 14 상가 업종(반경 300m)
def surrounding_business(feat: pd.DataFrame) -> pd.DataFrame:
    sd = load_stores_detail()
    sd["lon"] = pd.to_numeric(sd["lon"], errors="coerce")
    sd["lat"] = pd.to_numeric(sd["lat"], errors="coerce")
    sd = sd.dropna(subset=["lon", "lat"])
    x, y = wgs84_arr_to_5181(sd["lon"].values, sd["lat"].values)
    groups = {
        "biz_total": np.ones(len(sd), dtype=bool),
        "biz_food": (sd["indsLclsNm"] == "음식").values,
        "biz_pub": (sd["indsMclsNm"] == "주점").values,
        "biz_stay": (sd["indsLclsNm"] == "숙박").values,
        "biz_edu": (sd["indsLclsNm"] == "교육").values,
        "biz_clinic": (sd["indsLclsNm"] == "보건의료").values,
        "biz_office": (sd["indsLclsNm"].isin(["과학·기술", "시설관리·임대"])).values,
        "biz_retail": (sd["indsLclsNm"] == "소매").values,
    }
    res = {k: [] for k in groups}
    for cx, cy in zip(feat["XCNTS_VALU"], feat["YDNTS_VALU"]):
        for k, m in groups.items():
            res[k].append(count_within_radius(cx, cy, x[m], y[m], STORE_RADIUS_M))
    return pd.DataFrame(res, index=feat.index)


# ------------------------------------------------------------------ 15/17 임대료·공실 (R-ONE)
# 상권(구/행정동) -> R-ONE 권역 매핑 규칙. R-ONE은 상권 단위가 아니라 대표 상권 몇 개만 있어 대체 매핑이 필요하다(가정).
#  - 강남구 도심 행정동 -> 서울>강남 평균, 남부 외곽(개포·일원·수서·세곡) -> 서울>기타 평균
#  - 송파구 잠실권(잠실·방이·석촌·삼전·송파·신천) -> 잠실/송파, 그 밖(가락·문정·장지·거여·마천·오금·위례) -> 가락시장(송파 최저 관측치)
#  - 서초구 -> 남부터미널(서초 남부 관측치)
RENT_AREAS_GANGNAM_CORE = ("역삼", "논현", "삼성", "대치", "도곡", "청담", "압구정", "신사")
RENT_AREAS_GANGNAM_OUTER = ("개포", "일원", "수서", "세곡", "자곡", "율현")
RENT_AREAS_SONGPA_CORE = ("잠실", "방이", "석촌", "삼전", "송파", "신천")


def rent_area(gu: str, dong: str) -> str:
    dong = str(dong)
    if gu == "강남구":
        if any(k in dong for k in RENT_AREAS_GANGNAM_OUTER):
            return "서울>기타"
        return "서울>강남"
    if gu == "송파구":
        if any(k in dong for k in RENT_AREAS_SONGPA_CORE):
            return "서울>기타>잠실/송파"
        return "서울>기타>가락시장"
    if gu == "서초구":
        return "서울>강남>남부터미널"
    return "서울"


def rent_vacancy_tables() -> tuple[dict, dict, str]:
    """R-ONE 최신 분기의 {CLS_FULLNM: 값} 사전 두 개(임대료 천원/㎡·월, 공실률 %)와 기준 분기 설명."""
    def latest(df: pd.DataFrame) -> tuple[dict, str]:
        df = df.copy()
        df["DTA_VAL"] = pd.to_numeric(df["DTA_VAL"], errors="coerce")
        q = df["WRTTIME_IDTFR_ID"].max()
        d = df[df["WRTTIME_IDTFR_ID"] == q]
        # 임대료 15번은 CLS_FULLNM, 공실 17번도 CLS_FULLNM에 권역명이 있다(파일 확인 결과).
        return dict(zip(d["CLS_FULLNM"], d["DTA_VAL"])), str(d["WRTTIME_DESC"].iloc[0])
    rent, desc = latest(load_rent())
    vac, _ = latest(load_vacancy())
    return rent, vac, desc


def attach_rent(feat: pd.DataFrame) -> pd.DataFrame:
    rent, vac, desc = rent_vacancy_tables()
    areas = [rent_area(g, d) for g, d in zip(feat["구"], feat["행정동"])]
    out = pd.DataFrame(index=feat.index)
    out["rent_area"] = areas
    out["rent_per_m2_kwon"] = [rent.get(a, np.nan) for a in areas]  # 천원/㎡·월 (1층)
    out["vacancy_pct"] = [vac.get(a, np.nan) for a in areas]
    out.attrs["rent_period"] = desc
    return out


def main() -> None:
    feat = pd.read_csv(ROOT / "eda_output" / "feature_table_gangnam_songpa.csv", dtype={"TRDAR_CD": str}).set_index("TRDAR_CD")
    codes = set(feat.index)
    print("[expand] candidates:", len(codes))

    print("[expand] 01 sales profile ...")
    sp = sales_profile(codes)
    measured = feat["트랙"] == "실측"
    sp = sp.reindex(feat.index)
    sp.loc[~measured] = np.nan  # 매출 기반 지표는 실측 트랙에서만(기준서 8절)

    print("[expand] 02/03/04/05/06/07 ...")
    sd = store_dynamics(codes).reindex(feat.index)
    cm = change_months(codes).reindex(feat.index)
    pp = population_profile(codes).reindex(feat.index)
    fp = facility_profile(codes).reindex(feat.index)

    print("[expand] 08 living pop ...")
    lv = living_pop_night(feat)

    print("[expand] 14 surrounding business ...")
    sb = surrounding_business(feat)

    print("[expand] 15/17 rent & vacancy ...")
    rv = attach_rent(feat)

    ext = pd.concat([feat, sp, sd, cm, pp, fp, lv, sb, rv], axis=1)
    ext.attrs = {}
    ext.to_csv(OUT / "feature_extended.csv", encoding="utf-8-sig")
    print("[expand] saved", OUT / "feature_extended.csv", ext.shape)
    print("[expand] rent period:", rv.attrs.get("rent_period"))
    print(ext[["rent_area", "rent_per_m2_kwon", "vacancy_pct"]].drop_duplicates().to_string())


if __name__ == "__main__":
    main()
