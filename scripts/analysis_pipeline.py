# -*- coding: utf-8 -*-
"""팀 합의 분석 흐름: 상권 분석 -> 입지만 고려한 추천 3곳 -> ROI 반영 추천 3곳 -> 순위 변동 -> 시사점.

입력: eda_output/analysis/feature_extended.csv (analysis_expand.py 결과)
출력: eda_output/analysis/results.pkl (보고서 생성용 표 묶음), 각종 CSV, eda_output/analysis/fig_*.png

용어
- 입지 점수: recommend.py의 점수(WEIGHTS_A/WEIGHTS_B). 수요·접근성·경쟁·시간대·상권 안정성과 매출 수준을
  백분위로 반영한다. 임차료·초기 투자금·배분율·인건비 대비 수익은 반영하지 않는다.
- ROI: 연 추정 순이익 / 초기 투자금. 비용을 반영한 금액 기준 지표다.
"""
from __future__ import annotations

import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from data_paths import ROOT, load_franchise_terms, load_opex_baseline, load_trade_area_boundary
from recommend import B_NEARBY_RADIUS_M, WEIGHTS_A, WEIGHTS_B, score_candidates

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["axes.spines.right"] = False

OUT = ROOT / "eda_output" / "analysis"
FIG = OUT

# ---- 색 (모든 그림에서 같은 의미로 쓴다) ----
C_BLUE = "#1f5fbf"
C_ORANGE = "#e07b00"
C_GREEN = "#2a9d8f"
C_GRAY = "#8a93a3"
C_RED = "#c0392b"
C_LIGHT = "#cfd6e2"

# ---- ROI 가정 (보고서 3절에도 그대로 적는다) ----
GM_BASE = 0.25            # 총이익률(매출총이익/매출) 기본 가정
GM_ALT = 0.30             # 민감도
STORE_AREA_M2 = 50        # 점포 임차 면적(약 15평) 가정
DEPOSIT_MONTHS = 10       # 임차 보증금 = 월세 N개월분 가정
NIGHT_HOURS = 6           # 야간 알바 1명, 하루 6시간
DAYS = 30
SMALL_N = 4.0             # 소표본 표시 기준: 상권 평균 편의점 수 < 4


def wage_night_manwon() -> float:
    opex = load_opex_baseline()
    hourly = float(opex[(opex["항목"] == "최저임금_시간급") & (opex["적용연도"] == "2026")]["값"].iloc[0])
    return hourly * NIGHT_HOURS * DAYS / 1e4


def pick_franchise(brand: str, ftype: str, hours: str) -> dict:
    fr = load_franchise_terms()
    r = fr[(fr["브랜드"] == brand) & (fr["가맹타입"] == ftype) & (fr["운영시간"] == hours)].iloc[0]
    cash = float(r["개점투자_소계_만원"]) + float(r["담보_만원"])
    dep = pd.to_numeric(r["본부보증금_최소_만원"], errors="coerce")
    return {
        "brand": brand, "type": ftype, "hours": hours,
        "share": float(r["경영주_최종배분율"]) / 100.0,
        "cash": cash,
        "hq_deposit": 0.0 if pd.isna(dep) else float(dep),
        "lease": r["점포임차_부담"],
        "interior": r["인테리어_집기_부담"],
    }


# ============================================================ 1. 상권 분석
def stage1(e: pd.DataFrame) -> dict:
    R: dict = {}
    gs = e[e["구"].isin(["강남구", "송파구"])]
    meas = e[e["트랙"] == "실측"]
    meas_a = gs[gs["트랙"] == "실측"]

    R["overview_track_gu"] = e.groupby(["구", "트랙"]).size().unstack(fill_value=0)
    R["overview_track_type"] = gs.groupby(["상권구분", "트랙"]).size().unstack(fill_value=0)

    # --- 시장 규모·효율 (실측 A 풀 90곳)
    def agg_type(df, cols):
        g = df.groupby("상권구분")
        out = g[cols].median()
        out["n"] = g.size()
        return out

    meas_a = meas_a.copy()
    meas_a["sales_per_store_man"] = meas_a["sales_per_store"] / 1e4
    meas_a["night_amt_per_store_man"] = meas_a["night_amt_per_store"] / 1e4
    R["market_by_type"] = agg_type(meas_a, ["sales_per_store_man", "night_amt_per_store_man", "cvs_store_avg",
                                            "avg_ticket_won", "tx_per_month"])
    R["market_by_gu"] = meas_a.groupby("구")[["sales_per_store_man", "night_amt_per_store_man", "cvs_store_avg"]].median().join(
        meas_a.groupby("구").size().rename("n"))

    # --- 시간대·요일·연령 (실측 평균)
    tcols = ["h00_06_ratio", "h06_11_ratio", "h11_14_ratio", "h14_17_ratio", "h17_21_ratio", "h21_24_ratio"]
    R["time_by_type"] = meas_a.groupby("상권구분")[tcols].mean().join(meas_a.groupby("상권구분").size().rename("n"))
    acols = ["age10_ratio", "age20_ratio", "age30_ratio", "age40_ratio", "age50_ratio", "age60p_ratio"]
    R["age_by_type"] = meas_a.groupby("상권구분")[acols].mean().join(meas_a.groupby("상권구분").size().rename("n"))
    R["weekday_by_type"] = meas_a.groupby("상권구분")[["weekend_ratio", "friSat_ratio", "male_sales_ratio"]].mean()

    # --- 상권 동태
    dyn = meas_a.groupby("상권구분")[["open_rate", "close_rate", "store_growth", "sales_yoy", "sales_cv",
                                       "franchise_ratio", "opr_months", "cls_months"]].mean()
    dyn["n"] = meas_a.groupby("상권구분").size()
    R["dynamics_by_type"] = dyn
    R["chnge_by_type"] = gs.groupby(["상권구분", "TRDAR_CHNGE_IX_NM"]).size().unstack(fill_value=0)
    R["yoy_negative_share"] = float((meas_a["sales_yoy"] < 0).mean())
    R["yoy_median"] = float(meas_a["sales_yoy"].median())
    R["yoy_n"] = int(meas_a["sales_yoy"].notna().sum())

    # --- 입지 환경
    env_cols = ["population_total", "TOT_REPOP_CO", "TOT_WRC_POPLTN_CO", "TOT_FLPOP_CO", "apt_households",
                "accessibility", "competition_density", "biz_total", "biz_pub", "biz_office", "lv_night_ratio",
                "fl_age2030_ratio", "res_age60p_ratio"]
    env = gs.groupby("상권구분")[env_cols].median()
    env["n"] = gs.groupby("상권구분").size()
    R["env_by_type"] = env
    R["env_by_track"] = gs.groupby("트랙")[["population_total", "apt_households", "competition_density",
                                            "accessibility", "biz_total"]].median().join(gs.groupby("트랙").size().rename("n"))

    # --- 임대료·공실 (R-ONE)
    rent = e.groupby("rent_area")[["rent_per_m2_kwon", "vacancy_pct"]].first()
    rent["n_sanggwon"] = e.groupby("rent_area").size()
    R["rent_area"] = rent.sort_values("rent_per_m2_kwon", ascending=False)

    m = meas_a.copy()
    m["rent_month_man"] = m["rent_per_m2_kwon"] * STORE_AREA_M2 / 10
    m["rent_burden"] = m["rent_month_man"] / m["sales_per_store_man"]
    R["rent_burden_stats"] = m["rent_burden"].describe()
    R["rent_burden_over10"] = int((m["rent_burden"] > 0.10).sum())
    R["rent_burden_n"] = int(len(m))
    R["rent_burden_df"] = m[["상권명", "구", "상권구분", "sales_per_store_man", "rent_month_man", "rent_burden"]]

    # --- 지표 상관 (실측 A 풀, Spearman, BH-FDR)
    cand = ["population_total", "TOT_REPOP_CO", "TOT_WRC_POPLTN_CO", "TOT_FLPOP_CO", "apt_households",
            "accessibility", "competition_density", "night_flpop_ratio", "면적_ha", "VIATR_FCLTY_CO", "cvs_store_avg",
            "chnge_stability", "weekend_ratio", "h17_21_ratio", "age20_ratio", "age30_ratio", "age60p_ratio",
            "male_sales_ratio", "avg_ticket_won", "open_rate", "close_rate", "store_growth", "sales_yoy", "opr_months",
            "fl_age2030_ratio", "fl_weekend_ratio", "fl_evening_ratio", "res_age2030_ratio", "res_age60p_ratio",
            "households", "wrk_age2030_ratio", "fac_hospital", "fac_pharmacy", "fac_bank", "fac_school",
            "lv_night_ratio", "biz_total", "biz_food", "biz_pub", "biz_stay", "biz_edu", "biz_clinic", "biz_office",
            "biz_retail", "rent_per_m2_kwon", "vacancy_pct"]
    cand = [c for c in cand if c in meas_a.columns]
    rows = []
    for target in ["sales_per_store", "night_sales_ratio"]:
        for c in cand:
            sub = meas_a[[target, c]].dropna()
            if len(sub) < 20 or sub[c].nunique() < 3:
                continue
            rho, p = stats.spearmanr(sub[target], sub[c])
            rows.append((target, c, rho, p, len(sub)))
    cor = pd.DataFrame(rows, columns=["target", "feature", "rho", "p", "n"])
    qs = []
    for t, g in cor.groupby("target"):
        order = g["p"].rank(method="first").astype(int)
        mm = len(g)
        q = (g["p"] * mm / order)
        # BH: 오름차순 p에서 뒤에서부터 누적 최소
        gg = g.assign(q=q).sort_values("p")
        gg["q"] = gg["q"][::-1].cummin()[::-1].clip(upper=1.0)
        qs.append(gg)
    cor = pd.concat(qs)
    R["corr"] = cor.sort_values(["target", "p"])

    # --- 상권 유형화 (강남·송파 173곳; 서초는 14번·공동주택 원천이 없어 군집에서 제외)
    d = gs.copy()
    X = pd.DataFrame({
        "인구(로그)": np.log1p(d["population_total"]),
        "상주 비중": d["TOT_REPOP_CO"] / d["population_total"].where(d["population_total"] > 0),
        "유동인구(로그)": np.log1p(d["TOT_FLPOP_CO"]),
        "심야 유동 비중": d["night_flpop_ratio"],
        "아파트 세대(로그)": np.log1p(d["apt_households"]),
        "경쟁 편의점(로그)": np.log1p(d["competition_density"]),
        "접근성(로그)": np.log1p(d["accessibility"]),
        "면적(로그)": np.log(d["면적_ha"]),
        "주변 업소(로그)": np.log1p(d["biz_total"]),
    }, index=d.index)
    X = X.fillna(X.median())
    Z = StandardScaler().fit_transform(X)
    sil = {}
    for k in range(2, 8):
        sil[k] = float(silhouette_score(Z, KMeans(k, n_init=50, random_state=0).fit(Z).labels_))
    km = KMeans(4, n_init=50, random_state=0).fit(Z)
    prof = pd.DataFrame(Z, index=d.index, columns=X.columns).groupby(km.labels_).mean()
    # 규칙 기반 이름 부여(군집 번호는 실행마다 달라질 수 있어 중심값으로 이름을 정한다)
    names = {}
    left = set(prof.index)
    big = prof["인구(로그)"].idxmax(); names[big] = "대형 업무·번화"; left.discard(big)
    sparse = prof.loc[list(left), "경쟁 편의점(로그)"].idxmin(); names[sparse] = "상가·경쟁 희박 외곽"; left.discard(sparse)
    resi = prof.loc[list(left), "상주 비중"].idxmax(); names[resi] = "주거 밀집 골목"; left.discard(resi)
    for k_ in left:
        names[k_] = "소형 저밀 생활권"
    d["군집"] = pd.Series(km.labels_, index=d.index).map(names)
    prof.index = [names[i] for i in prof.index]
    R["cluster_profile"] = prof
    R["cluster_silhouette"] = sil
    gm = d[d["트랙"] == "실측"].copy()
    clu = d.groupby("군집").agg(상권수=("상권명", "size"))
    clu["실측"] = d[d["트랙"] == "실측"].groupby("군집").size()
    clu["매출불명"] = d[d["트랙"] == "매출불명"].groupby("군집").size()
    clu = clu.fillna(0).astype(int)
    for c, lab in [("sales_per_store", "점포당 월매출 중앙값(만원)"), ("night_sales_ratio", "심야매출비중 평균"),
                   ("age20_ratio", "20대 매출비중"), ("age30_ratio", "30대 매출비중"), ("age60p_ratio", "60대+ 매출비중"),
                   ("sales_yoy", "매출 전년비 평균"), ("rent_per_m2_kwon", "임대료(천원/㎡) 평균")]:
        g = gm.groupby("군집")[c]
        v = g.median() / 1e4 if c == "sales_per_store" else g.mean()
        clu[lab] = v
    R["cluster_summary"] = clu
    R["cluster_by_type"] = pd.crosstab(d["군집"], d["상권구분"])
    R["cluster_by_gu"] = pd.crosstab(d["군집"], d["구"])
    R["cluster_examples"] = {n: list(d[d["군집"] == n].sort_values("population_total", ascending=False)["상권명"].head(6))
                             for n in names.values()}
    R["cluster_series"] = d["군집"]
    R["cluster_sales_box"] = {n: (gm[gm["군집"] == n]["sales_per_store"] / 1e4).dropna().values for n in names.values()}

    # --- 매출불명 트랙: 수요 지표 상위
    unk = gs[gs["트랙"] == "매출불명"].sort_values("population_total", ascending=False)
    R["unknown_top"] = unk[["상권명", "구", "상권구분", "population_total", "apt_households", "competition_density",
                           "accessibility"]].head(10)
    R["unknown_n"] = int(len(unk))
    return R


# ============================================================ 2~4. 입지 · ROI
def roi_frame(df: pd.DataFrame, fr: dict, gm: float, rent_on: bool, wage: float, extra_cash: float = 0.0) -> pd.DataFrame:
    out = df[["상권명", "구", "행정동", "상권구분", "sales_per_store", "cvs_store_avg", "이상값_점포당매출",
              "rent_per_m2_kwon", "rent_area", "score", "top3_freq"]].copy()
    out["sales_man"] = out["sales_per_store"] / 1e4
    out["rent_man"] = out["rent_per_m2_kwon"] * STORE_AREA_M2 / 10 if rent_on else 0.0
    out["owner_man"] = out["sales_man"] * gm * fr["share"]
    out["net_man"] = out["owner_man"] - wage - out["rent_man"]
    out["invest_man"] = fr["cash"] + extra_cash + out["rent_man"] * DEPOSIT_MONTHS
    out["roi"] = out["net_man"] * 12 / out["invest_man"]
    out["payback_yr"] = np.where(out["net_man"] > 0, out["invest_man"] / (out["net_man"] * 12), np.nan)
    out["bep_sales_man"] = (wage + out["rent_man"]) / (gm * fr["share"])
    out["rent_burden"] = out["rent_man"] / out["sales_man"]
    out["small_n"] = out["cvs_store_avg"] < SMALL_N
    out["outlier"] = out["이상값_점포당매출"].fillna("") != ""
    return out


def top3_freq(scored: pd.DataFrame, weights: dict, n: int = 3000, spread: float = 0.3, seed: int = 0) -> pd.Series:
    """각 가중치를 ±spread 범위에서 흔들었을 때 상위 3위 안에 든 비율."""
    rng = np.random.default_rng(seed)
    cols = list(weights)
    N = np.column_stack([scored[f"{c}_norm"].values for c in cols])
    base = np.array([abs(weights[c]) for c in cols])
    counts = np.zeros(len(scored))
    for _ in range(n):
        w = base * rng.uniform(1 - spread, 1 + spread, size=len(cols))
        counts[np.argsort(-(N @ w))[:3]] += 1
    return pd.Series(counts / n, index=scored.index)


def stage234(e: pd.DataFrame, wage: float) -> dict:
    R: dict = {}
    A_pool = e[e["구"].isin(["강남구", "송파구"])]
    B_pool = e[e["dist_to_dogok_m"] <= B_NEARBY_RADIUS_M]
    sa = score_candidates(A_pool, WEIGHTS_A)
    sb = score_candidates(B_pool, WEIGHTS_B)
    sa["top3_freq"] = top3_freq(sa, WEIGHTS_A)
    sb["top3_freq"] = top3_freq(sb, WEIGHTS_B)
    R["score_a"] = sa
    R["score_b"] = sb

    # B 가중치 민감도: 점포당 매출 가중치 0.30으로 올리고 나머지를 비율대로 줄여 합 1.00 (팀 미결정, 참고)
    w = dict(WEIGHTS_B)
    others = {k: v for k, v in w.items() if k != "sales_per_store"}
    osum = sum(abs(v) for v in others.values())
    w030 = {k: v * (1 - 0.30) / osum for k, v in others.items()}
    w030["sales_per_store"] = 0.30
    R["score_b_w030"] = score_candidates(B_pool, w030)

    fr_a = pick_franchise("GS25", "GS2", "24시간")
    fr_b = pick_franchise("세븐일레븐", "기본투자형", "24시간")
    R["fr_a"], R["fr_b"] = fr_a, fr_b
    R["wage_night_man"] = wage
    R["assump"] = dict(GM_BASE=GM_BASE, GM_ALT=GM_ALT, STORE_AREA_M2=STORE_AREA_M2, DEPOSIT_MONTHS=DEPOSIT_MONTHS,
                       NIGHT_HOURS=NIGHT_HOURS, DAYS=DAYS, SMALL_N=SMALL_N)

    # A: 본부 임차형(임차료는 배분율에 포함) -> 월세 차감 없음. 전대보증금 3,000만원 별도 시나리오 병행.
    ra = roi_frame(sa, fr_a, GM_BASE, False, wage)
    ra_alt = roi_frame(sa, fr_a, GM_ALT, False, wage)
    ra_dep = roi_frame(sa, fr_a, GM_BASE, False, wage, extra_cash=fr_a["hq_deposit"])
    # 참고용: A 상권의 월세(본부 부담)와 매출 대비 비율
    ra["rent_ref_man"] = ra["rent_per_m2_kwon"] * STORE_AREA_M2 / 10
    ra["rent_ref_burden"] = ra["rent_ref_man"] / ra["sales_man"]

    # B: 경영주 임차형 -> 월세·보증금 반영
    rb = roi_frame(sb, fr_b, GM_BASE, True, wage)
    rb_alt = roi_frame(sb, fr_b, GM_ALT, True, wage)
    # 임차료를 안 넣은 B ROI(비교용): 앞서 보고한 "임차료 미반영" 상태
    rb_norent = roi_frame(sb, fr_b, GM_BASE, False, wage)

    for df, nm in [(ra, "A"), (rb, "B")]:
        df["rank_loc"] = df["score"].rank(ascending=False, method="first").astype(int)
        df["rank_roi"] = df["roi"].rank(ascending=False, method="first").astype(int)
        df["rank_move"] = df["rank_loc"] - df["rank_roi"]  # +면 ROI에서 순위 상승
    rb_norent["rank_roi_norent"] = rb_norent["roi"].rank(ascending=False, method="first").astype(int)
    rb = rb.join(rb_norent[["roi", "rank_roi_norent"]].rename(columns={"roi": "roi_norent"}))

    R["roi_a"], R["roi_a_alt"], R["roi_a_dep"] = ra, ra_alt, ra_dep
    R["roi_b"], R["roi_b_alt"] = rb, rb_alt

    # ---- 추천 3곳
    loc_a = ra.sort_values("rank_loc").head(3)
    # ROI 추천: 이상값(점포당 매출 상위 이상값) 상권은 금액 지표가 왜곡되므로 후보에서 제외하고 따로 표시한다.
    roi_pool_a = ra[~ra["outlier"]]
    roi_a = roi_pool_a.sort_values("roi", ascending=False).head(3)
    loc_b = rb.sort_values("rank_loc").head(3)
    roi_b = rb.sort_values("roi", ascending=False).head(3)
    R["rec"] = {"loc_a": loc_a, "roi_a": roi_a, "loc_b": loc_b, "roi_b": roi_b}

    # 30% 총이익률에서도 같은 3곳인가
    ra_alt_pool = ra_alt[~ra_alt["outlier"]].sort_values("roi", ascending=False).head(3)
    rb_alt_top = rb_alt.sort_values("roi", ascending=False).head(3)
    R["rec_alt_same_a"] = list(ra_alt_pool.index) == list(roi_a.index)
    R["rec_alt_same_b"] = list(rb_alt_top.index) == list(roi_b.index)
    R["roi_a_alt_top"], R["roi_b_alt_top"] = ra_alt_pool, rb_alt_top

    # ---- 순위 변동 지표
    from scipy.stats import spearmanr
    R["spearman_a"] = float(spearmanr(ra["score"], ra["roi"])[0])
    R["spearman_b"] = float(spearmanr(rb["score"], rb["roi"])[0])
    R["spearman_a_nonout"] = float(spearmanr(ra[~ra["outlier"]]["score"], ra[~ra["outlier"]]["roi"])[0])
    top10_loc_a = set(ra.sort_values("rank_loc").head(10).index)
    top10_roi_a = set(ra.sort_values("roi", ascending=False).head(10).index)
    R["top10_overlap_a"] = len(top10_loc_a & top10_roi_a)
    top10_roi_a_non = set(roi_pool_a.sort_values("roi", ascending=False).head(10).index)
    R["top10_overlap_a_nonout"] = len(top10_loc_a & top10_roi_a_non)
    R["a_negative_roi"] = int((ra["net_man"] <= 0).sum())
    R["a_n"] = int(len(ra))
    R["b_negative_roi"] = int((rb["net_man"] <= 0).sum())
    R["b_n"] = int(len(rb))
    R["a_bep_man"] = float(ra["bep_sales_man"].iloc[0])
    R["a_outlier_n"] = int(ra["outlier"].sum())

    # ---- B: 2호점 근접성 (ROI가 플러스인 상권 사이 거리)
    pos = rb[rb["net_man"] > 0].index
    xy = e.loc[pos, ["XCNTS_VALU", "YDNTS_VALU"]].astype(float)
    dist = pd.DataFrame(
        np.sqrt((xy["XCNTS_VALU"].values[:, None] - xy["XCNTS_VALU"].values[None, :]) ** 2 +
                (xy["YDNTS_VALU"].values[:, None] - xy["YDNTS_VALU"].values[None, :]) ** 2),
        index=[e.loc[i, "상권명"] for i in pos], columns=[e.loc[i, "상권명"] for i in pos])
    R["b_dist_matrix"] = dist
    return R


# ============================================================ 그림
def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=170)
    plt.close(fig)


def fig_overview(R1):
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6))
    t = R1["overview_track_gu"]
    cols = {"실측": C_BLUE, "매출불명": C_LIGHT, "후보제외": C_GRAY}
    bottom = np.zeros(len(t))
    for c in ["실측", "매출불명", "후보제외"]:
        if c in t.columns:
            ax[0].bar(t.index, t[c], bottom=bottom, color=cols[c], label=c)
            for i, v in enumerate(t[c]):
                if v > 0:
                    ax[0].text(i, bottom[i] + v / 2, str(v), ha="center", va="center", fontsize=9,
                               color="white" if c == "실측" else "#1d2433")
            bottom += t[c].values
    ax[0].set_title("구별 후보 상권과 트랙 (서초구는 도곡 반경 포함분)", fontsize=10)
    ax[0].legend(frameon=False, fontsize=8)
    ty = R1["overview_track_type"]
    bottom = np.zeros(len(ty))
    for c in ["실측", "매출불명"]:
        if c in ty.columns:
            ax[1].bar(ty.index, ty[c], bottom=bottom, color=cols[c], label=c)
            for i, v in enumerate(ty[c]):
                if v > 0:
                    ax[1].text(i, bottom[i] + v / 2, str(v), ha="center", va="center", fontsize=9,
                               color="white" if c == "실측" else "#1d2433")
            bottom += ty[c].values
    ax[1].set_title("상권구분별 트랙 (강남·송파 173곳)", fontsize=10)
    savefig(fig, "fig_01_overview.png")


def fig_time_age(R1):
    fig, ax = plt.subplots(1, 2, figsize=(10.8, 3.8))
    t = R1["time_by_type"]
    labs = ["00-06", "06-11", "11-14", "14-17", "17-21", "21-24"]
    pal = ["#27355a", "#5c82c9", "#9bb7e6", "#cfd6e2", C_ORANGE, "#7a2e00"]
    cols = ["h00_06_ratio", "h06_11_ratio", "h11_14_ratio", "h14_17_ratio", "h17_21_ratio", "h21_24_ratio"]
    y = np.arange(len(t))
    left = np.zeros(len(t))
    for c, l, p in zip(cols, labs, pal):
        ax[0].barh(y, t[c].values, left=left, color=p, label=l)
        left += t[c].values
    ax[0].set_yticks(y)
    ax[0].set_yticklabels([f"{i} (n={int(n)})" for i, n in zip(t.index, t["n"])], fontsize=9)
    ax[0].set_xlim(0, 1)
    ax[0].set_title("시간대별 매출 비중 (편의점, 실측 평균)", fontsize=10)
    ax[0].legend(ncol=6, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    a = R1["age_by_type"]
    alabs = ["10대", "20대", "30대", "40대", "50대", "60대+"]
    apal = ["#cfd6e2", "#9bb7e6", C_BLUE, "#27355a", C_ORANGE, "#7a2e00"]
    acols = ["age10_ratio", "age20_ratio", "age30_ratio", "age40_ratio", "age50_ratio", "age60p_ratio"]
    left = np.zeros(len(a))
    y = np.arange(len(a))
    for c, l, p in zip(acols, alabs, apal):
        ax[1].barh(y, a[c].values, left=left, color=p, label=l)
        left += a[c].values
    ax[1].set_yticks(y)
    ax[1].set_yticklabels([f"{i} (n={int(n)})" for i, n in zip(a.index, a["n"])], fontsize=9)
    ax[1].set_xlim(0, 1)
    ax[1].set_title("연령대별 매출 비중 (편의점, 실측 평균)", fontsize=10)
    ax[1].legend(ncol=6, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    savefig(fig, "fig_02_time_age.png")


def fig_cluster(R1):
    prof = R1["cluster_profile"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={"width_ratios": [1.5, 1]})
    im = ax[0].imshow(prof.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax[0].set_xticks(range(prof.shape[1]))
    ax[0].set_xticklabels(prof.columns, rotation=40, ha="right", fontsize=8)
    ax[0].set_yticks(range(prof.shape[0]))
    ax[0].set_yticklabels(prof.index, fontsize=9)
    for i in range(prof.shape[0]):
        for j in range(prof.shape[1]):
            ax[0].text(j, i, f"{prof.values[i, j]:.1f}", ha="center", va="center", fontsize=7)
    ax[0].set_title("군집 프로파일 (z-점수 평균)", fontsize=10)
    ax[0].spines[:].set_visible(False)
    names = list(prof.index)
    data = [R1["cluster_sales_box"][n] for n in names]
    bp = ax[1].boxplot(data, orientation="horizontal", patch_artist=True, tick_labels=[f"{n}\n(실측 {len(d)})" for n, d in zip(names, data)],
                       medianprops=dict(color="white", linewidth=1.5))
    for p in bp["boxes"]:
        p.set_facecolor(C_BLUE)
    ax[1].set_xscale("log")
    ax[1].set_title("군집별 점포당 월매출(만원, 로그축)", fontsize=10)
    ax[1].tick_params(axis="y", labelsize=8)
    savefig(fig, "fig_03_cluster.png")


def fig_rent(R1):
    rent = R1["rent_area"]
    d = R1["rent_burden_df"]
    fig, ax = plt.subplots(1, 2, figsize=(10.8, 3.8))
    labs = [i.replace("서울>", "").replace("기타>", "기타·") for i in rent.index]
    ax[0].barh(range(len(rent)), rent["rent_per_m2_kwon"], color=C_BLUE)
    for i, (v, vac) in enumerate(zip(rent["rent_per_m2_kwon"], rent["vacancy_pct"])):
        ax[0].text(v + 0.5, i, f"{v:.1f} (공실 {vac:.1f}%)", va="center", fontsize=8)
    ax[0].set_yticks(range(len(rent)))
    ax[0].set_yticklabels(labs, fontsize=8)
    ax[0].invert_yaxis()
    ax[0].set_xlim(0, 85)
    ax[0].set_title("R-ONE 1층 임대료(천원/㎡·월)와 공실률", fontsize=10)
    colors = d["구"].map({"강남구": C_BLUE, "송파구": C_ORANGE})
    ax[1].scatter(d["sales_per_store_man"], d["rent_burden"] * 100, c=colors, s=22, alpha=0.8)
    ax[1].set_xscale("log")
    ax[1].axhline(10, color=C_RED, linestyle="--", linewidth=1)
    ax[1].text(ax[1].get_xlim()[1], 10.4, "월세/매출 10%", ha="right", fontsize=8, color=C_RED)
    ax[1].set_xlabel("점포당 월매출(만원, 로그축)", fontsize=9)
    ax[1].set_ylabel("월세 ÷ 월매출(%)  (50㎡ 가정)", fontsize=9)
    ax[1].set_title("매출이 작을수록 월세 부담이 급격히 커진다", fontsize=10)
    ax[1].scatter([], [], c=C_BLUE, label="강남구")
    ax[1].scatter([], [], c=C_ORANGE, label="송파구")
    ax[1].legend(frameon=False, fontsize=8, loc="upper right")
    savefig(fig, "fig_04_rent.png")


def fig_corr(R1):
    cor = R1["corr"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), sharex=False)
    names = {
        "population_total": "상주+직장인구", "TOT_REPOP_CO": "상주인구", "TOT_WRC_POPLTN_CO": "직장인구", "TOT_FLPOP_CO": "유동인구",
        "apt_households": "아파트 세대", "accessibility": "접근성", "competition_density": "경쟁 편의점(300m)",
        "night_flpop_ratio": "심야 유동 비중", "면적_ha": "상권 면적", "VIATR_FCLTY_CO": "집객시설 수",
        "cvs_store_avg": "상권 내 편의점 수", "chnge_stability": "상권변화 안정성", "weekend_ratio": "주말 매출 비중",
        "h17_21_ratio": "17-21시 매출비중", "age20_ratio": "20대 매출비중", "age30_ratio": "30대 매출비중",
        "age60p_ratio": "60대+ 매출비중", "male_sales_ratio": "남성 매출비중", "avg_ticket_won": "객단가",
        "open_rate": "개업률", "close_rate": "폐업률", "store_growth": "점포 수 증감", "sales_yoy": "매출 전년비",
        "opr_months": "운영 영업개월", "fl_age2030_ratio": "유동 20·30대", "fl_weekend_ratio": "유동 주말비중",
        "fl_evening_ratio": "유동 저녁비중", "res_age2030_ratio": "상주 20·30대", "res_age60p_ratio": "상주 60대+",
        "households": "가구 수", "wrk_age2030_ratio": "직장 20·30대", "fac_hospital": "병원", "fac_pharmacy": "약국",
        "fac_bank": "은행", "fac_school": "학교", "lv_night_ratio": "심야 생활인구 비중", "biz_total": "주변 업소 수",
        "biz_food": "주변 음식점", "biz_pub": "주변 주점", "biz_stay": "주변 숙박", "biz_edu": "주변 교육",
        "biz_clinic": "주변 의원", "biz_office": "주변 사무·서비스", "biz_retail": "주변 소매",
        "rent_per_m2_kwon": "임대료", "vacancy_pct": "공실률",
    }
    for a, (tgt, ttl) in zip(ax, [("sales_per_store", "점포당 월매출"), ("night_sales_ratio", "심야 매출비중")]):
        g = cor[(cor["target"] == tgt) & (cor["q"] < 0.05)].copy()
        g = g.reindex(g["rho"].abs().sort_values(ascending=False).index).head(12).sort_values("rho")
        colors = [C_BLUE if v > 0 else C_ORANGE for v in g["rho"]]
        a.barh([names.get(f, f) for f in g["feature"]], g["rho"], color=colors)
        for i, v in enumerate(g["rho"]):
            a.text(v + (0.01 if v > 0 else -0.01), i, f"{v:+.2f}", va="center", ha="left" if v > 0 else "right", fontsize=8)
        a.axvline(0, color="#444", linewidth=0.8)
        a.set_title(f"{ttl}와 상관 (Spearman, FDR q<0.05)", fontsize=10)
        a.set_xlim(-1, 1)
        a.tick_params(axis="y", labelsize=8)
    savefig(fig, "fig_05_corr.png")


def fig_roi(R):
    ra, rb = R["roi_a"], R["roi_b"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
    pool = ra.sort_values("roi", ascending=False).head(14).iloc[::-1]
    colors = [C_GRAY if o else C_BLUE for o in pool["outlier"]]
    ax[0].barh(pool["상권명"], pool["roi"] * 100, color=colors)
    for i, (v, o, s) in enumerate(zip(pool["roi"] * 100, pool["outlier"], pool["small_n"])):
        ax[0].text(v + 5, i, f"{v:,.0f}%" + (" 이상값" if o else "") + (" 소표본" if s else ""), va="center", fontsize=8)
    ax[0].set_title("A: ROI 상위 14곳 (GS25 GS2, 총이익률 25%, 회색=점포당 매출 이상값)", fontsize=9.5)
    ax[0].set_xlim(0, pool["roi"].max() * 100 * 1.35)
    ax[0].tick_params(axis="y", labelsize=8)
    b = rb.sort_values("roi", ascending=True)
    colors = [C_RED if v < 0 else C_GREEN for v in b["roi"]]
    ax[1].barh(b["상권명"], b["roi"] * 100, color=colors)
    for i, v in enumerate(b["roi"] * 100):
        ax[1].text(v + (4 if v >= 0 else -4), i, f"{v:,.0f}%", va="center", ha="left" if v >= 0 else "right", fontsize=8)
    ax[1].axvline(0, color="#444", linewidth=0.8)
    ax[1].set_title("B: ROI (세븐일레븐 기본투자형, 임차료 반영, 25%)", fontsize=9.5)
    ax[1].set_xlim(-45, b["roi"].max() * 100 * 1.25)
    ax[1].tick_params(axis="y", labelsize=8)
    savefig(fig, "fig_07_roi.png")


def slope(ax, df, title, items, label_col="상권명"):
    sub = df.loc[items]
    ax.set_xlim(-0.55, 1.55)
    for _, r in sub.iterrows():
        up = r["rank_roi"] < r["rank_loc"]
        down = r["rank_roi"] > r["rank_loc"]
        col = C_GREEN if up else (C_RED if down else C_GRAY)
        ax.plot([0, 1], [r["rank_loc"], r["rank_roi"]], color=col, linewidth=1.6, alpha=0.9)
        ax.scatter([0, 1], [r["rank_loc"], r["rank_roi"]], color=col, s=14, zorder=3)
        ax.text(-0.05, r["rank_loc"], f"{int(r['rank_loc'])}  {r[label_col]}", ha="right", va="center", fontsize=7.5)
        ax.text(1.05, r["rank_roi"], f"{int(r['rank_roi'])}  {r[label_col]}", ha="left", va="center", fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["입지 순위", "ROI 순위"], fontsize=9)
    ax.set_yticks([])
    ax.spines[:].set_visible(False)
    ax.set_title(title, fontsize=10)


def fig_slope(R):
    ra, rb = R["roi_a"], R["roi_b"]
    ia = set(ra.sort_values("rank_loc").head(10).index) | set(ra.sort_values("rank_roi").head(10).index)
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 9.0), gridspec_kw={"width_ratios": [1.1, 1]})
    slope(ax[0], ra, "A: 입지 상위10 ∪ ROI 상위10 (90곳 중 순위)", list(ia))
    slope(ax[1], rb, "B: 후보 10곳의 순위", list(rb.index))
    savefig(fig, "fig_08_slope.png")


def plot_labeled_map(boundary_df, shapes, scored, title, out_name, top_df, offsets):
    """입지 점수 지도. 상위 3곳을 빨간 테두리와 번호·이름으로 표시하고, 점수화된 상권 범위로 확대한다."""
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon as MplPolygon

    code_to_idx = {code: i for i, code in enumerate(boundary_df["TRDAR_CD"])}
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    patches, colors = [], []
    xs, ys = [], []
    for code, row in scored.iterrows():
        idx = code_to_idx.get(code)
        if idx is None:
            continue
        shp = shapes[idx]
        parts = list(shp.parts) + [len(shp.points)]
        for p in range(len(parts) - 1):
            pts = shp.points[parts[p]:parts[p + 1]]
            patches.append(MplPolygon(pts, closed=True))
            colors.append(row["score"])
            xs.extend(pt[0] for pt in pts)
            ys.extend(pt[1] for pt in pts)
    pc = PatchCollection(patches, cmap=plt.get_cmap("YlGnBu"), edgecolor="#555555", linewidth=0.3)
    pc.set_array(np.array(colors))
    ax.add_collection(pc)
    for rank, (code, row) in enumerate(top_df.iterrows(), start=1):
        idx = code_to_idx[code]
        shp = shapes[idx]
        parts = list(shp.parts) + [len(shp.points)]
        for p in range(len(parts) - 1):
            ax.add_patch(MplPolygon(shp.points[parts[p]:parts[p + 1]], closed=True, fill=False, edgecolor="#d62728", linewidth=2))
        cx, cy = (shp.bbox[0] + shp.bbox[2]) / 2, (shp.bbox[1] + shp.bbox[3]) / 2
        dx, dy = offsets[rank - 1]
        ax.annotate(f"{rank}. {row['상권명']}", (cx, cy), xytext=(dx, dy), textcoords="offset points", fontsize=9, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#d62728", lw=1),
                    arrowprops=dict(arrowstyle="-", color="#d62728", lw=1.2))
    mx = (max(xs) - min(xs)) * 0.04
    my = (max(ys) - min(ys)) * 0.04
    ax.set_xlim(min(xs) - mx, max(xs) + mx)
    ax.set_ylim(min(ys) - my, max(ys) + my)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)
    ax.set_title(title, fontsize=10)
    cb = fig.colorbar(pc, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("입지 점수 (백분위 가중합, 상대비교)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    savefig(fig, out_name)


def fig_extra(R1, R2, e):
    """보고서용 추가 그림: 점포당 매출 분포, 업무형↔주거형 산점도, 입지-ROI 순위 산점도, 매출 대 손익분기(B)."""
    gs = e[(e["구"].isin(["강남구", "송파구"])) & (e["트랙"] == "실측")].copy()
    gs["sales_man"] = gs["sales_per_store"] / 1e4

    # (1) 점포당 월매출 분포
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    types = ["골목상권", "발달상권", "전통시장", "관광특구"]
    cols = {"골목상권": C_ORANGE, "발달상권": C_BLUE, "전통시장": C_GREEN, "관광특구": C_GRAY}
    rng = np.random.default_rng(1)
    for i, t in enumerate(types):
        v = gs.loc[gs["상권구분"] == t, "sales_man"].values
        ax.scatter(v, np.full(len(v), i) + rng.uniform(-0.18, 0.18, len(v)), s=22, color=cols[t], alpha=0.8)
        if len(v):
            ax.plot([np.median(v)] * 2, [i - 0.3, i + 0.3], color="#1d2433", linewidth=2)
            ax.text(np.median(v), i - 0.36, f"중앙값 {np.median(v):,.0f}", ha="center", va="top", fontsize=7.5)
    ax.set_xscale("log")
    ax.set_yticks(range(len(types)))
    ax.set_yticklabels([f"{t} (n={int((gs['상권구분'] == t).sum())})" for t in types], fontsize=9)
    ax.set_ylim(len(types) - 0.4, -0.8)
    ax.set_xlabel("점포당 월매출(만원, 로그축)", fontsize=9)
    ax.set_title("상권구분별 점포당 월매출 분포 (실측 90곳, 검은 선 = 중앙값)", fontsize=10)
    savefig(fig, "fig_09_dist.png")

    # (2) 업무형 <-> 주거형
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    clu = R1["cluster_series"].reindex(gs.index)
    pal = {"대형 업무·번화": C_BLUE, "주거 밀집 골목": C_ORANGE, "소형 저밀 생활권": C_GREEN, "상가·경쟁 희박 외곽": C_GRAY}
    x = gs["TOT_WRC_POPLTN_CO"] / gs["population_total"].where(gs["population_total"] > 0)
    for n, c in pal.items():
        m = clu == n
        if m.sum():
            ax.scatter(x[m] * 100, gs.loc[m, "night_sales_ratio"] * 100, s=np.clip(gs.loc[m, "sales_man"] / 40, 10, 400),
                       color=c, alpha=0.65, edgecolor="white", linewidth=0.6, label=f"{n} ({int(m.sum())})")
    ax.set_xlabel("직장인구 비중 (직장 ÷ (상주+직장), %)", fontsize=9)
    ax.set_ylabel("심야(21~06시) 매출 비중 (%)", fontsize=9)
    ax.set_title("직장 중심일수록 심야 비중이 낮다 (원 크기 = 점포당 월매출)", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    for nm in ["장지역(가든파이브)", "한티역", "삼성역", "석촌고분역 4번", "윗방죽마을공원"]:
        r = gs[gs["상권명"] == nm]
        if len(r):
            xi = (r["TOT_WRC_POPLTN_CO"] / r["population_total"]).iloc[0] * 100
            ax.annotate(nm, (xi, r["night_sales_ratio"].iloc[0] * 100), xytext=(4, 6), textcoords="offset points", fontsize=7.5)
    savefig(fig, "fig_10_type_scatter.png")

    # (3) A: 입지 순위 vs ROI 순위
    ra = R2["roi_a"]
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.plot([1, len(ra)], [1, len(ra)], color=C_LIGHT, linestyle="--", linewidth=1)
    colors = np.where(ra["outlier"], C_GRAY, C_BLUE)
    ax.scatter(ra["rank_loc"], ra["rank_roi"], c=colors, s=26, alpha=0.85)
    lab = set(R2["rec"]["loc_a"].index) | set(R2["rec"]["roi_a"].index)
    for tid in lab:
        r = ra.loc[tid]
        ax.annotate(r["상권명"], (r["rank_loc"], r["rank_roi"]), xytext=(5, 5), textcoords="offset points", fontsize=7.5)
    ax.set_xlabel("입지 순위 (90곳 중)", fontsize=9)
    ax.set_ylabel("ROI 순위 (90곳 중)", fontsize=9)
    ax.set_title("A: 입지 순위와 ROI 순위 (회색 = 점포당 매출 이상값)", fontsize=10)
    ax.set_xlim(0, len(ra) + 2)
    ax.set_ylim(len(ra) + 2, 0)
    savefig(fig, "fig_11_rank_scatter.png")

    # (4) B: 점포당 월매출 대 손익분기
    rb = R2["roi_b"].sort_values("sales_man")
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    y = np.arange(len(rb))
    ax.barh(y + 0.18, rb["sales_man"], height=0.36, color=C_BLUE, label="점포당 월매출")
    ax.barh(y - 0.18, rb["bep_sales_man"], height=0.36, color=C_ORANGE, label="손익분기 월매출 (월세·인건비 포함)")
    ax.set_yticks(y)
    ax.set_yticklabels(rb["상권명"], fontsize=8)
    for i, (s, b) in enumerate(zip(rb["sales_man"], rb["bep_sales_man"])):
        ax.text(max(s, b) + 80, i, "적자" if s < b else "흑자", va="center", fontsize=8, color=C_RED if s < b else C_GREEN)
    ax.set_xlabel("만원/월", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("B: 매출이 손익분기를 넘는 상권은 적다", fontsize=10)
    savefig(fig, "fig_12_bep.png")


def maps(R, boundary_df, shapes):
    sa, sb = R["score_a"], R["score_b"]
    plot_labeled_map(boundary_df, shapes, sa, "A 입지 점수: 강남·송파 실측 90곳 (빨간 테두리 = 입지 상위 3곳)", "fig_06_map_A.png",
                     R["rec"]["loc_a"], [(-110, 60), (-70, -75), (70, -70)])
    plot_labeled_map(boundary_df, shapes, sb, "B 입지 점수: 도곡동 반경 1.5km 실측 10곳 (빨간 테두리 = 입지 상위 3곳)", "fig_06_map_B.png",
                     R["rec"]["loc_b"], [(-130, 40), (60, -70), (-80, -75)])


def main() -> None:
    e = pd.read_csv(OUT / "feature_extended.csv", dtype={"TRDAR_CD": str}).set_index("TRDAR_CD")
    wage = wage_night_manwon()
    print("[pipeline] stage 1 ...")
    R1 = stage1(e)
    print("[pipeline] stage 2-4 ...")
    R2 = stage234(e, wage)
    print("[pipeline] figures ...")
    fig_overview(R1)
    fig_time_age(R1)
    fig_cluster(R1)
    fig_rent(R1)
    fig_corr(R1)
    fig_roi(R2)
    fig_slope(R2)
    fig_extra(R1, R2, e)
    boundary_df, shapes = load_trade_area_boundary()
    maps(R2, boundary_df, shapes)

    # 전체 표 저장
    cols = ["상권명", "구", "행정동", "상권구분", "score", "rank_loc", "sales_man", "cvs_store_avg", "outlier", "small_n",
            "rent_man", "owner_man", "net_man", "invest_man", "roi", "payback_yr", "bep_sales_man", "rank_roi", "rank_move"]
    R2["roi_a"][cols].sort_values("rank_loc").to_csv(OUT / "roi_A_all.csv", encoding="utf-8-sig")
    R2["roi_b"][cols + ["roi_norent", "rank_roi_norent"]].sort_values("rank_loc").to_csv(OUT / "roi_B_all.csv", encoding="utf-8-sig")
    with open(OUT / "results.pkl", "wb") as f:
        pickle.dump({"R1": R1, "R2": R2, "e": e}, f)
    print("[pipeline] done")
    print("A 입지 top3:", list(R2["rec"]["loc_a"]["상권명"]))
    print("A ROI  top3:", list(R2["rec"]["roi_a"]["상권명"]))
    print("B 입지 top3:", list(R2["rec"]["loc_b"]["상권명"]))
    print("B ROI  top3:", list(R2["rec"]["roi_b"]["상권명"]))


if __name__ == "__main__":
    main()
