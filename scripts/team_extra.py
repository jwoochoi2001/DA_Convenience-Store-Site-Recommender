# -*- coding: utf-8 -*-
"""발표 자료의 보조 분석을 재현하고 그림을 만든다.

1 시장 국면(점포 수·매출 시계열)        2 결제 건수가 매출을 가르는 정도(분산 분해)
3 상권 유형 군집(k=2)                 4 구조 지표 OLS(설명력)
5 심야 유동 vs 심야 매출              6 순위 하락폭과 상관
7 추천 6곳의 분기 안정성              8 민감도(가정 변경 · 가중치 ±30%)
입력: eda_output/team/*.csv   출력: eda_output/team/*.png, extra_results.pkl
"""
from __future__ import annotations

import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import team_spec as T
from data_paths import ROOT, load_franchise_terms
from team_score_roi import (evaluate, franchise_types, gu_rent_per_m2, pools, roi_table, team_score, top3_frequency)

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["axes.spines.right"] = False
OUT = ROOT / "eda_output" / "team"
C_BLUE, C_ORANGE, C_GREEN, C_GRAY, C_RED, C_LIGHT = "#1f5fbf", "#e07b00", "#2a9d8f", "#8a93a3", "#c0392b", "#cfd6e2"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------- 1 시장 국면
def market(panel: pd.DataFrame, codes) -> dict:
    p = panel[panel["TRDAR_CD"].isin(codes)]
    g = p.groupby("quarter").agg(stores=("SIMILR_INDUTY_STOR_CO", "sum"), sales=("THSMON_SELNG_AMT", "sum"))
    g["per_store"] = g["sales"] / g["stores"]
    last, prev = g.index[-1], g.index[-5]
    peak = g["stores"].idxmax()
    res = {
        "series": g, "last": last, "prev": prev, "peak": peak,
        "peak_stores": int(g.loc[peak, "stores"]), "last_stores": int(g.loc[last, "stores"]),
        "store_yoy": g.loc[last, "stores"] / g.loc[prev, "stores"] - 1,
        "sales_yoy": g.loc[last, "sales"] / g.loc[prev, "sales"] - 1,
        "per_store_yoy": g.loc[last, "per_store"] / g.loc[prev, "per_store"] - 1,
        "from_peak": g.loc[last, "stores"] / g.loc[peak, "stores"] - 1,
    }
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.6, 1]})
    x = np.arange(len(g))
    ax[0].plot(x, g["stores"], color=C_BLUE, marker="o", markersize=3.5)
    ax[0].axvline(list(g.index).index(peak), color=C_GRAY, linestyle=":")
    ax[0].annotate(f"정점 {peak}\n{res['peak_stores']:,}곳", (list(g.index).index(peak), res["peak_stores"]), xytext=(-60, -30), textcoords="offset points", fontsize=8,
                   arrowprops=dict(arrowstyle="-", color=C_GRAY))
    ax[0].annotate(f"{last}\n{res['last_stores']:,}곳 ({res['from_peak'] * 100:+.1f}%)", (len(g) - 1, res["last_stores"]), xytext=(-70, 20), textcoords="offset points", fontsize=8,
                   arrowprops=dict(arrowstyle="-", color=C_GRAY))
    ax[0].set_xticks(x[::2])
    ax[0].set_xticklabels(g.index[::2], rotation=45, fontsize=7.5)
    ax[0].set_title("편의점 수 (실측 90개 상권 합계)", fontsize=10)
    vals = [res["store_yoy"] * 100, res["sales_yoy"] * 100, res["per_store_yoy"] * 100]
    ax[1].bar(["점포 수", "상권 매출", "점포당 매출"], vals, color=[C_RED, C_ORANGE, C_GREEN])
    for i, v in enumerate(vals):
        ax[1].text(i, v + (0.3 if v >= 0 else -0.3), f"{v:+.1f}%", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    ax[1].axhline(0, color="#444", linewidth=0.8)
    ax[1].set_title(f"전년 동기({prev}) 대비 {last} 변화율", fontsize=10)
    ax[1].set_ylim(min(vals) - 2, max(vals) + 2)
    save(fig, "t01_market.png")
    return res


# ---------------------------------------------------------------- 2 결제 건수
def payments(f: pd.DataFrame) -> dict:
    a = f[(f["구"].isin(["강남구", "송파구"])) & (f["트랙"] == "실측")].copy()
    a["tx"] = a["tx_per_store_day"]
    a["sales_man"] = a["spq_month"] / 1e4
    a["ticket"] = a["spq_month"] / (a["tx"] * 30)
    lx, ly = np.log(a["tx"]), np.log(a["spq_month"])
    share = float(np.cov(lx, ly)[0, 1] / np.var(ly, ddof=1))
    # 상위 20% / 하위 20% = 80백분위 / 20백분위 값
    res = {
        "variance_share": share, "rank_corr": float(stats.spearmanr(a["tx"], a["spq_month"])[0]),
        "top20_tx": float(a["tx"].quantile(0.8)), "bottom20_tx": float(a["tx"].quantile(0.2)),
        "top20_sales": float(a["sales_man"].quantile(0.8)), "bottom20_sales": float(a["sales_man"].quantile(0.2)),
        "by_type": a.groupby("상권구분").agg(n=("tx", "size"), tx=("tx", "median"), sales=("sales_man", "median"), ticket=("ticket", "median")),
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    colors = {"발달상권": C_BLUE, "골목상권": C_ORANGE, "전통시장": C_GREEN, "관광특구": C_GRAY}
    for t, c in colors.items():
        m = a["상권구분"] == t
        ax.scatter(a.loc[m, "tx"], a.loc[m, "sales_man"], s=26, color=c, alpha=0.85, label=f"{t} ({int(m.sum())})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("점포 1곳의 하루 결제 건수 (로그축)", fontsize=9)
    ax.set_ylabel("점포당 월매출(만원, 로그축)", fontsize=9)
    ax.set_title(f"매출은 결제 건수가 가른다 (순위상관 {res['rank_corr']:+.2f}, 매출 분산의 {share * 100:.0f}%를 설명)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    save(fig, "t02_payments.png")
    return res


# ---------------------------------------------------------------- 3 군집 k=2
def clusters(f: pd.DataFrame) -> dict:
    a = f[(f["구"].isin(["강남구", "송파구"])) & (f["트랙"] == "실측")].copy()
    a["wrk_share"] = a["TOT_WRC_POPLTN_CO"] / a["population_total"]
    variants = {
        "log": pd.DataFrame({"wrk_share": a["wrk_share"], "night_fl": a["night_flpop_ratio"], "pop": np.log1p(a["population_total"]),
                              "stores": np.log1p(a["cvs_store_avg"]), "acc": np.log1p(a["accessibility"])}),
        "raw": pd.DataFrame({"wrk_share": a["wrk_share"], "night_fl": a["night_flpop_ratio"], "pop": a["population_total"],
                              "stores": a["cvs_store_avg"], "acc": a["accessibility"]}),
    }
    best = None
    for name, X in variants.items():
        Z = StandardScaler().fit_transform(X.fillna(X.median()))
        km = KMeans(2, n_init=50, random_state=0).fit(Z)
        sil = float(silhouette_score(Z, km.labels_))
        sizes = np.bincount(km.labels_)
        if best is None or abs(sil - 0.37) < abs(best[1] - 0.37):
            best = (name, sil, km.labels_, sizes, X)
    name, sil, labels, sizes, X = best
    lab = pd.Series(labels, index=a.index)
    hi = lab[a["wrk_share"] > a.groupby(lab)["wrk_share"].transform("median").median()].mode().iloc[0] if False else None
    big = a.groupby(lab)["wrk_share"].median().idxmax()
    a["cluster"] = np.where(lab == big, "직장·번화형", "주거·골목형")
    prof = a.groupby("cluster").agg(n=("cluster", "size"), spq=("spq_month", lambda s: s.median() / 1e4), wrk=("wrk_share", "median"),
                                   stores=("cvs_store_avg", "mean"), pop=("population_total", "median"), night=("night_sales_ratio", "mean"),
                                   day=("daytime_sales_ratio", "mean"), stab=("stability", "median"), acc=("accessibility", "median"))
    res = {"variant": name, "silhouette": sil, "profile": prof, "series": a["cluster"]}
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
    cols = {"직장·번화형": C_BLUE, "주거·골목형": C_ORANGE}
    for k, c in cols.items():
        ax[0].scatter(a.loc[a.cluster == k, "wrk_share"] * 100, a.loc[a.cluster == k, "night_sales_ratio"] * 100, s=24, color=c, label=f"{k} ({int((a.cluster == k).sum())})", alpha=0.85)
    ax[0].set_xlabel("직장인구 비중(%)", fontsize=9)
    ax[0].set_ylabel("심야 매출 비중(%)", fontsize=9)
    ax[0].legend(frameon=False, fontsize=8)
    ax[0].set_title("군집 분포", fontsize=10)
    ks = list(cols)
    ax[1].bar(ks, [prof.loc[k, "spq"] for k in ks], color=[cols[k] for k in ks])
    for i, k in enumerate(ks):
        ax[1].text(i, prof.loc[k, "spq"], f"{prof.loc[k, 'spq']:,.0f}", ha="center", va="bottom", fontsize=9)
    ax[1].set_title("점포당 월매출 중앙값(만원)", fontsize=10)
    ax[2].bar(ks, [prof.loc[k, "night"] * 100 for k in ks], color=[cols[k] for k in ks])
    for i, k in enumerate(ks):
        ax[2].text(i, prof.loc[k, "night"] * 100, f"{prof.loc[k, 'night'] * 100:.0f}%", ha="center", va="bottom", fontsize=9)
    ax[2].set_title("심야(21~06시) 매출 비중 평균", fontsize=10)
    save(fig, "t03_clusters.png")
    return res


# ---------------------------------------------------------------- 4 OLS
def ols(f: pd.DataFrame, e: pd.DataFrame) -> dict:
    a = f[(f["구"].isin(["강남구", "송파구"])) & (f["트랙"] == "실측")].join(e[["franchise_ratio"]])
    X = pd.DataFrame({
        "상주인구(log)": np.log(a["TOT_REPOP_CO"].clip(lower=1)), "접근성": a["accessibility"], "편의점 수(log)": np.log(a["cvs_store_avg"]),
        "프랜차이즈 비중": a["franchise_ratio"], "임대료": a["rent_per_m2_kwon"], "직장인구(log)": np.log(a["TOT_WRC_POPLTN_CO"].clip(lower=1)),
        "심야 유동비중": a["night_flpop_ratio"],
    })
    y = np.log(a["spq_month"])
    Z = (X - X.mean()) / X.std()
    yz = (y - y.mean()) / y.std()
    m = sm.OLS(yz, sm.add_constant(Z)).fit(cov_type="HC3")
    coef = pd.DataFrame({"beta": m.params.drop("const"), "p": m.pvalues.drop("const")})
    res = {"r2": float(m.rsquared), "adj_r2": float(m.rsquared_adj), "n": int(m.nobs), "coef": coef}
    c = coef.sort_values("beta")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.barh(c.index, c["beta"], color=[C_BLUE if p < 0.05 else C_LIGHT for p in c["p"]])
    for i, (b, p) in enumerate(zip(c["beta"], c["p"])):
        ax.text(b + (0.01 if b >= 0 else -0.01), i, f"{b:+.2f}", va="center", ha="left" if b >= 0 else "right", fontsize=8)
    ax.axvline(0, color="#444", linewidth=0.8)
    ax.set_title(f"구조 지표 7개로 점포당 매출의 {res['r2'] * 100:.0f}%만 설명 (R² {res['r2']:.2f}, 보정 {res['adj_r2']:.2f}, n={res['n']})\n진한 파랑 = 5% 수준에서 유의 (HC3)", fontsize=9.5)
    save(fig, "t04_ols.png")
    return res


# ---------------------------------------------------------------- 5 심야 유동 vs 매출
def night_flow(f: pd.DataFrame) -> dict:
    a = f[(f["구"].isin(["강남구", "송파구"])) & (f["트랙"] == "실측")]
    rho = float(stats.spearmanr(a["night_flpop_ratio"], a["night_sales_ratio"])[0])
    below = int((a["night_sales_ratio"] < a["night_flpop_ratio"]).sum())
    by = a.groupby("상권구분")[["night_flpop_ratio", "night_sales_ratio"]].median()
    res = {"rho": rho, "below": below, "n": len(a), "median_fl": float(a["night_flpop_ratio"].median()), "median_sales": float(a["night_sales_ratio"].median()), "by_type": by}
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9))
    ax[0].scatter(a["night_sales_ratio"] * 100, a["night_flpop_ratio"] * 100, s=22, color=C_BLUE, alpha=0.8)
    lim = [0, 55]
    ax[0].plot(lim, lim, color=C_GRAY, linestyle="--", linewidth=1)
    ax[0].set_xlabel("심야 매출 비중(%)", fontsize=9)
    ax[0].set_ylabel("심야 유동 비중(%)", fontsize=9)
    ax[0].set_title(f"유동은 밤에 더 많다: {below}/{len(a)}곳이 점선(유동=매출) 위 (순위상관 {rho:+.2f})", fontsize=9.5)
    order = ["발달상권", "골목상권", "전통시장", "관광특구"]
    x = np.arange(len(order))
    ax[1].bar(x - 0.2, by.loc[order, "night_flpop_ratio"] * 100, 0.4, color=C_ORANGE, label="심야 유동 비중")
    ax[1].bar(x + 0.2, by.loc[order, "night_sales_ratio"] * 100, 0.4, color=C_BLUE, label="심야 매출 비중")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(order, fontsize=9)
    ax[1].legend(frameon=False, fontsize=8)
    ax[1].set_title("상권구분별 중앙값(%)", fontsize=10)
    save(fig, "t05_night_flow.png")
    return res


# ---------------------------------------------------------------- 6 순위 하락폭 상관
def rank_drop(a_tbl: pd.DataFrame, f: pd.DataFrame) -> dict:
    d = a_tbl.join(f[["night_flpop_ratio", "TOT_WRC_POPLTN_CO", "population_total", "competition", "spq_month"]])
    d["drop"] = d["rank_roi"] - d["rank_loc"]
    d["wrk_share"] = d["TOT_WRC_POPLTN_CO"] / d["population_total"]
    vars_ = {"심야 매출 비중": "night_sales_ratio", "심야 유동 비중": "night_flpop_ratio", "직장인구 비중": "wrk_share",
             "경쟁밀도": "competition", "점포당 월매출": "spq_month", "상주+직장인구": "population_total"}
    rows = {k: float(stats.spearmanr(d["drop"], d[v])[0]) for k, v in vars_.items()}
    s = pd.Series(rows).sort_values()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(s.index, s.values, color=[C_BLUE if v > 0 else C_ORANGE for v in s.values])
    for i, v in enumerate(s.values):
        ax.text(v + (0.02 if v >= 0 else -0.02), i, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=8.5)
    ax.axvline(0, color="#444", linewidth=0.8)
    ax.set_xlim(-1, 1)
    ax.set_title("A: 입지→ROI 순위 하락폭과의 상관 (+ = ROI에서 더 나빠짐)\n입지는 밤을, ROI는 하루 전체를 본다", fontsize=9.5)
    save(fig, "t06_rank_drop.png")
    return {"corr": s}


# ---------------------------------------------------------------- 7 분기 안정성
def quarterly(panel: pd.DataFrame, a_tbl: pd.DataFrame, b_tbl: pd.DataFrame) -> dict:
    recs = list(a_tbl[(a_tbl["rec_rank"] <= 3)].sort_values("rec_rank").index) + list(b_tbl[(b_tbl["rec_rank"] <= 3)].sort_values("rec_rank").index)
    names = {c: (a_tbl.loc[c, "상권명"] if c in a_tbl.index else b_tbl.loc[c, "상권명"]) for c in recs}
    last4 = sorted(panel["STDR_YYQU_CD"].unique())[-4:]
    p = panel[panel["TRDAR_CD"].isin(recs) & panel["STDR_YYQU_CD"].isin(last4)]
    m = (p.pivot(index="TRDAR_CD", columns="quarter", values="sales_per_store_q") / 3 / 1e4).reindex(recs)
    m["변동폭"] = (m.max(axis=1) - m.min(axis=1)) / m.mean(axis=1)
    m.index = [names[c] for c in m.index]
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    q = [c for c in m.columns if c != "변동폭"]
    for nm, r in m.iterrows():
        ax.plot(q, r[q].values, marker="o", markersize=4, label=f"{nm} ({r['변동폭'] * 100:.0f}%)")
    ax.set_ylabel("점포당 월매출(만원)", fontsize=9)
    ax.set_title("추천 6곳의 최근 4분기 점포당 월매출 (범례 = 변동폭)", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    save(fig, "t07_quarterly.png")
    return {"table": m}


# ---------------------------------------------------------------- 8 민감도
def hours_table(pa, pb) -> pd.DataFrame:
    sa, sb = team_score(pa, T.WEIGHTS_A_TEAM), team_score(pb, T.WEIGHTS_B_TEAM)
    rows = []
    for h in [6, 12, 18]:
        rows.append({"알바 하루 근무시간": h, "A 흑자": int(evaluate(sa, "A", hours=h)["profit"].sum()), "B 흑자": int(evaluate(sb, "B", hours=h)["profit"].sum())})
    return pd.DataFrame(rows)


def deposit_extra() -> dict:
    """본부보증금(전대·예치)이 필요현금과 별개라고 볼 때 더할 금액. 세븐일레븐 가맹보증금은 담보와 같은 금액이라 더하지 않는다."""
    fr = load_franchise_terms()
    fr["본부보증금_최소_만원"] = pd.to_numeric(fr["본부보증금_최소_만원"], errors="coerce")
    m = fr["본부보증금_종류"].isin(["전대보증금", "예치보증금"]) & fr["본부보증금_최소_만원"].notna()
    out = {}
    for _, r in fr[m].iterrows():
        out[r["브랜드"] + " " + r["가맹타입"]] = float(r["본부보증금_최소_만원"])
    return out


def sensitivity(pa, pb) -> dict:
    sa, sb = team_score(pa, T.WEIGHTS_A_TEAM), team_score(pb, T.WEIGHTS_B_TEAM)
    own_a = T.HOURLY_WAGE * T.DAYS_PER_MONTH * (24 - T.HOURS["A"]) / 1e4      # 점주 본인 노동(최저임금, 주휴 제외)
    own_b = T.HOURLY_WAGE * T.DAYS_PER_MONTH * (24 - T.HOURS["B"]) / 1e4
    scen = {
        "기본": {},
        "총이익률 30%": {"gm": 0.30},
        "점주 노동 환산": {"owner": True},
        "GS2 전대보증금 3,000만원 포함(발표 방식)": {"extra": {"GS25 GS2": 3000.0}},
        "전·예치보증금이 있는 타입 모두 포함(확장)": {"extra": deposit_extra()},
    }
    res = {}
    for persona, sc, own in [("A", sa, own_a), ("B", sb, own_b)]:
        rows = []
        for name, kw in scen.items():
            k = {}
            if "gm" in kw:
                k["gm"] = kw["gm"]
            if kw.get("owner"):
                k["owner_labor_man"] = own
            if "extra" in kw:
                k["extra_cash"] = kw["extra"]
            ev = evaluate(sc, persona, **k)
            ev = ev.join(sc[["상권명", "outlier"]])
            pick = ev[ev["profit"] & ~ev["outlier"]].sort_values("roi", ascending=False).head(3)
            rows.append({"시나리오": name, "흑자 상권": int(ev["profit"].sum()),
                         "추천 3곳": " · ".join(pick["상권명"]), "ROI": " / ".join(f"{v * 100:.0f}%" for v in pick["roi"]),
                         "타입": " / ".join(pick["type"].str.replace(r"\(.*?\)", "", regex=True))})
        res[persona] = pd.DataFrame(rows)
        res[f"owner_{persona}"] = own
    res["hours"] = hours_table(pa, pb)
    res["top3_freq_A"] = top3_frequency(sa, T.WEIGHTS_A_TEAM)
    res["top3_freq_B"] = top3_frequency(sb, T.WEIGHTS_B_TEAM)
    res["score_A"], res["score_B"] = sa, sb
    return res


def figs_core(a_tbl, b_tbl, sens):
    # 입지 가중치 비교
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6))
    names = {"night_sales_ratio": "심야매출비중", "spq_month": "점포당매출", "night_labor_coverage": "심야인건비배율", "night_flpop_ratio": "심야유동비중", "population_total": "상주직장인구",
             "stability": "상권안정성", "competition": "경쟁밀도", "apt_households": "아파트세대수", "daytime_sales_ratio": "주간매출비중", "accessibility": "대중교통접근성"}
    for a_, (W, ttl) in zip(ax, [(T.WEIGHTS_A_TEAM, "A 직장인 부업의 가중치"), (T.WEIGHTS_B_TEAM, "B 은퇴부부의 가중치")]):
        ks = list(W)
        a_.barh([names[k] for k in ks][::-1], [W[k] for k in ks][::-1], color=[C_RED if W[k] < 0 else C_BLUE for k in ks][::-1])
        for i, k in enumerate(ks[::-1]):
            a_.text(W[k] + (0.005 if W[k] >= 0 else -0.005), i, f"{W[k]:+.2f}", va="center", ha="left" if W[k] >= 0 else "right", fontsize=8)
        a_.axvline(0, color="#444", linewidth=0.8)
        a_.set_xlim(-0.3, 0.3)
        a_.set_title(ttl, fontsize=10)
    save(fig, "t08_weights.png")

    # ROI 막대 A/B
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0), gridspec_kw={"width_ratios": [1.1, 1]})
    pick = a_tbl[a_tbl["profit"] & ~a_tbl["outlier"]].sort_values("roi", ascending=False).head(10).iloc[::-1]
    ax[0].barh(pick["상권명"], pick["roi"] * 100, color=C_BLUE)
    for i, (r, ip) in enumerate(zip(pick["roi"] * 100, pick["rank_loc"])):
        ax[0].text(r + 5, i, f"{r:.0f}%  (입지 {ip}위)", va="center", fontsize=8)
    ax[0].set_xlim(0, pick["roi"].max() * 100 * 1.35)
    ax[0].set_title("A: ROI 상위 10곳 (GS25 GS2, 알바 18h, 이상값 제외)", fontsize=9.5)
    ax[0].tick_params(axis="y", labelsize=8)
    b = b_tbl.sort_values("roi")
    ax[1].barh(b["상권명"], b["roi"] * 100, color=[C_GREEN if p else C_RED for p in b["profit"]])
    for i, (r, p) in enumerate(zip(b["roi"] * 100, b["profit"])):
        ax[1].text(r + (5 if r >= 0 else -5), i, f"{r:.0f}%", va="center", ha="left" if r >= 0 else "right", fontsize=8)
    ax[1].axvline(0, color="#444", linewidth=0.8)
    ax[1].set_title("B: 후보 10곳의 ROI (알바 12h)", fontsize=9.5)
    ax[1].set_xlim(-60, b["roi"].max() * 100 * 1.25)
    ax[1].tick_params(axis="y", labelsize=8)
    save(fig, "t09_roi_bars.png")

    # 손익분기 분해(장지역) + 알바 시간별 흑자 수
    r = a_tbl[a_tbl["상권명"] == "장지역(가든파이브)"].iloc[0]
    sales = r["sales_man"]
    gross = sales * T.GROSS_MARGIN
    owner = gross * r["share"]
    wage = T.monthly_wage_man(T.HOURS["A"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    labels = ["월매출", "총이익(25%)", f"점주 몫({r['share'] * 100:.0f}%)", "알바비 뺀\n월수익"]
    vals = [sales, gross, owner, owner - wage]
    ax[0].bar(labels, vals, color=[C_LIGHT, C_GRAY, C_BLUE, C_GREEN])
    for i, v in enumerate(vals):
        ax[0].text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=9)
    ax[0].set_title(f"장지역(가든파이브) 월 손익 분해(만원)\n원가 {sales - gross:,.0f} 차감 · 본부 몫 {gross - owner:,.0f} 차감 · 알바비 {wage:,.0f} 차감", fontsize=8.5)
    h = sens["hours"]
    x = np.arange(len(h))
    ax[1].bar(x - 0.2, h["A 흑자"], 0.4, color=C_BLUE, label="A (90곳 중)")
    ax[1].bar(x + 0.2, h["B 흑자"], 0.4, color=C_ORANGE, label="B 후보 (10곳 중)")
    for i, (va, vb) in enumerate(zip(h["A 흑자"], h["B 흑자"])):
        ax[1].text(i - 0.2, va, str(va), ha="center", va="bottom", fontsize=9)
        ax[1].text(i + 0.2, vb, str(vb), ha="center", va="bottom", fontsize=9)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([f"{v}시간" for v in h["알바 하루 근무시간"]])
    ax[1].legend(frameon=False, fontsize=8)
    ax[1].set_title("알바 하루 근무시간별 흑자 상권 수", fontsize=10)
    save(fig, "t10_cost.png")

    # 입지 vs ROI 순위 기울기
    def dodge(vals, gap):
        vals = np.asarray(vals, dtype=float)
        order = np.argsort(vals)
        y = vals[order].copy()
        for i in range(1, len(y)):
            y[i] = max(y[i], y[i - 1] + gap)
        out = np.empty_like(y)
        out[order] = y
        return out

    def slope(ax_, tbl, title, k):
        t = tbl.copy()
        sel = t[(t["rank_loc"] <= k) | ((t["rank_roi"] <= k))]
        ax_.set_xlim(-0.6, 1.6)
        gap = 1.0 if len(t) <= 12 else 2.4
        ly = dodge(sel["rank_loc"].values, gap)
        ry = dodge(sel["rank_roi"].values, gap)
        for (_, r_), yl, yr in zip(sel.iterrows(), ly, ry):
            col = C_GREEN if r_["rank_roi"] < r_["rank_loc"] else (C_RED if r_["rank_roi"] > r_["rank_loc"] else C_GRAY)
            ax_.plot([0, 1], [r_["rank_loc"], r_["rank_roi"]], color=col, linewidth=1.5)
            ax_.text(-0.05, yl, f"{int(r_['rank_loc'])} {r_['상권명']}", ha="right", va="center", fontsize=7.5)
            ax_.text(1.05, yr, f"{int(r_['rank_roi'])} {r_['상권명']}", ha="left", va="center", fontsize=7.5)
        ax_.invert_yaxis()
        ax_.set_xticks([0, 1])
        ax_.set_xticklabels(["입지 순위", "ROI 순위"])
        ax_.set_yticks([])
        ax_.spines[:].set_visible(False)
        ax_.set_title(title, fontsize=10)
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 8.5), gridspec_kw={"width_ratios": [1.1, 1]})
    slope(ax[0], a_tbl, "A: 입지 상위10 ∪ ROI 상위10 (90곳 중)", 10)
    slope(ax[1], b_tbl, "B: 후보 10곳 (적자는 ROI 아래쪽)", 10)
    save(fig, "t11_rank_shift.png")

    # 가중치 흔들기
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for a_, key, sc, ttl in [(ax[0], "top3_freq_A", sens["score_A"], "A: 가중치 ±30%에서 상위 3위에 든 비율"), (ax[1], "top3_freq_B", sens["score_B"], "B: 가중치 ±30%에서 상위 3위에 든 비율")]:
        s = sens[key].sort_values(ascending=False).head(7)
        nm = sc.loc[s.index, "상권명"]
        a_.barh(nm[::-1], s.values[::-1] * 100, color=C_BLUE)
        for i, v in enumerate(s.values[::-1]):
            a_.text(v * 100 + 1, i, f"{v * 100:.0f}%", va="center", fontsize=8.5)
        a_.set_xlim(0, 115)
        a_.set_title(ttl, fontsize=9.5)
        a_.tick_params(axis="y", labelsize=8.5)
    save(fig, "t12_top3_freq.png")


def maps(sens) -> None:
    """팀 가중치로 만든 입지 점수 지도(상위 3곳 표시). 라벨 지도 함수는 analysis_pipeline.plot_labeled_map을 쓴다."""
    import shutil

    import analysis_pipeline as AP
    from data_paths import load_trade_area_boundary

    bdf, shapes = load_trade_area_boundary()
    for key, name, title, offs in [
        ("score_A", "t13_map_A.png", "A 입지 점수(팀 가중치): 강남·송파 실측 90곳 (빨간 테두리 = 상위 3곳)", [(-120, 55), (-110, -65), (95, -60)]),
        ("score_B", "t14_map_B.png", "B 입지 점수(팀 가중치): 도곡동 반경 1.5km 실측 10곳 (빨간 테두리 = 상위 3곳)", [(-130, 40), (60, -75), (-90, -80)]),
    ]:
        sc = sens[key]
        AP.plot_labeled_map(bdf, shapes, sc, title, name, sc.head(3), offs)
        shutil.move(str(AP.FIG / name), str(OUT / name))


def main() -> None:
    e = pd.read_csv(ROOT / "eda_output" / "analysis" / "feature_extended.csv", dtype={"TRDAR_CD": str}).set_index("TRDAR_CD")
    f = pd.read_csv(OUT / "feature_team.csv", dtype={"TRDAR_CD": str}).set_index("TRDAR_CD")
    panel = pd.read_csv(OUT / "quarterly_panel.csv", dtype={"TRDAR_CD": str, "STDR_YYQU_CD": str})
    pa, pb = pools(f)
    a_tbl = pd.read_csv(OUT / "roi_A.csv", dtype={"TRDAR_CD": str}).set_index("TRDAR_CD")
    b_tbl = pd.read_csv(OUT / "roi_B.csv", dtype={"TRDAR_CD": str}).set_index("TRDAR_CD")
    R = {
        "market": market(panel, set(pa.index)),
        "payments": payments(f),
        "clusters": clusters(f),
        "ols": ols(f, e),
        "night": night_flow(f),
        "drop": rank_drop(a_tbl, f),
        "quarterly": quarterly(panel, a_tbl, b_tbl),
    }
    sens = sensitivity(pa, pb)
    R["sens"] = sens
    figs_core(a_tbl, b_tbl, sens)
    maps(sens)
    with open(OUT / "extra_results.pkl", "wb") as fh:
        pickle.dump(R, fh)
    m, p, c, o, n = R["market"], R["payments"], R["clusters"], R["ols"], R["night"]
    print(f"시장: 정점 {m['peak']} {m['peak_stores']:,} -> {m['last']} {m['last_stores']:,} ({m['from_peak'] * 100:+.1f}%), 점포 {m['store_yoy'] * 100:+.1f}% 매출 {m['sales_yoy'] * 100:+.1f}% 점포당 {m['per_store_yoy'] * 100:+.1f}%")
    print(f"결제: 분산 {p['variance_share'] * 100:.0f}% 순위상관 {p['rank_corr']:+.2f} 상위20 {p['top20_tx']:.0f} 하위20 {p['bottom20_tx']:.0f} / 매출 {p['top20_sales']:,.0f} {p['bottom20_sales']:,.0f}")
    print(p["by_type"].round(1))
    print(f"군집: {c['variant']} 실루엣 {c['silhouette']:.2f}"); print(c["profile"].round(2))
    print(f"OLS: R2 {o['r2']:.2f} adj {o['adj_r2']:.2f}"); print(o["coef"].round(2))
    print(f"심야: rho {n['rho']:+.2f} 유동>매출 {n['below']}/{n['n']}")
    print(R["drop"]["corr"].round(2))
    print(R["quarterly"]["table"].round(2))
    for k in ["A", "B"]:
        print(k); print(sens[k].to_string(index=False))
    print(sens["hours"])
    print(sens["top3_freq_A"].sort_values(ascending=False).head(6).round(2).to_dict())
    print(sens["top3_freq_B"].sort_values(ascending=False).head(6).round(2).to_dict())


if __name__ == "__main__":
    main()
