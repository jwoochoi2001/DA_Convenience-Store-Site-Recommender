# -*- coding: utf-8 -*-
"""전체 원천 데이터 1차 프로파일링.

각 파일에 대해 행/열 수, 메모리 사용량, 결측치, 수치형 컬럼 기술통계,
소수 카테고리 컬럼의 상위값을 뽑아 eda_output/profile_report.md 에 저장한다.
용량이 큰 파일도 있으므로 데이터프레임은 한 번에 하나씩만 메모리에 올리고 바로 해제한다.
"""
from __future__ import annotations

import gc
from datetime import datetime

import pandas as pd

from data_paths import DATASETS, ROOT, gangnam_songpa_trdar_codes, load_trade_area_boundary

OUT_DIR = ROOT / "eda_output"
OUT_DIR.mkdir(exist_ok=True)
REPORT_PATH = OUT_DIR / "profile_report.md"

# 값 종류가 이 개수 이하인 object 컬럼만 "카테고리 컬럼"으로 보고 상위값을 출력한다.
CATEGORICAL_NUNIQUE_LIMIT = 30
TOP_N_MISSING = 15
TOP_N_CATEGORY_VALUES = 10


def numeric_coercible_columns(df: pd.DataFrame) -> dict[str, pd.Series]:
    """문자열로 읽었지만 숫자로 변환 가능한 컬럼만 골라 변환된 Series를 돌려준다."""
    result = {}
    for col in df.columns:
        s = df[col]
        non_null = s.notna().sum()
        if non_null == 0:
            continue
        converted = pd.to_numeric(s, errors="coerce")
        # 원래 비어있지 않던 값의 95% 이상이 숫자로 변환되면 수치형 컬럼으로 취급
        if converted.notna().sum() >= non_null * 0.95:
            result[col] = converted
    return result


def profile_dataframe(name: str, df: pd.DataFrame, lines: list[str]) -> None:
    n_rows, n_cols = df.shape
    mem_mb = df.memory_usage(deep=True).sum() / 1024 / 1024

    lines.append(f"## {name}")
    lines.append("")
    lines.append(f"- 행 수: {n_rows:,}  /  열 수: {n_cols}  /  메모리 사용량: {mem_mb:,.1f} MB")
    lines.append(f"- 완전 중복행 수: {df.duplicated().sum():,}")

    # 결측치
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing):
        lines.append("")
        lines.append(f"### 결측치 (상위 {TOP_N_MISSING}개 컬럼)")
        lines.append("")
        lines.append("| 컬럼 | 결측 수 | 결측률 |")
        lines.append("|---|---:|---:|")
        for col, cnt in missing.head(TOP_N_MISSING).items():
            lines.append(f"| {col} | {cnt:,} | {cnt / n_rows:.1%} |")
    else:
        lines.append("")
        lines.append("### 결측치 없음")

    # 수치형 기술통계
    numeric = numeric_coercible_columns(df)
    if numeric:
        lines.append("")
        lines.append("### 수치형 컬럼 기술통계")
        lines.append("")
        lines.append("| 컬럼 | count | mean | std | min | 25% | 50% | 75% | max |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for col, s in numeric.items():
            desc = s.describe()
            lines.append(
                f"| {col} | {desc['count']:,.0f} | {desc['mean']:,.2f} | {desc['std']:,.2f} "
                f"| {desc['min']:,.2f} | {desc['25%']:,.2f} | {desc['50%']:,.2f} "
                f"| {desc['75%']:,.2f} | {desc['max']:,.2f} |"
            )

    # 카테고리형 상위값 (수치형으로 이미 잡힌 컬럼은 제외)
    cat_cols = [c for c in df.columns if c not in numeric and df[c].nunique(dropna=True) <= CATEGORICAL_NUNIQUE_LIMIT]
    if cat_cols:
        lines.append("")
        lines.append("### 카테고리 컬럼 (고유값 30개 이하) 상위값")
        for col in cat_cols:
            vc = df[col].value_counts(dropna=False).head(TOP_N_CATEGORY_VALUES)
            lines.append("")
            lines.append(f"**{col}** (고유값 {df[col].nunique(dropna=True)}개)")
            lines.append("")
            lines.append("| 값 | 건수 |")
            lines.append("|---|---:|")
            for val, cnt in vc.items():
                val_display = "(NaN)" if pd.isna(val) else str(val)
                lines.append(f"| {val_display} | {cnt:,} |")

    lines.append("")


def profile_boundary(lines: list[str]) -> None:
    boundary_df, shapes = load_trade_area_boundary()
    lines.append("## 11_상권경계 (shapefile)")
    lines.append("")
    lines.append(f"- 폴리곤(상권) 수: {len(boundary_df):,}")
    lines.append("")
    lines.append("### 구(SIGNGU_CD_)별 상권 개수")
    lines.append("")
    lines.append("| 구 | 상권 수 |")
    lines.append("|---|---:|")
    for gu, cnt in boundary_df["SIGNGU_CD_"].value_counts().items():
        lines.append(f"| {gu} | {cnt:,} |")
    lines.append("")
    lines.append("### 상권구분(TRDAR_SE_1)별 개수")
    lines.append("")
    lines.append("| 상권구분 | 상권 수 |")
    lines.append("|---|---:|")
    for se, cnt in boundary_df["TRDAR_SE_1"].value_counts().items():
        lines.append(f"| {se} | {cnt:,} |")
    lines.append("")


def profile_scope_check(lines: list[str]) -> None:
    codes = gangnam_songpa_trdar_codes()
    lines.append("## 강남/송파 범위 확인 (seoul 01~07번용)")
    lines.append("")
    lines.append(
        f"- 상권경계 기준 강남구+송파구 소재 상권코드(TRDAR_CD) 수: **{len(codes):,}개** "
        "(서울 전체 1,650개 중 일부)"
    )
    lines.append(
        "- seoul/01~07번 파일을 강남·송파로 좁히려면 "
        "`df[df['TRDAR_CD'].isin(gangnam_songpa_trdar_codes())]` 로 필터링하면 된다."
    )
    lines.append("")


def main() -> None:
    lines: list[str] = []
    lines.append("# 원천 데이터 프로파일링 리포트")
    lines.append("")
    lines.append(f"생성 시각: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")

    profile_scope_check(lines)
    profile_boundary(lines)

    for name, loader in DATASETS.items():
        print(f"[profile_all] loading {name} ...")
        df = loader()
        profile_dataframe(name, df, lines)
        del df
        gc.collect()

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"[profile_all] done -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
