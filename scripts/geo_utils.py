# -*- coding: utf-8 -*-
"""좌표 변환 및 거리 계산 유틸리티.

상권경계 shapefile은 EPSG:5181(중부원점 TM, 미터 단위)이고, 다른 원천 데이터의
좌표(지하철역/버스정류소/공동주택/상가정보)는 전부 WGS84 경위도다. 짧은 거리(반경
몇 백 m~수 km) 계산은 EPSG:5181 평면 좌표에서 단순 유클리드 거리로 계산하는 편이
정확하고 빠르므로, 이 모듈에서 WGS84 좌표를 전부 EPSG:5181로 변환해 쓴다.
"""
from __future__ import annotations

import numpy as np
from pyproj import Transformer

_TO_WGS84 = Transformer.from_crs("EPSG:5181", "EPSG:4326", always_xy=True)
_FROM_WGS84 = Transformer.from_crs("EPSG:4326", "EPSG:5181", always_xy=True)


def epsg5181_to_wgs84(x, y):
    """상권경계 좌표(EPSG:5181, m) -> (lon, lat) WGS84."""
    return _TO_WGS84.transform(x, y)


def wgs84_to_epsg5181(lon, lat):
    """일반 위경도(WGS84) -> (x, y) EPSG:5181(m). 스칼라/배열 모두 지원."""
    return _FROM_WGS84.transform(lon, lat)


def wgs84_arr_to_5181(lon_arr, lat_arr):
    """WGS84 경도/위도 배열 -> (x배열, y배열) EPSG:5181(m). numpy 배열로 반환."""
    x, y = _FROM_WGS84.transform(np.asarray(lon_arr, dtype=float), np.asarray(lat_arr, dtype=float))
    return np.asarray(x), np.asarray(y)


def count_within_radius(cx: float, cy: float, xs, ys, radius_m: float, weights=None):
    """중심점(cx, cy, EPSG:5181 m) 기준 반경 내 포인트 개수 또는 가중치 합.

    weights=None이면 반경 내 포인트 개수, weights를 주면(예: 세대수) 그 가중합을 돌려준다.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    mask = d <= radius_m
    if weights is None:
        return int(mask.sum())
    return float(np.asarray(weights, dtype=float)[mask].sum())


def min_max_normalize(series, invert: bool = False):
    """0~1로 min-max 정규화. invert=True면 값이 작을수록 좋은 지표(예: 경쟁점포수)를 뒤집는다."""
    lo, hi = series.min(), series.max()
    if hi == lo:
        norm = series * 0 + 0.5  # 전부 같은 값이면 중립값(0.5) 처리
    else:
        norm = (series - lo) / (hi - lo)
    return 1 - norm if invert else norm


def percentile_normalize(series, invert: bool = False):
    """백분위 순위(0~1)로 정규화. 왜도가 큰 변수(점포당매출, 인구 등)에서 min-max보다
    극단값 하나가 전체를 0 근처로 눌러버리는 왜곡이 적다(전처리_기준서.md 9절).
    결측은 결측으로 남긴다(순위 계산에서 빠지고 NaN 유지).
    """
    rank = series.rank(pct=True)
    return 1 - rank if invert else rank


def iqr_outlier_flag(series):
    """IQR(사분위범위) 기준 상/하위 이상값 표시. 삭제하지 않고 표시만 한다(전처리_기준서.md 11절)."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return np.select(
        [series > hi, series < lo],
        ["상위 이상값", "하위 이상값"],
        default="",
    )
