# -*- coding: utf-8 -*-
"""원천 데이터 경로 상수 및 로더 함수 모음.

README.md 권장사항대로 코드값 앞자리 0이 사라지지 않도록 전부 dtype=str로 읽는다.
숫자로 써야 하는 컬럼은 각 분석 스크립트에서 pd.to_numeric으로 필요한 것만 변환한다.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

SEOUL = DATA / "seoul"
DATA_GO_KR = DATA / "data_go_kr"
R_ONE = DATA / "r_one"
MANUAL = DATA / "manual"

# 01/02번(추정매출_상권, 점포_상권)의 SVC_INDUTY_CD_NM 기준 편의점 서비스업종코드.
# data/seoul/01_추정매출_상권 SVC_INDUTY_CD_NM 고유값에서 확인함.
CONVENIENCE_STORE_SVC_INDUTY_CD = "CS300002"

# 14번(소상공인_상가정보)의 indsSclsNm 기준 편의점 소분류명.
CONVENIENCE_STORE_INDS_SCLS_NM = "편의점"


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    kwargs.setdefault("dtype", str)
    kwargs.setdefault("encoding", "utf-8-sig")
    return pd.read_csv(path, **kwargs)


# ---- seoul: 서울 열린데이터광장 (서울 전체 범위, 상권/행정동 단위) ----

def load_sales() -> pd.DataFrame:
    """01_추정매출_상권: 분기x상권x서비스업종 단위 추정매출."""
    return _read_csv(SEOUL / "01_추정매출_상권" / "추정매출_상권.csv")


def load_stores() -> pd.DataFrame:
    """02_점포_상권: 2021~2026 연도별 파일을 합쳐 분기x상권x서비스업종 단위 점포 현황으로 반환."""
    files = sorted((SEOUL / "02_점포_상권").glob("점포_상권_*.csv"))
    return pd.concat([_read_csv(f) for f in files], ignore_index=True)


def load_change_index() -> pd.DataFrame:
    """03_상권변화지표_상권."""
    return _read_csv(SEOUL / "03_상권변화지표_상권" / "상권변화지표_상권.csv")


def load_street_pop() -> pd.DataFrame:
    """04_길단위인구_상권: 유동인구."""
    return _read_csv(SEOUL / "04_길단위인구_상권" / "길단위인구_상권.csv")


def load_resident_pop() -> pd.DataFrame:
    """05_상주인구_상권."""
    return _read_csv(SEOUL / "05_상주인구_상권" / "상주인구_상권.csv")


def load_worker_pop() -> pd.DataFrame:
    """06_직장인구_상권."""
    return _read_csv(SEOUL / "06_직장인구_상권" / "직장인구_상권.csv")


def load_facilities() -> pd.DataFrame:
    """07_집객시설_상권."""
    return _read_csv(SEOUL / "07_집객시설_상권" / "집객시설_상권.csv")


def load_living_pop() -> pd.DataFrame:
    """08_생활인구_행정동: 상권이 아니라 행정동x일자x시간대 단위."""
    return _read_csv(SEOUL / "08_생활인구_행정동" / "생활인구_행정동.csv")


def load_subway() -> pd.DataFrame:
    """09_지하철역_좌표: 수도권 전철역 전체."""
    return _read_csv(SEOUL / "09_지하철역_좌표" / "지하철역_좌표.csv")


def load_bus_stops() -> pd.DataFrame:
    """10_버스정류소_좌표."""
    return _read_csv(SEOUL / "10_버스정류소_좌표" / "버스정류소_좌표.csv")


def load_trade_area_boundary():
    """11_상권경계: shapefile(zip). (속성 DataFrame, shapes) 튜플로 반환.

    geopandas가 없는 환경이라 pyshp(shapefile 모듈)를 사용한다.
    """
    import shapefile

    zip_path = SEOUL / "11_상권경계" / "서울시_상권분석서비스_영역-상권.zip"
    sf = shapefile.Reader(str(zip_path), encoding="utf-8")
    records = [r.as_dict() for r in sf.records()]
    df = pd.DataFrame(records)
    shapes = sf.shapes()
    return df, shapes


# ---- data_go_kr: 공공데이터포털 (강남/송파만) ----

def load_apt_basic() -> pd.DataFrame:
    """12_공동주택_기본정보."""
    return _read_csv(DATA_GO_KR / "12_13_18_공동주택" / "공동주택_기본정보.csv")


def load_apt_list() -> pd.DataFrame:
    """13_공동주택_단지목록."""
    return _read_csv(DATA_GO_KR / "12_13_18_공동주택" / "공동주택_단지목록.csv")


def load_apt_geo() -> pd.DataFrame:
    """18_공동주택_좌표변환."""
    return _read_csv(DATA_GO_KR / "12_13_18_공동주택" / "공동주택_좌표변환.csv")


def load_stores_detail() -> pd.DataFrame:
    """14_소상공인_상가정보: 강남/송파 개별 상가업소 (중복 등록 가능성 있음, README 참고)."""
    return _read_csv(DATA_GO_KR / "14_소상공인_상가정보" / "상가정보_강남송파.csv")


def load_convenience_stores(dedup: bool = True) -> pd.DataFrame:
    """14번 중 편의점(indsSclsNm == '편의점')만 추출.

    같은 점포가 점주명의/브랜드명의로 두 번 등록된 경우가 있어(README 참고),
    dedup=True면 상호명+도로명주소 기준 중복에서 bizesId가 가장 작은(먼저 등록된) 행만 남긴다.
    """
    df = load_stores_detail()
    cvs = df[df["indsSclsNm"] == CONVENIENCE_STORE_INDS_SCLS_NM].copy()
    if dedup:
        cvs = cvs.sort_values("bizesId").drop_duplicates(subset=["bizesNm", "rdnmAdr"], keep="first")
    return cvs


def filter_convenience(df: pd.DataFrame) -> pd.DataFrame:
    """01/02번처럼 SVC_INDUTY_CD 컬럼이 있는 데이터프레임을 편의점 업종만으로 좁힌다."""
    return df[df["SVC_INDUTY_CD"] == CONVENIENCE_STORE_SVC_INDUTY_CD].copy()


# ---- r_one: 한국부동산원 (전국 권역 단위, long format) ----

def load_rent() -> pd.DataFrame:
    """15_임대료_소규모상가."""
    return _read_csv(R_ONE / "15_임대료_소규모상가" / "임대료_소규모상가.csv")


def load_floor_rent_ratio() -> pd.DataFrame:
    """16_층별임대료_효용비율_소규모상가."""
    return _read_csv(R_ONE / "16_층별임대료_효용비율_소규모상가" / "층별임대료_효용비율_소규모상가.csv")


def load_vacancy() -> pd.DataFrame:
    """17_공실률_소규모상가."""
    return _read_csv(R_ONE / "17_공실률_소규모상가" / "공실률_소규모상가.csv")


# ---- manual: 수작업 수집 ----

def load_franchise_terms() -> pd.DataFrame:
    """19_가맹조건_브랜드별."""
    return _read_csv(MANUAL / "19_가맹조건" / "가맹조건_브랜드별.csv")


def load_opex_baseline() -> pd.DataFrame:
    """20_운영비_기준값."""
    return _read_csv(MANUAL / "20_운영비기준값" / "운영비_기준값.csv")


# 프로파일링 등 전체 파일을 순회할 때 쓰는 등록부.
# shapefile(11번)은 DataFrame 단일 반환이 아니라서 별도 취급한다.
DATASETS = {
    "01_추정매출_상권": load_sales,
    "02_점포_상권_전체연도": load_stores,
    "03_상권변화지표_상권": load_change_index,
    "04_길단위인구_상권": load_street_pop,
    "05_상주인구_상권": load_resident_pop,
    "06_직장인구_상권": load_worker_pop,
    "07_집객시설_상권": load_facilities,
    "08_생활인구_행정동": load_living_pop,
    "09_지하철역_좌표": load_subway,
    "10_버스정류소_좌표": load_bus_stops,
    "12_공동주택_기본정보": load_apt_basic,
    "13_공동주택_단지목록": load_apt_list,
    "18_공동주택_좌표변환": load_apt_geo,
    "14_소상공인_상가정보": load_stores_detail,
    "15_임대료_소규모상가": load_rent,
    "16_층별임대료_효용비율_소규모상가": load_floor_rent_ratio,
    "17_공실률_소규모상가": load_vacancy,
    "19_가맹조건_브랜드별": load_franchise_terms,
    "20_운영비_기준값": load_opex_baseline,
}


def gangnam_songpa_trdar_codes() -> set[str]:
    """상권경계 shapefile에서 강남구/송파구 소재 상권코드(TRDAR_CD) 집합을 뽑는다.

    01~07번(서울 전체 범위) 데이터를 강남/송파로 좁힐 때 이 코드로 TRDAR_CD를 필터링한다.
    """
    boundary_df, _ = load_trade_area_boundary()
    mask = boundary_df["SIGNGU_CD_"].isin(["강남구", "송파구"])
    return set(boundary_df.loc[mask, "TRDAR_CD"])
