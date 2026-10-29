# -*- coding: utf-8 -*-
"""발표 PDF에서 장표 본문(제목·출처 줄 제외)을 잘라 eda_output/deck/slideNN.png로 저장한다.

사용: python crop_deck_slides.py "경로/편의점_입지추천_최종발표.pdf"
README 14절의 그림이 이 이미지들이다. 장표는 코드로 만든 그림이 아니라 발표 자료의 화면이다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image

from data_paths import ROOT
from readme_deck_section import SLIDES

OUT = ROOT / "eda_output" / "deck"
SCALE = 2.5          # PDF 1쪽(720x405pt)을 1800x1012px로 렌더링
TOP, BOTTOM = 200, 895   # 렌더링 기준 본문 영역(제목 아래 ~ 출처 줄 위)
WIDTH = 1400


def main(pdf_path: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pdf = pdfium.PdfDocument(pdf_path)
    for n, *_ in SLIDES:
        im = pdf[n - 1].render(scale=SCALE).to_pil().convert("RGB")
        crop = im.crop((0, TOP, im.size[0], BOTTOM))
        crop = crop.resize((WIDTH, int(crop.size[1] * WIDTH / crop.size[0])), Image.LANCZOS)
        crop.save(OUT / f"slide{n:02d}.png", optimize=True)
    print(f"{len(SLIDES)} slides -> {OUT}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("사용: python crop_deck_slides.py <발표 PDF 경로>")
    main(sys.argv[1])
