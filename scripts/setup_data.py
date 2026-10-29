# -*- coding: utf-8 -*-
"""저장소의 data_bundle/ 조각을 합쳐 data/ 폴더를 복원한다.

사용: python scripts/setup_data.py
조각의 SHA256을 확인한 뒤 합치고, 합친 zip을 저장소 루트에 풀어 data/를 만든다.
이미 data/가 있으면 건드리지 않는다(--force로 덮어쓰기).
"""
from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "data_bundle"
DATA = ROOT / "data"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if DATA.is_dir() and any(DATA.iterdir()) and "--force" not in sys.argv:
        print(f"{DATA} 가 이미 있다. 다시 풀려면 --force")
        return
    sums = {}
    for line in (BUNDLE / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        sums[name] = digest
    parts = sorted(p for p in BUNDLE.glob("data.zip.part*"))
    if not parts:
        raise SystemExit("data_bundle/ 에 조각이 없다. 저장소를 다시 받아 보라.")
    for p in parts:
        if sha(p) != sums[p.name]:
            raise SystemExit(f"{p.name} 의 SHA256이 다르다(손상). git pull 후 다시 시도하라.")
    tmp = BUNDLE / "data.zip"
    with open(tmp, "wb") as out:
        for p in parts:
            out.write(p.read_bytes())
    if sha(tmp) != sums["data.zip"]:
        tmp.unlink()
        raise SystemExit("합친 zip의 SHA256이 다르다.")
    with zipfile.ZipFile(tmp) as z:
        z.extractall(ROOT)
    tmp.unlink()
    print(f"완료: {DATA} ({sum(1 for _ in DATA.rglob('*') if _.is_file())}개 파일)")


if __name__ == "__main__":
    main()
