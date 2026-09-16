"""인물 키 상한 + ScANT 저자 격리로 코퍼스를 재빌드한다 — **오프라인**.

왜 오프라인이 가능한가: `pg_shorts_dev3_nocap_nofilter.jsonl` 은 이미
  · dev-3 격리 · 길이 30–900 · 중앙 어휘수 ≥5 · (작가,제목) 중복 제거
가 적용됐고 **상한과 다운로드 컷만 없다**(매니페스트 `per_author_cap: 9999`,
`download_cut: null`). 따라서 Gutenberg 재다운로드 없이 컷·격리·상한만 다시 걸면 된다.

고치는 것 (2026-08-03):
  ① 상한 키를 author **문자열** → **인물**(정규화 이름 + 생몰연도)로. 문자열 키는 같은 인물이
     삽화가·서문저자와 함께 표기되면 다른 키가 되어 상한을 여러 번 받는다(실측: 71명 초과, 925편).
  ② ScANT 평가셋 저자를 격리. 격리는 보수적으로 — **해당 인물이 저자로 올라간 모든 레코드**를 뺀다.

⚠️ 생몰연도가 없거나 충돌하면 인물을 자동 병합하지 않는다(동명이인 위험. 실측 사례:
   `Austen, Jane, 1775-1817` vs `Austen, Sidney`).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/eacl_phase1"))
from author_review_list import is_anthology, parse_person  # noqa: E402

POOL = REPO / "data/eacl_phase1/pg_shorts_dev3_nocap_nofilter.jsonl"
FINAL_MF = REPO / "data/eacl_phase1/pg_shorts_final_cap20_manifest.json"
DOWNLOADS = REPO / "data/eacl_phase1/pg_downloads.json"

# ScANT(SceneML) 평가 텍스트의 저자 — 인물 키로 지정한다. 성만 맞추면 동명이인을 잘못 뺀다.
SCANT_AUTHORS = [
    ("Doyle, Arthur Conan", "1859-1930"),          # 셜록 홈즈의 모험
    ("Dickens, Charles", "1812-1870"),             # 두 도시 이야기
    ("Fitzgerald, F. Scott", "1896-1940"),         # 위대한 개츠비
    ("Baum, L. Frank", "1856-1919"),               # 오즈의 마법사
    ("Austen, Jane", "1775-1817"),                 # 오만과 편견 (코퍼스에 0편 — 확인용)
]


def persons(author: str) -> list[tuple[str, str]]:
    """author 필드에 등장하는 모든 인물 (이름, 생몰). 격리 판정에 쓴다."""
    out = []
    for seg in author.split(";"):
        name, years, _role = parse_person(seg)
        if name:
            out.append((name, years))
    return out


def norm_key(name: str, years: str) -> tuple[str, str]:
    """확장형 괄호는 비교에서 제외 — 'Fitzgerald, F. Scott (Francis Scott)' == 'Fitzgerald, F. Scott'."""
    base = name.split("(")[0].strip().rstrip(",").strip()
    return (base.lower(), years)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, required=True, help="작가당 상한 N (인물 단위)")
    ap.add_argument("--isolate-scant", action="store_true", help="ScANT 저자 격리")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    cut = json.loads(FINAL_MF.read_text())["download_cut"]
    dl = json.loads(DOWNLOADS.read_text())
    scant = {norm_key(n, y) for n, y in SCANT_AUTHORS}

    rows, st = [], Counter()
    for line in POOL.open(encoding="utf-8"):
        r = json.loads(line)
        st["pool"] += 1
        if dl.get(str(r["gutenberg_id"]), 0) >= cut:
            st["skip_download"] += 1
            continue
        if a.isolate_scant:
            hit = {norm_key(*p) for p in persons(r["author"])} & scant
            if hit:
                st["skip_scant"] += 1
                st[f"scant::{sorted(hit)[0][0]}"] += 1
                continue
        rows.append(r)

    per: Counter = Counter()
    keep = []
    for r in rows:
        if is_anthology(r["author"]):
            st["anthology_exempt"] += 1
            keep.append(r)
            continue
        name, years, _ = parse_person(r["author"].split(";")[0])
        k = norm_key(name, years)
        if per[k] >= a.cap:
            st["skip_author_cap"] += 1
            continue
        per[k] += 1
        keep.append(r)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8") as fh:
        for r in keep:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    over = [(k, v) for k, v in per.items() if v > a.cap]
    mf = {"n_pieces": len(keep), "n_persons": len(per), "per_author_cap": a.cap,
          "author_key": "person(name+years)", "isolate": "dev3" + ("+scant" if a.isolate_scant else ""),
          "download_cut": cut, "sent_total": sum(r["n_sent"] for r in keep),
          "n_volumes": len({r["gutenberg_id"] for r in keep}),
          "cap_violations": len(over), "stats": dict(st)}
    Path(str(a.out).replace(".jsonl", "_manifest.json")).write_text(
        json.dumps(mf, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"풀 {st['pool']:,} → 다운로드컷 -{st['skip_download']:,} "
          f"→ ScANT격리 -{st['skip_scant']:,} → 상한 -{st['skip_author_cap']:,}")
    print(f"= {len(keep):,}편 / 권 {mf['n_volumes']:,} / 문장 {mf['sent_total']:,} "
          f"· 인물 {len(per):,}명 · 상한위반 {len(over)}")
    for k, v in sorted(st.items()):
        if k.startswith("scant::"):
            print(f"    격리 {k[7:]}: {v}편")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
