"""01 의 근거-단락 매칭이 진짜인지 표본 검증. 새 실행 없음, 같은 입력 재확인."""
import json
import random
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module

m = import_module("01_evidence_distance")

NHQ = Path(sys.argv[1])
books = {}
for line in m.BOOKS.open():
    d = json.loads(line)
    books[m.title_key(d["book_name"].split("_-_")[0])] = d


def find_book(t):
    k = m.title_key(t)
    if k in books:
        return books[k]
    for gk, gv in books.items():
        if k in gk or gk in k:
            return gv


rng = random.Random(0)
for hop in [2, 4]:
    df = pd.read_parquet(NHQ / f"hop_{hop}.parquet")
    df = df[[find_book(b) is not None for b in df.book]]
    sample = df.sample(n=40, random_state=1)
    ok = bad = dup = 0
    bad_ex = []
    for _, r in sample.iterrows():
        bk = m.Book(find_book(r.book)["paragraphs"])
        parts = [p.strip() for p in re.split(r"Paragraph\s+\d+\s*:", r.context) if p.strip()]
        idxs = []
        for p in parts:
            i, why = bk.locate(p)
            if i is None:
                bad += 1
                bad_ex.append(("nofind", r.book, p[:80]))
                continue
            e = m.norm(p)
            got = bk.paras[i]
            # 근거가 그 단락에 실제로 들어있는가 (앞 120자 기준)
            if e[:120] in got or got[:120] in e:
                ok += 1
            else:
                bad += 1
                bad_ex.append(("mismatch", r.book, e[:80], got[:80]))
            idxs.append(i)
        if len(set(idxs)) < len(idxs):
            dup += 1
    print(f"hop{hop}: 근거조각 검증 ok={ok} bad={bad} | 같은단락중복 QA={dup}/40")
    for b in bad_ex[:4]:
        print("   ", b)

# 근거 단락 위치가 책 전체에 퍼지는지 vs 128k 토큰 발췌 안인지
df = pd.read_parquet(NHQ / "hop_4.parquet")
df = df[[find_book(b) is not None for b in df.book]]
print("\n표본 QA 5건의 단락 인덱스/책 단락수:")
for _, r in df.sample(n=5, random_state=3).iterrows():
    bk = m.Book(find_book(r.book)["paragraphs"])
    parts = [p.strip() for p in re.split(r"Paragraph\s+\d+\s*:", r.context) if p.strip()]
    idxs = [bk.locate(p)[0] for p in parts]
    print(f"  {r.book[:28]:30s} idx={idxs} / {len(bk.paras)} paras, {bk.total_words} words")
