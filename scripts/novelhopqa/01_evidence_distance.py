"""NovelHopQA 근거 단락 거리 분포 (hop별).

목적: 한 질문의 근거 단락들이 원작에서 얼마나 떨어져 있는가.
  - 가까이 뭉쳐 있다 -> 사건 정렬 청크가 여러 hop 을 한 청크에 담는다 (진행)
  - 책 전체에 흩어져 있다 -> 4번의 독립 검색 문제, 분절 방식 무관 (중단)

입력: HF datasets `abhaygupta1266/novelhopqa` parquet (hop_1..hop_4)
      + 기존 GutenQA 코퍼스 books.jsonl (단락 분할된 PG 원문)
출력: outputs/experiments/2026-07-31_novelhopqa_probe/evidence_distance.json

새 LLM 호출 없음. 네트워크 없음(파일은 미리 받아둠).
"""

import json
import re
import statistics as st
import sys
import unicodedata
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
BOOKS = REPO / "outputs/experiments/2026-07-29_gutenqa_chunking/data/books.jsonl"
OUT = REPO / "outputs/experiments/2026-07-31_novelhopqa_probe"
NHQ = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/nhq")

# 350 토큰 ~= 270 단어 (NovelHopQA RAG 설정). 청크 창 후보.
CHUNK_WORDS = [270, 500, 1000]


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("’", "'").replace("‘", "'")
    s = s.replace("“", '"').replace("”", '"')
    s = s.replace("—", "-").replace("–", "-")
    s = re.sub(r"[^a-z0-9' ]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def title_key(s: str) -> str:
    s = s.lower().replace("_", " ").replace("-", " ")
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return re.sub(r"\s+", " ", s).strip()


class Book:
    """정규화 단락 리스트 + 단어 오프셋 인덱스."""

    def __init__(self, paragraphs):
        self.paras = [norm(p) for p in paragraphs]
        self.wc = [len(p.split()) for p in self.paras]
        self.wstart = []  # 각 단락의 시작 단어 오프셋
        acc = 0
        for w in self.wc:
            self.wstart.append(acc)
            acc += w
        self.total_words = acc
        # 문자 기반 substring 검색용 결합 텍스트 + 단락 경계
        self.joined = " \x00 ".join(self.paras)
        self.cstart = []
        acc = 0
        for p in self.paras:
            self.cstart.append(acc)
            acc += len(p) + 3
        # 첫 60자 -> 단락 인덱스 (빠른 조회)
        self.head = {}
        for i, p in enumerate(self.paras):
            self.head.setdefault(p[:60], i)

    def locate(self, ev: str):
        """근거 단락 -> (단락 인덱스, 매칭 방식). 못 찾으면 (None, 사유)."""
        e = norm(ev)
        if len(e) < 20:
            return None, "too_short"
        if e[:60] in self.head:
            return self.head[e[:60]], "head60"
        pos = self.joined.find(e)
        if pos < 0 and len(e) > 200:
            pos = self.joined.find(e[:200])  # 뒤가 잘린 근거
        if pos < 0 and len(e) > 120:
            pos = self.joined.find(e[-120:])  # 앞이 잘린 근거
        if pos < 0:
            return None, "not_found"
        # pos 를 포함하는 단락
        lo, hi = 0, len(self.cstart) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.cstart[mid] <= pos:
                lo = mid
            else:
                hi = mid - 1
        return lo, "substring"


def max_in_window(word_positions, window):
    """정렬된 단어 오프셋들 중 길이 window 창에 최대 몇 개가 들어가는가."""
    best, j = 0, 0
    for i in range(len(word_positions)):
        while word_positions[i] - word_positions[j] > window:
            j += 1
        best = max(best, i - j + 1)
    return best


def main():
    books = {}
    for line in BOOKS.open():
        d = json.loads(line)
        books[title_key(d["book_name"].split("_-_")[0])] = d

    def find_book(title):
        k = title_key(title)
        if k in books:
            return books[k]
        for gk, gv in books.items():
            if k in gk or gk in k:
                return gv
        return None

    cache = {}
    rows, unmatched_books = [], {}
    for hop in [1, 2, 3, 4]:
        df = pd.read_parquet(NHQ / f"hop_{hop}.parquet")
        for _, r in df.iterrows():
            bd = find_book(r.book)
            if bd is None:
                unmatched_books[r.book] = unmatched_books.get(r.book, 0) + 1
                continue
            bid = bd["book_name"]
            if bid not in cache:
                cache[bid] = Book(bd["paragraphs"])
            bk = cache[bid]

            ctx = r.context
            parts = re.split(r"Paragraph\s+\d+\s*:", ctx)
            parts = [p.strip() for p in parts if p.strip()]
            if not parts:
                continue

            idxs, fails = [], []
            for p in parts:
                i, why = bk.locate(p)
                (idxs if i is not None else fails).append(i if i is not None else why)

            rows.append(
                dict(
                    hop=hop,
                    book=r.book,
                    n_parts=len(parts),
                    n_located=len(idxs),
                    idxs=sorted(idxs),
                    fails=fails,
                    book_paras=len(bk.paras),
                    book_words=bk.total_words,
                    wpos=sorted(bk.wstart[i] for i in idxs),
                )
            )

    def q(xs, p):
        xs = sorted(xs)
        if not xs:
            return None
        return round(xs[min(len(xs) - 1, int(p * len(xs)))], 1)

    report = {"chunk_words": CHUNK_WORDS, "unmatched_books": unmatched_books, "by_hop": {}}
    for hop in [1, 2, 3, 4]:
        hr = [r for r in rows if r["hop"] == hop]
        full = [r for r in hr if r["n_located"] == r["n_parts"] and r["n_parts"] >= 2]
        loc_rate = (
            sum(r["n_located"] for r in hr) / max(1, sum(r["n_parts"] for r in hr))
        )
        e = dict(
            n_qa_with_book=len(hr),
            n_parts_mean=round(st.mean([r["n_parts"] for r in hr]), 2) if hr else None,
            evidence_locate_rate=round(loc_rate, 4),
            n_fully_located_multi=len(full),
        )
        if full:
            span_p = [r["idxs"][-1] - r["idxs"][0] for r in full]
            span_w = [r["wpos"][-1] - r["wpos"][0] for r in full]
            frac = [
                (r["wpos"][-1] - r["wpos"][0]) / max(1, r["book_words"]) for r in full
            ]
            gaps = [
                b - a for r in full for a, b in zip(r["idxs"], r["idxs"][1:])
            ]
            e["span_paras"] = {
                "p25": q(span_p, 0.25), "median": q(span_p, 0.5),
                "p75": q(span_p, 0.75), "p90": q(span_p, 0.9), "max": max(span_p),
            }
            e["span_words"] = {
                "p25": q(span_w, 0.25), "median": q(span_w, 0.5),
                "p75": q(span_w, 0.75), "p90": q(span_w, 0.9), "max": max(span_w),
            }
            e["span_frac_of_book"] = {
                "median": round(st.median(frac), 4),
                "p90": q(frac, 0.9),
            }
            e["consecutive_gap_paras"] = {
                "median": q(gaps, 0.5),
                "adjacent_rate(<=1)": round(sum(1 for g in gaps if g <= 1) / len(gaps), 4),
            }
            for cw in CHUNK_WORDS:
                cov = [max_in_window(r["wpos"], cw) / r["n_parts"] for r in full]
                allin = sum(
                    1 for r in full if max_in_window(r["wpos"], cw) == r["n_parts"]
                )
                e[f"chunk{cw}w"] = {
                    "mean_frac_evidence_in_one_chunk": round(st.mean(cov), 4),
                    "all_evidence_in_one_chunk_rate": round(allin / len(full), 4),
                }
        report["by_hop"][hop] = e

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "evidence_distance.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    with (OUT / "per_qa.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
