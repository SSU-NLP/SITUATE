#!/usr/bin/env python3
r"""D-B35 — 설계문서 §4.6 **D1(extractor-defined space)** 용 축별 태거 피처 추출.

## 왜
D-B34 분해: **M5−M4 이득의 실질 전부가 입력경로 분리에서 온다**(상태 분리·dynamics 분리는 단독
기여 없음). 그런데 현 입력경로는 *"768D → 32 → 8D, 축이 뭔지 아무도 안 알려주고 앵커 손실이
뒤에서 당겨주길 기대"* 하는 **학습형** 경로다(설계문서 분류로는 D2). 설계문서 §4.6 의 진짜 **D1**
은 *"run cheap NER/coref/time-tagger at inference"* — **외부 태거가 축 공간을 정의**하는 것이다.
D-B34 가 가리킨 유일한 상승 방향이므로 그 경로를 실제로 만든다.

## 무엇을 만드나
문장마다 **축별 8차원 = 총 40차원**(= 5×bottleneck, D-B33 의 공유입력40 과 폭이 같아 비교 가능).
전부 spaCy `en_core_web_sm`(NER + POS + morph + dep)과 **어휘 사전**으로만 만든다 — LLM 없음,
추론 시 1,069 문장/초.

⚠️ **축별로 태거 품질이 다르다** (설계문서 §10 이 이미 인정): protagonist(PERSON)·space(GPE/LOC/
FAC)·time(DATE/TIME+시제)은 태거가 직접 잡지만, **causality·intentionality 는 깨끗한 태거가 없어
어휘 사전에 의존**한다. 그래서 학습 쪽에서 `--tagger-axes 3` (앞 3축만 대체) 조건을 함께 돌린다.

## 정렬 규약 (⚠️ 어기면 전부 무의미)
문장 순서 = `build_anchor_dataset.collect_docs` 와 **동일**(첫 행의 previous + 각 행의 current).
행 순서는 `{split}_index.npz` 의 `doc_ids` 순서를 그대로 따르며, 문서별 문장 수가
`sent_offsets` 차이와 **정확히 일치하는지 검증한 뒤에만** 쓴다(불일치 시 즉시 중단).

## 정규화
`log1p` 후 **앵커(phase1) 코퍼스 전체에서 구한 mean/std 로 표준화**하고 그 통계를 저장한다.
dev-3 는 **저장된 통계를 그대로 적용**한다(dev-3 통계로 재정규화하면 누출).

산출: `outputs/experiments/m4m5_phase1_anchor/data/{split}_axisfeat.npy` [N_sent, 40] float16
      + `axisfeat_stats.json`(정규화 통계 · 열 이름 · 축 슬라이스)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))
from build_anchor_dataset import DEV3, PHASE1_MASTER, collect_docs  # noqa: E402

OUT = REPO / "outputs/experiments/m4m5_phase1_anchor/data"
STATS = OUT / "axisfeat_stats.json"
PER_AXIS = 8                      # 축당 차원 = bottleneck 기본값과 일치
AXES = ["protagonist", "space", "time", "causality", "intentionality"]

# ── 어휘 사전 (태거가 없는 축용) ────────────────────────────────────────────────
LOC_PREP = {"in", "at", "on", "into", "onto", "from", "toward", "towards", "inside",
            "outside", "beneath", "under", "above", "across", "through", "beside"}
MOVE_VERB = {"go", "come", "walk", "run", "enter", "leave", "arrive", "depart", "return",
             "climb", "descend", "cross", "travel", "move", "step", "ride", "drive", "fly"}
TIME_ADV = {"then", "later", "now", "suddenly", "meanwhile", "afterward", "afterwards",
            "soon", "immediately", "eventually", "finally", "already", "yesterday",
            "tomorrow", "today", "tonight", "again", "still", "next"}
TIME_PREP = {"after", "before", "during", "until", "till", "since", "while", "when"}
CAUSE = {"because", "since", "so", "therefore", "thus", "hence", "consequently", "as"}
CONTRAST = {"but", "however", "although", "though", "yet", "nevertheless", "whereas"}
COND = {"if", "unless", "whether", "lest"}
SEQ = {"then", "next", "afterward", "afterwards", "subsequently", "finally"}
INTENT_VERB = {"want", "decide", "plan", "intend", "try", "hope", "wish", "need", "mean",
               "aim", "seek", "attempt", "determine", "choose", "refuse", "promise"}
MENTAL_VERB = {"think", "feel", "believe", "know", "realize", "remember", "forget",
               "wonder", "imagine", "understand", "suppose", "expect", "fear"}
MODAL = {"will", "would", "should", "must", "can", "could", "might", "may", "shall", "ought"}
P1 = {"i", "we", "me", "us", "my", "our", "myself", "ourselves"}
P2 = {"you", "your", "yours", "yourself", "yourselves"}
P3 = {"he", "she", "they", "him", "her", "them", "his", "their", "hers", "theirs",
      "himself", "herself", "themselves"}

COLS = (
    # protagonist — PERSON 엔티티 + 인칭 대명사 + 고유명사
    ["prot_n_person", "prot_uniq_person", "prot_p1", "prot_p2", "prot_p3",
     "prot_propn", "prot_has_person", "prot_subj_animate"]
    # space — 장소 엔티티 + 장소 전치사 + 이동 동사
    + ["spc_n_gpe", "spc_n_loc", "spc_n_fac", "spc_loc_prep", "spc_here_there",
       "spc_move_verb", "spc_n_place_ent", "spc_has_new_place"]
    # time — 시간 엔티티 + 시제/상 + 시간 부사/전치사
    + ["tim_n_date", "tim_n_time", "tim_adv", "tim_past", "tim_pres", "tim_perfect",
       "tim_prep", "tim_has_time_ent"]
    # causality — 담화 접속어 + 절 구조 (태거 없음, 어휘 사전)
    + ["cau_cause", "cau_contrast", "cau_cond", "cau_seq", "cau_n_verb",
       "cau_n_clause", "cau_neg", "cau_root_is_verb"]
    # intentionality — 의도/양태/사고 동사 (태거 없음, 어휘 사전)
    + ["int_intent_verb", "int_purpose_to", "int_modal", "int_imperative",
       "int_mental_verb", "int_question", "int_quote", "int_future"]
)
assert len(COLS) == 5 * PER_AXIS


def featurize(doc) -> list[float]:
    """spaCy Doc -> 40차원 원시(비정규화) 피처. 순서는 COLS 와 동일."""
    ents = [(e.text, e.label_) for e in doc.ents]
    lower = [t.lower_ for t in doc]
    person = [t for t, l in ents if l == "PERSON"]
    gpe = sum(1 for _, l in ents if l == "GPE")
    loc = sum(1 for _, l in ents if l == "LOC")
    fac = sum(1 for _, l in ents if l == "FAC")
    date = sum(1 for _, l in ents if l == "DATE")
    time_ = sum(1 for _, l in ents if l == "TIME")
    verbs = [t for t in doc if t.pos_ in ("VERB", "AUX")]
    past = sum(1 for t in verbs if "Past" in t.morph.get("Tense"))
    pres = sum(1 for t in verbs if "Pres" in t.morph.get("Tense"))
    perfect = sum(1 for t in doc if t.tag_ == "VBN" and t.head.lemma_ in ("have", "be"))
    lemmas = [t.lemma_.lower() for t in doc]
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    subj = next((t for t in doc if t.dep_ in ("nsubj", "nsubjpass")), None)
    return [
        # protagonist
        len(person), len(set(person)),
        sum(1 for w in lower if w in P1), sum(1 for w in lower if w in P2),
        sum(1 for w in lower if w in P3),
        sum(1 for t in doc if t.pos_ == "PROPN"), float(len(person) > 0),
        float(subj is not None and (subj.pos_ == "PRON" or subj.ent_type_ == "PERSON")),
        # space
        gpe, loc, fac, sum(1 for t in doc if t.pos_ == "ADP" and t.lower_ in LOC_PREP),
        sum(1 for w in lower if w in ("here", "there")),
        sum(1 for t in doc if t.lemma_.lower() in MOVE_VERB and t.pos_ == "VERB"),
        gpe + loc + fac, float(gpe + loc + fac > 0),
        # time
        date, time_, sum(1 for w in lower if w in TIME_ADV), past, pres, perfect,
        sum(1 for t in doc if t.lower_ in TIME_PREP), float(date + time_ > 0),
        # causality
        sum(1 for w in lower if w in CAUSE), sum(1 for w in lower if w in CONTRAST),
        sum(1 for w in lower if w in COND), sum(1 for w in lower if w in SEQ),
        len(verbs), sum(1 for t in doc if t.dep_ in ("advcl", "ccomp", "xcomp", "relcl")),
        sum(1 for t in doc if t.dep_ == "neg" or t.lower_ in ("not", "never", "no")),
        float(root is not None and root.pos_ in ("VERB", "AUX")),
        # intentionality
        sum(1 for lm in lemmas if lm in INTENT_VERB),
        sum(1 for t in doc if t.lower_ == "to" and t.head.pos_ == "VERB" and t.dep_ == "aux"),
        sum(1 for w in lower if w in MODAL),
        float(root is not None and root.tag_ == "VB" and subj is None),
        sum(1 for lm in lemmas if lm in MENTAL_VERB),
        float("?" in doc.text), float('"' in doc.text or "“" in doc.text),
        sum(1 for w in lower if w in ("will", "shall")) + float("going to" in doc.text.lower()),
    ]


def sentences_for(split: str) -> tuple[list[str], list[str], np.ndarray]:
    """(doc_ids, 평탄화된 문장, sent_offsets) — index.npz 의 문서 순서를 그대로 따른다."""
    ix = np.load(OUT / f"{split}_index.npz", allow_pickle=True)
    ids = [str(x) for x in ix["doc_ids"]]
    so = ix["sent_offsets"]
    if split == "phase1":
        d, s, _ = collect_docs(PHASE1_MASTER)
        have = dict(zip(d, s))
    else:
        # test3 = 봉인 시험지(EventRecall). 정규화 통계는 앵커 것을 그대로 쓴다(누출 방지).
        srcs = ([REPO / f"outputs/eacl_phase0/{n}_michelmann_master.jsonl" for n in DEV3]
                if split == "dev3" else
                [REPO / f"outputs/eacl_phase0/{n}_eventrecall_goldtopn_master.jsonl"
                 for n in ("run", "gohitler", "mymotherslife")])
        have = {}
        for src in srcs:
            d, s, _ = collect_docs(src)
            have.update(dict(zip(d, s)))
    flat: list[str] = []
    for i, doc in enumerate(ids):
        need = int(so[i + 1] - so[i])
        got = have.get(doc)
        if got is None or len(got) != need:
            raise SystemExit(f"🔴 정렬 실패: {doc} 문장 {0 if got is None else len(got)} ≠ {need}"
                             " — 임베딩 행과 어긋난 피처는 만들지 않는다")
        flat.extend(got)
    assert len(flat) == int(so[-1]), f"{len(flat)} != {so[-1]}"
    return ids, flat, so


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["phase1", "dev3", "test3"])
    ap.add_argument("--n-process", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=512)
    args = ap.parse_args()
    import spacy

    ids, sents, so = sentences_for(args.split)
    print(f"[align] {args.split}: {len(ids):,} docs · {len(sents):,} sentences ✅", flush=True)

    nlp = spacy.load("en_core_web_sm")
    X = np.zeros((len(sents), len(COLS)), dtype=np.float32)
    for i, doc in enumerate(nlp.pipe(sents, batch_size=args.batch_size,
                                     n_process=args.n_process)):
        X[i] = featurize(doc)
        if (i + 1) % 200_000 == 0:
            print(f"  {i+1:,}/{len(sents):,}", flush=True)

    X = np.log1p(np.clip(X, 0, None))              # 카운트 → log1p (꼬리 압축)
    if args.split == "phase1":
        mu, sd = X.mean(0), X.std(0)
        sd = np.where(sd < 1e-6, 1.0, sd)          # 상수 열은 그대로 0 으로 남긴다
        STATS.write_text(json.dumps({"cols": COLS, "per_axis": PER_AXIS, "axes": AXES,
                                     "mean": mu.tolist(), "std": sd.tolist(),
                                     "source_split": "phase1"}, indent=2))
        print(f"[stats] 앵커에서 정규화 통계 저장 -> {STATS.name}", flush=True)
    else:
        if not STATS.exists():
            raise SystemExit("🔴 phase1 을 먼저 돌려 정규화 통계를 만들어야 한다(누출 방지)")
        st = json.loads(STATS.read_text())
        mu, sd = np.asarray(st["mean"], np.float32), np.asarray(st["std"], np.float32)
        print("[stats] 앵커 통계 재사용 (dev-3 통계로 재정규화하지 않는다)", flush=True)
    X = ((X - mu) / sd).astype(np.float16)

    p = OUT / f"{args.split}_axisfeat.npy"
    np.save(p, X)
    nz = (np.abs(X.astype(np.float32)) > 1e-6).mean(0)
    print(f"-> {p}  {X.shape}")
    for a in range(5):
        s = slice(a * PER_AXIS, (a + 1) * PER_AXIS)
        print(f"   {AXES[a]:16s} 비영 비율 " + " ".join(f"{v:.2f}" for v in nz[s]))


if __name__ == "__main__":
    main()
