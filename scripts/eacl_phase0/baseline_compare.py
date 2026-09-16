#!/usr/bin/env python3
"""M5 vs `methods/` baseline — **단일 하네스** 대조. 예측을 원자료로 저장한다.

## 왜 새로 만드나 (하네스 불일치, 2026-07-29 실측)
채점기가 **두 개**였다:
- **(A)** `hi_ontop.dts_scoring` = SuperDialseg 공식 `SegmentationEvaluation`.
  CLAUDE.md 가 *"이걸 그대로 호출, 재구현 금지"* 로 강제. M1/M4/M5 의 EACL 수치 전부 이것.
- **(B)** `methods/` online 쪽 = `segeval` 직접 + per-dialogue macro mean.
  `outputs/reports/dts_result.md` 의 online baseline 행이 이것.

**동일 gold·동일 예측으로 재보니 ΔScore = 0.045** (Pk 0.4446 vs 0.4192 · WD 0.4829 vs 0.4568 ·
F1 0.2087 vs 0.2729). 우리가 재려는 효과(M5−M4 = 0.025)보다 **크다.**
⇒ 두 출처의 숫자를 한 표에 나란히 두는 것은 **무효**다(D-B24 와 같은 계열의 사고).

## 원칙 (codex 위임 판정)
1. **주 표 하네스 = (A) 로 사전 고정.** A/B 중 유리한 쪽을 고르지 않는다.
2. **예측 파일이 원자료**, 집계 점수는 파생물. 전 방법이 예측을 덤프한다 → 재채점·감사 가능.
3. **(B) 점수도 함께 계산**하되 *하네스 차이 진단용 브릿지*이지 주 결과가 아니다.
4. **논문 인용 수치(Jiang et al. 2023 Table 3)는 이 표에 넣지 않는다** — 제3의 하네스다.

## 주 무대는 서사다 (codex Q4)
baseline 6종은 전부 **대화** 벤치마크용이지만 비지도 4종(TextTiling/GraphSeg/GreedySeg/CSM)은
도메인 무관하게 돌아간다. **우리 주장(EISM 다축 리셋)의 검증 무대는 서사**이므로 dev-3(서사)가
주 비교이고, DTS-3(대화)는 **OOD 전이 실험**으로만 읽는다.

🔴 dev-3 = 연습장 3편. 시험지 미개봉. 탐색적이다.
"""
from __future__ import annotations

import argparse
import os
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))
from hi_ontop.baselines._seg_utils import boundary_set_f1, pk_wd  # noqa: E402
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402
from hi_ontop.factored_slot import AXES, FactoredSlot  # noqa: E402
from train_anchor_m4m5 import EXP, Corpus, axis_probs  # noqa: E402

OUT = REPO / "outputs/experiments/2026-07-29_baseline_compare"
M1 = dict(m=2, rho=0.7, a=0.5)                 # registry §2 canonical
ANCHOR_R = 0.0642                              # 🔒 D-B30 앵커 동결 경계율


# ── 데이터 ────────────────────────────────────────────────────────────────────
def load_dataset(name: str) -> list[tuple[str, list[str], np.ndarray]]:
    """(doc_id, 문장 리스트, gold 전이 라벨 [n-1]) 목록. 문장 원문이 필요하다(baseline 입력)."""
    if name == "dev3":
        docs = ["pieman", "tunnel", "monkey"]
        src = [(d, REPO / f"outputs/eacl_phase0/{d}_michelmann_master.jsonl") for d in docs]
    elif name.startswith("test3"):
        # 🔒 봉인 시험지 (EventRecall / *Born a Crime* 3편). gold 규칙을 이름에 박는다 —
        # `test3` = 밀도 정합 top-N(D-B62 게이트 결과), `test3_goldauthor` = 저자 규칙(부록).
        tag = "goldauthor" if name.endswith("goldauthor") else "goldtopn"
        docs = ["run", "gohitler", "mymotherslife"]
        src = [(d, REPO / f"outputs/eacl_phase0/{d}_eventrecall_{tag}_master.jsonl") for d in docs]
    else:
        d = name.replace("dts_full_", "")
        src = [(d, REPO / f"outputs/eacl_phase0/dts_full_{d}_master.jsonl")]
    out = []
    for doc, p in src:
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        by: dict[str, list] = {}
        for r in rows:
            by.setdefault(r.get("doc_id", doc), []).append(r)
        for did, rs in by.items():
            rs.sort(key=lambda r: r["transition"])
            sents = [rs[0]["previous"]] + [r["current"] for r in rs]
            gold = np.asarray([int(r["boundary"]) for r in rs], np.int8)
            if len(sents) >= 3:
                out.append((did, sents, gold))
    return out


# ── 우리 방법 ─────────────────────────────────────────────────────────────────
def preds_ours(tag: str, data, corp: Corpus, dev: str) -> dict[str, np.ndarray]:
    """M5 / M4 — `Σ_k P(reset_k)` + 🔒 앵커 동결 고정분위수(D-B30)."""
    ck = torch.load(EXP / f"{tag}_per_slot_slot_seed0.pt", map_location=dev, weights_only=False)
    m = FactoredSlot(slot_dim=ck["args"]["slot_dim"], mono=ck["mono"], pos_weight=ck["pos_weight"],
                     dynamics=ck.get("dynamics"),
                     pred_space=ck["args"].get("pred_space", "obs")).to(dev).eval()
    m.load_state_dict(ck["state"])
    P = axis_probs(m, corp, list(range(corp.n_docs)), dev, bs=4, with_anchor=False)
    return {data[i][0]: (lambda s: (s >= np.quantile(s, 1 - ANCHOR_R)).astype(np.int8))(p.sum(1))
            for i, p in enumerate(P)}


ROBERTA_DIR = REPO / "outputs/experiments/2026-07-29_roberta_retrain/model"


def preds_roberta(data, device: str) -> tuple[dict[str, np.ndarray], float]:
    """지도학습 RoBERTa (SuperDialseg Table 3 재현) — **strict-causal online** 추론.

    유일한 **supervised** baseline이자 M5 의 가장 가까운 peer다: 둘 다 지도학습이고
    각자 자기 도메인(RoBERTa=대화 gold, M5=서사 앵커)에서 학습됐다. 그래서 2×2 가 나온다 —
    각 방법을 자기 도메인과 OOD 양쪽에서 본다.

    ⚠️ 원 REPORT 의 수치는 **하네스 (C)**(official SuperDialseg window=평균 segment 길이/2)라
    이 표에 그대로 옮길 수 없다. 여기서는 예측만 받아 **하네스 (A)** 로 다시 채점한다.
    """
    import torch as _t
    from transformers import AutoTokenizer, RobertaForTokenClassification
    sys.path.insert(0, str(REPO / "methods/RoBERTa/online"))
    from segment import predict_online                                   # noqa: E402
    assert ROBERTA_DIR.exists(), f"RoBERTa 체크포인트 없음: {ROBERTA_DIR}"
    tok = AutoTokenizer.from_pretrained(str(ROBERTA_DIR))
    m = RobertaForTokenClassification.from_pretrained(str(ROBERTA_DIR)).to(device)
    cfg = {"sliding_window": 20, "max_utt_len": 25, "max_seq_len": 512}
    dialogs = [(sents, [0] * len(sents)) for _, sents, _ in data]        # 라벨은 추론에 미사용
    t0 = time.perf_counter()
    with _t.no_grad():
        yps = predict_online(m, tok, dialogs, cfg, device)
    n_turn = sum(len(s) for _, s, _ in data)
    # yp[i]=1 = 발화 i 가 세그먼트 마지막 turn ⇔ 전이 i 가 경계. gold 는 [n-1] 이므로 앞부분만.
    return ({did: np.asarray(yp[: len(g)], np.int8) for (did, _, g), yp in zip(data, yps)},
            (time.perf_counter() - t0) / max(n_turn, 1) * 1000.0)


def preds_m1(data, corp: Corpus) -> dict[str, np.ndarray]:
    """M1-CosDrift δ_eff — 같은 인코더·같은 밀도 규칙(공정 비교)."""
    out = {}
    for i, (did, _, _) in enumerate(data):
        e = np.asarray(corp.emb[int(corp.so[i]):int(corp.so[i + 1])], np.float32)
        s = np.zeros(len(e) - 1)
        for t in range(1, len(e)):
            dp = 1.0 - float(e[t - 1] @ e[t])
            lo = max(0, t - M1["m"])
            w = np.array([M1["rho"] ** j for j in range(t - lo)])[::-1]
            ctx = (w[:, None] * e[lo:t]).sum(0)
            nrm = np.linalg.norm(ctx)
            dc = 1.0 - float(ctx @ e[t] / nrm) if nrm > 0 else dp
            s[t - 1] = M1["a"] * dp + (1 - M1["a"]) * dc
        out[did] = (s >= np.quantile(s, 1 - ANCHOR_R)).astype(np.int8)
    return out



# ── 🔒 동결 기본 작동점판 (D-B60) ─────────────────────────────────────────────
# `configs/m5_default.json` 이 선언한 **현행 동결 설정**을 그대로 쓴 행이다.
#   체크포인트 = models/m5_default/M5.pt (= M5_phase1_i2_..._pwone_seed0, 의도축 I2 + pos_weight=1)
#   임계       = online_nf c=1.0 (적응 임계, 과거 통계만 사용)
# 🔴 위의 `M5`/`M4`/`M1-CosDrift` 행은 **고정분위수 r=0.0642 + 구 체크포인트**다.
#    그 분위수는 프로젝트가 **폐기 선언**했다 — r 은 `count>=t*=3` 의 앵커 경계율인데
#    t*=3 자체가 D-B33 에서 폐기됐다(정의가 사라진 수치를 쓰면 순환).
#    두 작동점을 **나란히 보고**한다. 하나를 조용히 갈아끼우지 않는다.
def preds_ours_adaptive(tag: str, data, corp: Corpus, dev: str) -> dict[str, np.ndarray]:
    """M5-adaptive / M4-adaptive / M1-adaptive — 적응 임계 `s_t >= mean(s_<t) + 1.0*std(s_<t)`."""
    from hi_ontop.hi_ontop_v2 import adaptive_boundaries                 # noqa: PLC0415

    def cut(s: np.ndarray) -> np.ndarray:
        b = adaptive_boundaries(np.concatenate([[0.0], s]).tolist(), c=1.0, mode="online_nf")
        return np.asarray(b, np.int8)[: len(s)]

    if tag == "M1-CosDrift":
        raw = preds_m1_signal(data, corp)
    elif tag == "M5":
        # 🔒 동결 기본 진입점을 그대로 호출한다(설정·체크포인트 재지정 금지).
        from hi_ontop.m5_default import segment                          # noqa: PLC0415
        out = {}
        for did, sents, gold in data:
            r = segment(sents)
            v = np.zeros(len(gold), np.int8)
            for b in r["boundaries"]:
                if 1 <= b <= len(gold):
                    v[b - 1] = 1
            out[did] = v
        return out
    else:                                    # M4 — 같은 신호에 같은 임계 규칙
        ck = torch.load(EXP / f"{tag}_per_slot_slot_seed0.pt", map_location=dev,
                        weights_only=False)
        m = FactoredSlot(slot_dim=ck["args"]["slot_dim"], mono=ck["mono"],
                         pos_weight=ck["pos_weight"], dynamics=ck.get("dynamics"),
                         pred_space=ck["args"].get("pred_space", "obs")).to(dev).eval()
        m.load_state_dict(ck["state"])
        P = axis_probs(m, corp, list(range(corp.n_docs)), dev, bs=4, with_anchor=False)
        raw = {data[i][0]: p.sum(1) for i, p in enumerate(P)}
    return {k: cut(v) for k, v in raw.items()}


def preds_m1_signal(data, corp: Corpus) -> dict[str, np.ndarray]:
    """M1 δ_eff **신호**(임계 적용 전). 분위수판·적응판이 같은 신호를 쓰게 한다."""
    out = {}
    for i, (did, _, _) in enumerate(data):
        e = np.asarray(corp.emb[int(corp.so[i]):int(corp.so[i + 1])], np.float32)
        s = np.zeros(len(e) - 1)
        for t in range(1, len(e)):
            dp = 1.0 - float(e[t - 1] @ e[t])
            lo = max(0, t - M1["m"])
            w = np.array([M1["rho"] ** j for j in range(t - lo)])[::-1]
            ctx = (w[:, None] * e[lo:t]).sum(0)
            nrm = np.linalg.norm(ctx)
            dc = 1.0 - float(ctx @ e[t] / nrm) if nrm > 0 else dp
            s[t - 1] = M1["a"] * dp + (1 - M1["a"]) * dc
        out[did] = s
    return out


# ── 🔒 D-B41 동결 작동점 ───────────────────────────────────────────────────────
# 서사는 **dev-3(사람 gold)**, 대화는 **superseg/validation** 에서 골랐다.
# (앵커 의사-gold 선택은 무효 — 사람 경계 라벨이 없다. D-B41 §1)
# 값이 발표 기본값과 같으면 `=pub` 로 표시했다 — 탐색 결과 대부분 발표값이 그대로 최적이었다.
FROZEN: dict[str, dict[str, dict]] = {
    "narrative": {
        "TextTiling": {"min_gap": 4, "warmup_gaps": 3},        # =pub (15셀 중 1위)
        "GraphSeg": {"window_d": 10},                          # =pub (1위는 경계 0개 퇴화 → 미채택)
        "GreedySeg": {},                                       # 상수 5개 전부 발표값
        "CSM": {"alpha": 1.0, "delay": 2, "min_gap": 2},       # =pub, 탐색 안 함 (D-B41 부칙)
    },
    "dialogue": {
        "TextTiling": {"min_gap": 2, "warmup_gaps": 3},        # 탐색 (pub 대비 +0.0013)
        "GraphSeg": {"window_d": 10},                          # =pub
        "GreedySeg": {},
        "CSM": {"alpha": 1.0, "delay": 2, "min_gap": 2},       # =pub
    },
}
CSM_BACKBONE = "bert-base-uncased"    # 🔒 SuperDialseg `CSMSegmenter` 백본 = 우리 패널 전체와 같은 계보


# 🔒 D-B42 — CSM 체크포인트 선택 프로토콜을 **두 가지 다** 보고한다.
#   CSM          = 마지막 체크포인트 (원 repo 관행: 정해진 epoch 수만큼 돌리고 끝난 것을 쓴다)
#   CSM-devbest  = superseg/validation 60대화에서 Score 최대인 체크포인트
#                  (RoBERTa 가 받은 것과 **같은** model-selection 프로토콜 — 비대칭 제거용)
# devbest step 은 `csm_ckpt_sweep_bert-base-uncased.json` 240점 중 최대값에서 왔다.
# 🔴 선택셋 = superseg/**validation**. test 아님. dev-3 도 아님(서사 선택셋 개봉 금지).
CSM_DEVBEST_STEP = 108620

# 🔒 D-B62 — CSM 세 번째 선택 프로토콜: **오프라인 논문 재현으로 고른 체크포인트**.
#   사용자 지시("오프라인으로 정확했던 학습본을 온라인으로 돌려라")를 그대로 구현한 것이다.
#   선택셋 = CSM **자기 논문 벤치의 공식 test**(tiage·dialseg711·superseg), 우리 dev-3·시험지 아님.
#   사전 규칙 = 3셋 **평균 오프라인 Score 최대**. 산출 = `methods/CSM/offline/whole_dialogue.py`
#   11개 후보(에폭 말 10 + devbest) 스윕, 요약 = 원장 D-B62.
CSM_OFFVERIFIED_STEP = 175335


def csm_ckpt(step: int | None = None) -> str:
    """step=None 이면 가장 진행된 체크포인트. 학습 미완이면 그 step 수를 표에 병기한다."""
    d = REPO / "external/Dialogue-Topic-Segmenter/checkpoints"
    if step is not None:
        p = d / f"cpt_{step}.pth"
        if not p.exists():
            raise FileNotFoundError(f"CSM 체크포인트 없음: {p}")
        return str(p)
    cks = sorted(d.glob("cpt_*.pth"), key=lambda p: int(re.search(r"(\d+)", p.name).group(1)))
    if not cks:
        raise FileNotFoundError(f"CSM 체크포인트 없음: {d} (학습 미완)")
    return str(cks[-1])



# ── Membox Topic Loom (D-B59/D-B60) ──────────────────────────────────────────
MEMBOX_MODEL = "openrouter/openai/gpt-4o-mini"
MEMBOX_CACHE = OUT / "cache_membox_topicloom.jsonl"


def preds_membox(data, ds: str) -> tuple[dict[str, np.ndarray], float]:
    """Membox 의 **Topic Loom** 만 떼어낸 온라인 경계 판정기 (arXiv 2601.03785v3).

    논문 명세 그대로 — 창 = 직전 2 문장, 라벨 {Yes, Partially Shifted, No},
    **Partially Shifted 도 경계**, 최소문맥 규칙(새 박스는 1개를 무조건 흡수 후 판정 재개).
    하이퍼는 없다(논문이 전부 지정) → 작동점 동결 대상 아님.

    🔴 **원 논문에 이 무대(분절)의 수치가 아예 없다.** 논문은 Pk/WD/경계F1 을 일절 보고하지
    않고 LoCoMo/DialSim QA 로만 평가한다(D-B59 §1). 따라서 이 행은 **우리 이식판**이며
    원 논문 수치와 대응시킬 대상이 없다.

    🔴 서사(dev-3)에서는 도메인 명사만 교체한다 — message→sentence. SeCom 에 적용한 규칙과
    동일(§5.5): 원문 그대로 두면 대화 전용 어휘라 **불리한 허수아비**가 된다. 기준·출력형식은 불변.
    """
    import hashlib
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(REPO / ".env")
    hdr = {}
    if os.environ.get("CF_ACCESS_CLIENT_ID"):
        hdr["CF-Access-Client-Id"] = os.environ["CF_ACCESS_CLIENT_ID"]
        hdr["CF-Access-Client-Secret"] = os.environ["CF_ACCESS_CLIENT_SECRET"]
    cl = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_BASE_URL"],
                default_headers=hdr or None, timeout=120.0, max_retries=0)

    tmpl = re.sub(r"<!--.*?-->", "",
                  (REPO / "scripts/segmentation_prompts/membox/msg_continuation.md").read_text(),
                  flags=re.S).strip()
    if ds == "dev3" or ds.startswith("test3"):          # 서사 무대 = 도메인 명사만 교체
        tmpl = tmpl.replace("message", "sentence").replace("messages", "sentences")

    lock = threading.Lock()
    cache: dict[str, str] = {}
    if MEMBOX_CACHE.exists():
        for line in MEMBOX_CACHE.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                cache[r["k"]] = r["v"]
    fp = MEMBOX_CACHE.open("a")

    def ask(ref: list[str], curr: str) -> str:
        pr = tmpl.replace("`ref`", "\n".join(ref)).replace("`curr`", curr)
        k = hashlib.sha1(pr.encode()).hexdigest()
        with lock:
            if k in cache:
                return cache[k]
        out = ""
        for a in range(5):
            try:
                r = cl.chat.completions.create(model=MEMBOX_MODEL,
                                               messages=[{"role": "user", "content": pr}],
                                               max_tokens=12, temperature=0.0)
                out = (r.choices[0].message.content or "").strip()
                break
            except Exception:                                    # noqa: BLE001
                if a == 4:
                    return ""
                time.sleep(2 * (a + 1))
        with lock:
            cache[k] = out
            fp.write(json.dumps({"k": k, "v": out}) + "\n")
            fp.flush()
        return out

    def run(item):
        did, sents, gold = item
        v = np.zeros(len(gold), np.int8)
        cur: list[str] = []
        just = False
        for i, s in enumerate(sents):
            if not cur:
                cur, just = [s], True
                continue
            if just:                       # 최소문맥 규칙
                cur.append(s)
                just = False
                continue
            raw = ask(cur[-2:], s).lower()
            seal = ("partial" in raw) or bool(re.search(r"\bno\b", raw)) \
                or not re.search(r"\byes\b", raw)
            if seal:
                if 1 <= i <= len(gold):
                    v[i - 1] = 1           # 전이 i-1 = (sent_{i-1} -> sent_i)
                cur, just = [s], True
            else:
                cur.append(s)
        return did, v

    t0 = time.perf_counter()
    out: dict[str, np.ndarray] = {}
    with ThreadPoolExecutor(max_workers=48) as ex:
        for did, v in ex.map(run, data):
            out[did] = v
    fp.close()
    n_turn = sum(len(s) for _, s, _ in data)
    return out, (time.perf_counter() - t0) / max(n_turn, 1) * 1000.0


# ── baseline (전부 `push(utterance) -> list[int]` 스트리밍 API) ────────────────
def preds_baseline(kind: str, data, device: str, ds: str) -> tuple[dict[str, np.ndarray], float]:
    # test3(봉인 시험지)은 서사이므로 dev-3 와 같은 서사 작동점을 쓴다(동결값, 재탐색 금지).
    dom = "narrative" if ds == "dev3" or ds.startswith("test3") else "dialogue"
    hp = FROZEN[dom][kind.split("-")[0] if kind.startswith("CSM") else kind]
    if kind == "TextTiling":
        from hi_ontop.baselines.texttiling_streaming import StreamingTextTiling as C
        mk = lambda: C(**hp)                                                # noqa: E731
    elif kind == "GraphSeg":
        from hi_ontop.baselines.graphseg_window import GraphSegWindowD as C
        mk = lambda: C(**hp)                                                # noqa: E731
    elif kind == "GreedySeg":
        from hi_ontop.baselines.greedyseg_delay2 import GreedySegOnlineDelay2 as C
        mk = lambda: C(device=device, **hp)                                 # noqa: E731
    elif kind in ("CSM", "CSM-devbest", "CSM-offverified"):
        from hi_ontop.baselines.csm_online import CSMOnlineDelay2 as C
        ck = csm_ckpt({"CSM-devbest": CSM_DEVBEST_STEP,
                       "CSM-offverified": CSM_OFFVERIFIED_STEP}.get(kind))
        mk = lambda: C(device=device, backbone=CSM_BACKBONE,                # noqa: E731
                       ckpt_path=ck, **hp)
    else:
        raise ValueError(kind)
    out, t_tot, n_turn = {}, 0.0, 0
    for did, sents, gold in data:
        seg = mk()
        bset: set[int] = set()
        t0 = time.perf_counter()
        for u in sents:
            for b in seg.push(u):
                bset.add(int(b))
        t_tot += time.perf_counter() - t0
        n_turn += len(sents)
        # 경계 인덱스 규약 통일: baseline 은 "새 세그먼트를 여는 발화 번호" 를 준다.
        # 전이 j = (sent_j -> sent_{j+1}) 이므로 발화 b 가 새 세그먼트면 전이 b-1 이 경계.
        v = np.zeros(len(gold), np.int8)
        for b in bset:
            if 1 <= b <= len(gold):
                v[b - 1] = 1
        out[did] = v
    return out, (t_tot / max(n_turn, 1)) * 1000.0


# ── 채점 ──────────────────────────────────────────────────────────────────────
def score_A(data, preds: dict[str, np.ndarray]) -> dict:
    """🔒 주 지표 — 공식 `hi_ontop.dts_scoring` (CLAUDE.md 강제)."""
    ys = [np.concatenate([g, [0]]).tolist() for _, _, g in data]
    ps = [np.concatenate([preds[d], [0]]).tolist() for d, _, _ in data]
    r = score_dialogues(ys, ps)
    return {"score": r["score"], "pk": r["pk"], "wd": r["wd"], "f1": r["f1"],
            "n_pred": int(sum(int(preds[d].sum()) for d, _, _ in data))}


def score_B(data, preds: dict[str, np.ndarray]) -> dict:
    """[브릿지] `methods/` 하네스 — segeval 직접 + per-dialogue macro. **주 결과 아님.**"""
    pks, wds, f1s = [], [], []
    for did, sents, g in data:
        gb = [j for j, v in enumerate(g) if v]
        pb = [j for j, v in enumerate(preds[did]) if v]
        pk, wd = pk_wd(pb, gb, len(g) + 1)
        pks.append(pk)
        wds.append(wd)
        f1s.append(boundary_set_f1(pb, gb))
    pk, wd, f1 = float(np.mean(pks)), float(np.mean(wds)), float(np.mean(f1s))
    return {"score": 0.5 * f1 + 0.25 * (1 - pk) + 0.25 * (1 - wd), "pk": pk, "wd": wd, "f1": f1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="dev3")
    ap.add_argument("--methods", default="M5,M4,M1-CosDrift,TextTiling,GraphSeg,GreedySeg,CSM")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "preds").mkdir(exist_ok=True)
    res: dict = {}

    for ds in [x.strip() for x in args.datasets.split(",") if x.strip()]:
        data = load_dataset(ds)
        corp = Corpus(ds)
        assert corp.n_docs == len(data), f"{ds}: 문서 수 불일치 {corp.n_docs} vs {len(data)}"
        for i, (did, sents, g) in enumerate(data):
            n = int(corp.so[i + 1] - corp.so[i])
            assert n == len(sents), f"{ds}/{did}: 문장 수 불일치 {n} vs {len(sents)}"
        print(f"[{ds}] 문서 {len(data)} · 전이 {sum(len(g) for _,_,g in data)} · "
              f"gold {sum(int(g.sum()) for _,_,g in data)}", flush=True)

        for meth in [x.strip() for x in args.methods.split(",") if x.strip()]:
            t0 = time.perf_counter()
            try:
                if meth.endswith("-adaptive"):
                    pr, lat = preds_ours_adaptive(meth[:-9], data, corp, dev), None
                elif meth in ("M5", "M4"):
                    pr, lat = preds_ours(meth, data, corp, dev), None
                elif meth == "M1-CosDrift":
                    pr, lat = preds_m1(data, corp), None
                elif meth == "RoBERTa":
                    pr, lat = preds_roberta(data, dev)
                elif meth == "Membox-TopicLoom":
                    pr, lat = preds_membox(data, ds)
                else:
                    pr, lat = preds_baseline(meth, data, dev, ds)
            except Exception as e:                       # 실패는 표에 그대로 남긴다
                res[(ds, meth)] = {"error": f"{type(e).__name__}: {e}"}
                print(f"  🔴 {meth}: {type(e).__name__}: {e}", flush=True)
                continue
            (OUT / "preds" / f"{meth}_{ds}.json").write_text(json.dumps(
                {d: pr[d].tolist() for d in pr}), encoding="utf-8")
            A, B = score_A(data, pr), score_B(data, pr)
            res[(ds, meth)] = {"A": A, "B": B, "ms_per_turn": lat,
                               "wall_s": time.perf_counter() - t0}
            print(f"  {meth:14s} A-Score {A['score']:.4f} (Pk {A['pk']:.3f} WD {A['wd']:.3f} "
                  f"F1 {A['f1']:.3f}, 경계 {A['n_pred']})  | B {B['score']:.4f}"
                  + (f"  {lat:.2f} ms/turn" if lat else ""), flush=True)

    # 누적 병합 — 부분 재실행(예: GraphSeg 만)이 앞선 결과를 지우지 않게 한다.
    rp = OUT / "results.json"
    allr = json.loads(rp.read_text()) if rp.exists() else {}
    allr.update({f"{k[0]}|{k[1]}": v for k, v in res.items()})
    rp.write_text(json.dumps(allr, indent=2, default=float))
    print(f"\n-> {OUT}/results.json · 예측 {OUT}/preds/")


if __name__ == "__main__":
    main()
