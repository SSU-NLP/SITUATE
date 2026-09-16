#!/usr/bin/env python3
"""Membox 재현 — LoCoMo (arXiv 2601.03785v3, Tao et al., Tsinghua).

🔴 **공식 코드 없음.** Awesome-Memory-for-Agents / awesome-long-memory 둘 다 paper 링크만 등재.
   프롬프트 4종을 arXiv e-print LaTeX 부록에서 전사한 것이 유일한 1차 자료다
   (`scripts/segmentation_prompts/membox/`).

🔴 **왜 LoCoMo 까지 가는가.** 논문은 분절 지표(Pk/WD/경계F1)를 **일절 보고하지 않는다**
   (tiage/superseg 0회 · WindowDiff 0회 · dialseg 는 참고문헌 1건뿐). Topic Loom 은 논문 안에서
   독립 채점된 적이 없고 **QA 성능으로만 간접 평가**된다. 따라서 구현 충실도를 검증할 수 있는
   유일한 지점이 LoCoMo QA 다.

논문 명세 (§Topic Loom · §Trace Weaver · §Experiments):
  - 슬라이딩 윈도 = **연속 2 메시지**, 현재 미봉인 박스의 최근분
  - 라벨 ∈ {continuous, partial shift, discontinuous}. **partial shift 도 봉인**
  - 최소문맥 규칙 = 새 박스는 가능하면 인접 응답 1개 이상 포함(그 후 판정 재개)
  - 봉인 시 P_extract → {messages, topic, events, keywords}
  - Trace Weaver = 이벤트별 최근접 trace 투표 → 후보집합 → P_verify 일괄검증 → 미연결분 P_init
  - 검색 = text-embedding-3-small, 박스 top-k=10 / (Trace 모드) 이벤트 top-k=2
  - 백본 = GPT-4o / GPT-4o-mini (구축·QA 동일 백본)

단계:
  build → memory/{sid}.json   (Topic Loom + 추출 + Trace Weaver)
  qa    → qa/{mode}_{tag}_{sid}.json
  score → results_{tag}.json  (카테고리별 F1 / BLEU-1, adversarial 제외)

사용:
  .venv/bin/python scripts/eacl_phase0/membox_repro.py --stage build --limit 1     # 스모크
  .venv/bin/python scripts/eacl_phase0/membox_repro.py --stage build
  .venv/bin/python scripts/eacl_phase0/membox_repro.py --stage qa --mode compact
  .venv/bin/python scripts/eacl_phase0/membox_repro.py --stage qa --mode trace
  .venv/bin/python scripts/eacl_phase0/membox_repro.py --stage score
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

REPO = Path(__file__).resolve().parents[2]
PROMPTS = REPO / "scripts/segmentation_prompts/membox"
LOCOMO = REPO / "benchmarks/locomo/locomo10.json"
OUT = REPO / "outputs/experiments/2026-08-01_membox_repro"

# 🔒 논문 §Experiments 고정값
BOX_TOPK = 10
EVENT_TOPK = 2
EMBED_MODEL = "openrouter/openai/text-embedding-3-small"
CATNAME = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
SKIP_CATEGORY = {5}          # adversarial — 정답 라벨 부재로 논문도 제외

# LoCoMo QA 프롬프트 = Mem0 계보(nemori `evaluation/locomo/search.py::ANSWER_PROMPT`).
# 논문은 "QA 는 원 벤치마크 프롬프트를 따른다"고만 밝히고 부록에 싣지 않았다.
# Membox 가 대조한 Mem0 와 **같은 답변 프롬프트**를 써야 F1 이 비교 가능하다(토큰중첩 F1 은
# 답변 장황도에 극도로 민감). episodic/semantic 2분할만 우리 메모리 구조에 맞게 1개로 합쳤다.
ANSWER_PROMPT = """You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

# CONTEXT:
You have access to memories from two speakers in a conversation. These memories contain
timestamped information that may be relevant to answering the question.

# INSTRUCTIONS:
1. Carefully analyze all provided memories from both speakers
2. Pay special attention to the timestamps to determine the answer
3. If the question asks about a specific event or fact, look for direct evidence in the memories
4. If the memories contain contradictory information, prioritize the most recent memory
5. If there is a question about time references (like "last year", "two months ago", etc.),
   calculate the actual date based on the memory timestamp. For example, if a memory from
   4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
6. Always convert relative time references to specific dates, months, or years. For example,
   convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory
   timestamp. Ignore the reference while answering the question.
7. Focus only on the content of the memories from both speakers. Do not confuse character
   names mentioned in memories with the actual users who created those memories.
8. The answer should be less than 5-6 words.

# APPROACH (Think step by step):
1. First, examine all memories that contain information related to the question
2. Examine the timestamps and content of these memories carefully
3. Look for explicit mentions of dates, times, locations, or events that answer the question
4. If the answer requires calculation (e.g., converting relative time references), show your work
5. Formulate a precise, concise answer based solely on the evidence in the memories
6. Double-check that your answer directly addresses the question asked
7. Ensure your final answer is specific and avoids vague time references

Memories:
{memories}

Question: {question}

Answer:
"""

_lock = threading.Lock()
_cache: dict[str, str] = {}
_cache_fp = None


# ── LLM / 임베딩 ─────────────────────────────────────────────────────────────
def make_client() -> OpenAI:
    load_dotenv(REPO / ".env")
    hdr = {}
    if os.environ.get("CF_ACCESS_CLIENT_ID"):
        hdr["CF-Access-Client-Id"] = os.environ["CF_ACCESS_CLIENT_ID"]
        hdr["CF-Access-Client-Secret"] = os.environ["CF_ACCESS_CLIENT_SECRET"]
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"],
                  base_url=os.environ["OPENAI_BASE_URL"],
                  default_headers=hdr or None, timeout=180.0, max_retries=0)


def cache_open(tag: str) -> None:
    """호출 캐시 — 중단·재개해도 API 를 다시 태우지 않는다."""
    global _cache_fp
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"cache_{tag}.jsonl"
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                    _cache[r["k"]] = r["v"]
                except Exception:
                    pass
    _cache_fp = p.open("a")
    print(f"[cache] {p.name} 적재 {len(_cache)}건", flush=True)


def llm(client: OpenAI, model: str, prompt: str, max_tokens: int, retries: int = 5) -> str:
    key = hashlib.sha1(f"{model}\x00{max_tokens}\x00{prompt}".encode()).hexdigest()
    with _lock:
        if key in _cache:
            return _cache[key]
    out = ""
    for a in range(retries):
        try:
            r = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens, temperature=0.0)
            out = (r.choices[0].message.content or "").strip()
            break
        except Exception as e:                                   # noqa: BLE001
            if a == retries - 1:
                print(f"[llm] 실패 {type(e).__name__} {str(e)[:110]}", flush=True)
                return ""
            time.sleep(2 * (a + 1))
    with _lock:
        _cache[key] = out
        if _cache_fp:
            _cache_fp.write(json.dumps({"k": key, "v": out}) + "\n")
            _cache_fp.flush()
    return out


def embed(client: OpenAI, texts: list[str], batch: int = 128) -> np.ndarray:
    """텍스트 → L2 정규화 임베딩 행렬. 코사인 유사도 = 내적."""
    vs: list[list[float]] = []
    for i in range(0, len(texts), batch):
        chunk = [t[:8000] if t.strip() else "empty" for t in texts[i:i + batch]]
        for a in range(5):
            try:
                r = client.embeddings.create(model=EMBED_MODEL, input=chunk)
                vs.extend(d.embedding for d in r.data)
                break
            except Exception as e:                               # noqa: BLE001
                if a == 4:
                    raise RuntimeError(f"임베딩 실패: {e}") from e
                time.sleep(2 * (a + 1))
    m = np.asarray(vs, dtype=np.float32)
    return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)


def load_prompt(name: str) -> str:
    """전사 프롬프트 적재. HTML 주석(출처 표기)만 제거하고 본문은 그대로 쓴다."""
    t = (PROMPTS / name).read_text()
    return re.sub(r"<!--.*?-->", "", t, flags=re.S).strip()


def parse_json(text: str) -> dict | None:
    """LLM 출력에서 JSON 객체 추출 (코드펜스·머리말 허용)."""
    if not text:
        return None
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(t)
    except Exception:                                            # noqa: BLE001
        pass
    d = 0
    start = -1
    for i, ch in enumerate(t):
        if ch == "{":
            if d == 0:
                start = i
            d += 1
        elif ch == "}":
            d -= 1
            if d == 0 and start >= 0:
                try:
                    return json.loads(t[start:i + 1])
                except Exception:                                # noqa: BLE001
                    start = -1
    return None


# ── 데이터 ───────────────────────────────────────────────────────────────────
def load_messages(sample: dict) -> list[dict]:
    """LoCoMo 대화를 세션 순서대로 평탄화. 세션 날짜를 각 메시지에 붙인다."""
    conv = sample["conversation"]
    idx = sorted(int(k.split("_")[1]) for k in conv
                 if re.fullmatch(r"session_\d+", k) and isinstance(conv[k], list))
    msgs: list[dict] = []
    for si in idx:
        date = conv.get(f"session_{si}_date_time", "")
        for t in conv[f"session_{si}"]:
            msgs.append({"dia_id": t.get("dia_id", ""), "speaker": t.get("speaker", ""),
                         "text": t.get("text", ""), "session": si, "date": date})
    return msgs


def msg_line(m: dict) -> str:
    return f"[{m['date']}] {m['speaker']}: {m['text']}"


# ── ① Topic Loom ─────────────────────────────────────────────────────────────
def classify_continuity(client: OpenAI, model: str, window: list[dict], cur: dict,
                        tmpl: str) -> str:
    """PROMPT_MSG_CONTINUATION → {continuous, partial shift, discontinuous}.

    🔴 파싱 규칙은 논문 미명시 = 우리 결정. "Partially Shifted" 를 먼저 검사한다
    (뒤로 미루면 'Partially Shifted' 안의 다른 토큰에 먼저 걸릴 수 있다).
    """
    p = (tmpl.replace("`ref`", "\n".join(msg_line(m) for m in window))
             .replace("`curr`", msg_line(cur)))
    raw = llm(client, model, p, max_tokens=12).strip().lower()
    if "partial" in raw:
        return "partial shift"
    if re.search(r"\byes\b", raw):
        return "continuous"
    if re.search(r"\bno\b", raw):
        return "discontinuous"
    return "unparsed"        # 형식 이탈 — 봉인 쪽으로 처리하고 개수를 세어 보고한다


def topic_loom(client: OpenAI, model: str, msgs: list[dict], tmpl: str) -> tuple[list[list[dict]], dict]:
    """온라인 박스 분할. 반환 = (박스별 메시지 목록, 라벨 분포).

    논문: continuous 면 현재 박스에 추가, partial shift/discontinuous 면 봉인 후 새 박스.
    최소문맥 규칙 — 새 박스는 인접 응답 1개를 무조건 포함한 뒤 판정을 재개한다.
    """
    boxes: list[list[dict]] = []
    cur: list[dict] = []
    just_opened = False
    labels: dict[str, int] = {}
    for m in msgs:
        if not cur:
            cur, just_opened = [m], True
            continue
        if just_opened:                       # 최소문맥 규칙: 판정 없이 1개 부착
            cur.append(m)
            just_opened = False
            continue
        lab = classify_continuity(client, model, cur[-2:], m, tmpl)
        labels[lab] = labels.get(lab, 0) + 1
        if lab == "continuous":
            cur.append(m)
        else:
            boxes.append(cur)
            cur, just_opened = [m], True
    if cur:
        boxes.append(cur)
    return boxes, labels


def m5_boxes(msgs: list[dict], ckpt: str | None = None) -> tuple[list[list[dict]], dict]:
    """🔄 Topic Loom 을 **M5-FactoredSlot 로 교체**한 판 (D-B60).

    Membox 의 나머지 층(추출·Trace Weaver·검색·QA)은 **한 글자도 바꾸지 않는다.**
    그래야 차이가 곧 **경계 판정기의 기여**가 된다.

    🔒 작동점은 앵커 val 5-fold OOF 가 고른 동결값(fixed_quantile r=0.0642, D-B29/D-B30)을
    그대로 쓴다 — LoCoMo 를 보고 고르지 않는다. 서사에서 동결한 밀도를 대화에 그대로 들고 가는
    것이므로 **과소분절이 예상되며 그것이 정직한 배포 조건**이다.
    """
    from hi_ontop.m5_default import segment                              # noqa: PLC0415
    r = segment([m["text"] for m in msgs], checkpoint=ckpt)
    cuts = [0] + list(r["boundaries"]) + [len(msgs)]
    boxes = [msgs[a:b] for a, b in zip(cuts[:-1], cuts[1:]) if b > a]
    return boxes, {"m5_boundaries": len(r["boundaries"])}


def extract_box(client: OpenAI, model: str, box: list[dict], tmpl: str) -> dict:
    """PROMPT_DIALOG_EXTRACT → {keywords, topic, explicit_mentions}."""
    text = "\n".join(msg_line(m) for m in box)
    d = parse_json(llm(client, model, tmpl.replace("{text}", text), max_tokens=900)) or {}
    ev = d.get("explicit_mentions") or []
    return {"topic": str(d.get("topic") or ""),
            "keywords": [str(x) for x in (d.get("keywords") or []) if str(x).strip()],
            "events": [str(x) for x in ev if str(x).strip()],
            "extract_ok": bool(d)}


# ── ② Trace Weaver ───────────────────────────────────────────────────────────
def trace_init(client: OpenAI, model: str, events: list[str], tmpl: str,
               isolated: str = "singleton") -> list[list[str]]:
    """PROMPT_TRACE_INIT → 새 trace 목록.

    🔴 논문 미명시 = 우리 결정. `isolated` 로 두 판을 모두 돌려 보고한다:
      singleton — isolated_events 를 **각각 단일 이벤트 trace** 로 남긴다(초판).
                  ⚠️ 이 판은 단일박스 trace 를 양산한다(실측 59.6%) → trace 확장이 중복이 되어
                  Trace 모드 이득이 사라진다(D-B57 §2.2).
      drop      — isolated_events 를 **trace 로 만들지 않는다.** 사슬을 이루지 못한 이벤트는
                  macro-topic 이 아니라는 해석. 박스 안에는 그대로 남아 검색된다.
    """
    if not events:
        return []
    d = parse_json(llm(client, model, tmpl.replace("{events}", json.dumps(events, ensure_ascii=False)),
                       max_tokens=1400)) or {}
    out: list[list[str]] = []
    pc = [str(x) for x in (d.get("primary_chain") or []) if str(x).strip()]
    if pc:
        out.append(pc)
    for sc in (d.get("secondary_chains") or []):
        s = [str(x) for x in (sc or []) if str(x).strip()]
        if s:
            out.append(s)
    if isolated == "singleton":
        for iso in (d.get("isolated_events") or []):
            if str(iso).strip():
                out.append([str(iso)])
    if not out and isolated == "singleton":   # 파싱 실패 시 전체를 1 trace 로 (손실 방지)
        out = [events]
    return out


def trace_verify(client: OpenAI, model: str, chain: list[str], new: list[str],
                 tmpl: str) -> list[str]:
    """PROMPT_TRACE_EVENT_FILTER → chain 에 붙일 이벤트."""
    p = (tmpl.replace("{content_a}", json.dumps(chain, ensure_ascii=False))
             .replace("{content_b}", json.dumps(new, ensure_ascii=False)))
    d = parse_json(llm(client, model, p, max_tokens=1600)) or {}
    rel = {str(x).strip() for x in (d.get("related_events") or [])}
    return [e for e in new if e in rel]        # 원문 이벤트만 채택(LLM 재작성 방지)


def weave(client: OpenAI, model: str, boxes: list[dict], t_init: str, t_ver: str,
          client_emb: OpenAI, isolated: str = "singleton") -> list[dict]:
    """이벤트별 최근접 trace 투표 → 후보집합 → 일괄 검증 → 미연결분 재초기화."""
    traces: list[dict] = []                    # {"events":[...], "boxes":[bi]}
    tvecs: list[np.ndarray] = []               # trace 별 이벤트 임베딩 행렬
    for bi, b in enumerate(boxes):
        ev = b["events"]
        if not ev:
            continue
        if not traces:
            for ch in trace_init(client, model, ev, t_init, isolated):
                traces.append({"events": ch, "boxes": [bi]})
            if traces:
                tvecs = [embed(client_emb, t["events"]) for t in traces]
            continue
        evec = embed(client_emb, ev)
        cand: set[int] = set()
        for k in range(len(ev)):               # 이벤트 1개 = trace 1표
            best = int(np.argmax([float(np.max(evec[k] @ tv.T)) for tv in tvecs]))
            cand.add(best)
        accepted: set[str] = set()
        for ti in sorted(cand):
            rel = trace_verify(client, model, traces[ti]["events"], ev, t_ver)
            if rel:
                traces[ti]["events"].extend(rel)
                traces[ti]["boxes"].append(bi)
                tvecs[ti] = embed(client_emb, traces[ti]["events"])
                accepted |= set(rel)
        unlinked = [e for e in ev if e not in accepted]
        for ch in trace_init(client, model, unlinked, t_init, isolated):
            traces.append({"events": ch, "boxes": [bi]})
            tvecs.append(embed(client_emb, ch))
    return traces


# ── 검색용 표현 ──────────────────────────────────────────────────────────────
def box_repr(b: dict) -> str:
    """박스 표현 = messages · topic · events · keywords (논문 §Experiments).

    🔴 4개 필드의 결합 방식은 논문 미명시 = 우리 결정: 하나의 문자열로 이어 붙여 단일 임베딩.
    """
    return (f"Topic: {b['topic']}\nKeywords: {', '.join(b['keywords'])}\n"
            f"Events: {' | '.join(b['events'])}\nMessages:\n" + "\n".join(b["messages"]))


def box_context(b: dict) -> str:
    return (f"[Box] topic: {b['topic']}\nevents: {' | '.join(b['events'])}\n"
            + "\n".join(b["messages"]))


# ── 단계 ─────────────────────────────────────────────────────────────────────
def ns(args) -> str:
    """조건 네임스페이스 — 기존 산출을 덮지 않는다."""
    s = "" if args.segmenter == "topicloom" else f"_{args.segmenter}"
    if args.segmenter == "m5" and getattr(args, "m5_label", ""):
        s += f"_{args.m5_label}"        # 체크포인트별 네임스페이스 — 기존 산출을 덮지 않는다
    return s + ("" if args.isolated == "singleton" else f"_iso{args.isolated}")


def stage_build(args) -> None:
    client = make_client()
    cache_open(f"build_{args.tag}")
    p_cont, p_ext = load_prompt("msg_continuation.md"), load_prompt("dialog_extract.md")
    p_init, p_ver = load_prompt("trace_init.md"), load_prompt("trace_event_filter.md")
    data = json.loads(LOCOMO.read_text())
    if args.limit:
        data = data[:args.limit]
    mdir = OUT / f"memory{ns(args)}"
    mdir.mkdir(parents=True, exist_ok=True)

    def run(sample: dict) -> str:
        sid = sample["sample_id"]
        dst = mdir / f"{sid}.json"
        if dst.exists() and not args.overwrite:
            return f"{sid} 건너뜀(존재)"
        t0 = time.perf_counter()
        msgs = load_messages(sample)
        if args.segmenter == "m5":
            raw_boxes, labels = m5_boxes(msgs, args.m5_ckpt or None)
        else:
            raw_boxes, labels = topic_loom(client, args.model, msgs, p_cont)
        boxes = []
        for bx in raw_boxes:
            e = extract_box(client, args.model, bx, p_ext)
            boxes.append({"messages": [msg_line(m) for m in bx],
                          "dia_ids": [m["dia_id"] for m in bx],
                          "date": bx[0]["date"], "session": bx[0]["session"], **e})
        traces = weave(client, args.model, boxes, p_init, p_ver, client, args.isolated)
        dst.write_text(json.dumps({
            "sample_id": sid, "n_msg": len(msgs), "n_box": len(boxes),
            "n_trace": len(traces), "labels": labels,
            "extract_fail": sum(1 for b in boxes if not b["extract_ok"]),
            "boxes": boxes, "traces": traces}, ensure_ascii=False, indent=1))
        return (f"{sid} 메시지 {len(msgs)} → 박스 {len(boxes)} · trace {len(traces)} · "
                f"라벨 {labels} · {time.perf_counter()-t0:.0f}s")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for f in as_completed([ex.submit(run, s) for s in data]):
            print("[build]", f.result(), flush=True)


def stage_qa(args) -> None:
    client = make_client()
    cache_open(f"qa_{args.mode}_{args.tag}")
    data = json.loads(LOCOMO.read_text())
    if args.limit:
        data = data[:args.limit]
    (OUT / "qa").mkdir(parents=True, exist_ok=True)

    for sample in data:
        sid = sample["sample_id"]
        src = OUT / f"memory{ns(args)}" / f"{sid}.json"
        dst = OUT / "qa" / f"{args.mode}_{args.tag}{ns(args)}_{sid}.json"
        if not src.exists():
            print(f"[qa] {sid} 메모리 없음 — build 먼저", flush=True)
            continue
        if dst.exists() and not args.overwrite:
            print(f"[qa] {sid} 건너뜀(존재)", flush=True)
            continue
        mem = json.loads(src.read_text())
        boxes, traces = mem["boxes"], mem["traces"]
        bvec = embed(client, [box_repr(b) for b in boxes])
        # 이벤트 → 소속 trace 색인
        ev2tr: dict[str, list[int]] = {}
        for ti, t in enumerate(traces):
            for e in t["events"]:
                ev2tr.setdefault(e, []).append(ti)

        qs = [q for q in sample.get("qa", []) if int(q.get("category", 0)) not in SKIP_CATEGORY]
        qvec = embed(client, [str(q["question"]) for q in qs])

        def answer(i: int) -> dict:
            q = qs[i]
            order = np.argsort(-(bvec @ qvec[i]))[:BOX_TOPK]
            top = [boxes[j] for j in order]
            parts = [box_context(b) for b in top]
            used_tr: list[int] = []
            if args.mode == "trace":
                cand = [e for b in top for e in b["events"]]
                if cand:
                    cvec = embed(client, cand)
                    for k in np.argsort(-(cvec @ qvec[i]))[:EVENT_TOPK]:
                        used_tr.extend(ev2tr.get(cand[int(k)], []))
                    seen: set[int] = set()
                    for ti in used_tr:
                        if ti not in seen:
                            seen.add(ti)
                            parts.append("[Trace] " + " -> ".join(traces[ti]["events"]))
            ctx = "\n\n".join(parts)
            pred = llm(client, args.model,
                       ANSWER_PROMPT.format(memories=ctx, question=q["question"]),
                       max_tokens=args.max_answer_tokens)
            return {"question": q["question"], "answer": str(q.get("answer", "")),
                    "category": int(q["category"]), "prediction": pred,
                    "n_box": len(top), "n_trace": len(set(used_tr)), "ctx_chars": len(ctx)}

        rows: list[dict] = []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(answer, i): i for i in range(len(qs))}
            for f in as_completed(futs):
                rows.append(f.result())
        dst.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
        print(f"[qa] {sid} {len(rows)}문항 · mode={args.mode} → {dst.name}", flush=True)


def stage_score(args) -> None:
    sys.path.insert(0, str(REPO / "external/nemori/evaluation/locomo"))
    from metrics.utils import calculate_metrics                  # noqa: PLC0415

    res: dict[str, Any] = {}
    for mode in ("compact", "trace"):
        files = sorted((OUT / "qa").glob(f"{mode}_{args.tag}{ns(args)}_*.json"))
        if not files:
            continue
        by: dict[int, list[dict]] = {}
        ctx: list[int] = []
        for fp in files:
            for r in json.loads(fp.read_text()):
                m = calculate_metrics(r["prediction"], r["answer"])
                by.setdefault(r["category"], []).append(m)
                ctx.append(r["ctx_chars"])
        row = {"n_sample": len(files), "mean_ctx_chars": float(np.mean(ctx)) if ctx else 0.0}
        for cat in sorted(by):
            v = by[cat]
            row[CATNAME[cat]] = {
                "n": len(v),
                "f1": round(100 * float(np.mean([x["f1"] for x in v])), 2),
                "bleu1": round(100 * float(np.mean([x["bleu1"] for x in v])), 2)}
        res[mode] = row
    (OUT / f"results_{args.tag}{ns(args)}.json").write_text(json.dumps(res, indent=1))

    pub = {  # 논문 Table 1 (재현 대조 목표)
        "gpt-4o-mini": {"compact": (39.88, 58.03, 27.96, 60.09),
                        "trace": (41.19, 59.63, 30.36, 61.18)},
        "gpt-4o": {"compact": (48.35, 65.06, 30.61, 61.69),
                   "trace": (50.48, 66.61, 38.77, 62.56)}}
    cols = ["multi-hop", "temporal", "open-domain", "single-hop"]
    print(f"\n{'mode':9} {'':6} " + " ".join(f"{c:>12}" for c in cols))
    for mode, row in res.items():
        got = [row.get(c, {}).get("f1") for c in cols]
        print(f"{mode:9} {'우리':6} " + " ".join(f"{g:>12.2f}" if g is not None else f"{'—':>12}"
                                                for g in got))
        p = pub.get(args.tag, {}).get(mode)
        if p:
            print(f"{'':9} {'논문':6} " + " ".join(f"{x:>12.2f}" for x in p))
            print(f"{'':9} {'Δ':6} " + " ".join(
                f"{(g - x):>+12.2f}" if g is not None else f"{'—':>12}" for g, x in zip(got, p)))
    print(f"\n-> {OUT / f'results_{args.tag}{ns(args)}.json'}")

    # 연속성 라벨 분포 — 🔴 논문이 보고하지 않은 수치
    agg: dict[str, int] = {}
    nb = nt = 0
    for fp in sorted((OUT / f"memory{ns(args)}").glob("*.json")):
        m = json.loads(fp.read_text())
        for k, v in m["labels"].items():
            agg[k] = agg.get(k, 0) + v
        nb += m["n_box"]
        nt += m["n_trace"]
    if agg:
        tot = sum(agg.values())
        print("\n[연속성 라벨 분포 — 논문 미보고]")
        for k, v in sorted(agg.items(), key=lambda x: -x[1]):
            print(f"  {k:16} {v:6d}  {100*v/tot:5.1f}%")
        print(f"  총 판정 {tot} · 박스 {nb} · trace {nt}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["build", "qa", "score"])
    ap.add_argument("--model", default="openrouter/openai/gpt-4o-mini")
    ap.add_argument("--tag", default="gpt-4o-mini", help="산출 네임스페이스(백본별)")
    ap.add_argument("--mode", default="compact", choices=["compact", "trace"])
    ap.add_argument("--limit", type=int, default=0, help="샘플 수 제한(스모크)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-answer-tokens", type=int, default=120)
    ap.add_argument("--m5-ckpt", default="", help="M5 체크포인트 경로(기본 = 동결본)")
    ap.add_argument("--m5-label", default="", help="체크포인트 네임스페이스 꼬리표")
    ap.add_argument("--segmenter", default="topicloom", choices=["topicloom", "m5"],
                    help="경계 판정기 교체 — 나머지 층은 동일 (D-B60)")
    ap.add_argument("--isolated", default="singleton", choices=["singleton", "drop"],
                    help="P_init 의 isolated_events 처리 (논문 미명시 · D-B57 §2.2)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    {"build": stage_build, "qa": stage_qa, "score": stage_score}[args.stage](args)


if __name__ == "__main__":
    main()
