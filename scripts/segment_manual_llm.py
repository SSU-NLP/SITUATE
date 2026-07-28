#!/usr/bin/env python3
"""LLM 있는 우리 방법론 — two-stage spike-gated LLM judge (사용자 직접-입력).

우리 방법론(Hi-OnTop) 의 **LLM 사용 버전**. LLM 을 처음부터 끝까지 분절에 쓰는
베이스라인(SeCom)이 아니라, **우리 신호가 임계를 넘은 후보 turn 에서만 LLM 을 호출**한다.
(정식 러너 `scripts/llm_judge_universal.py` 의 manual-입력판, 2026-06-20 방향.)

Stage 1 — 후보 생성 (우리 신호, 0-lag, LLM 0콜):
  δ_eff peak+prominence. s_t 가 직전보다 상승 AND 최근 W턴 robust baseline
  (median + k·1.4826·MAD) 초과 → 후보 경계. 발화율 stream-적응, look-ahead 없음.

Stage 2 — LLM 판정 (후보에서만):
  각 후보에서 [과거 맥락(0-lag) + 현재 발화] 를 LLM 에 주고 "NEW / SAME" 이진 판정.
  system 프롬프트 = ``segmentation_prompts/judge_binary_neutral_v1.md`` (도메인중립).
  경계 = NEW 판정 후보. temperature 0.

→ 버퍼드 LLM 대비 호출 대폭 절감(후보 ~10-20% turn 만) + 0-lag. 임베딩 인코더는
후보 생성에 무관함이 확인됐고(러너 주석), 여기선 한국어 지원 다국어 인코더 기본.

API 자격증명은 ``.env`` (``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` / 선택 CF Access).

usage:
  uv run python scripts/segment_manual_llm.py                    # 대화형 입력
  uv run python scripts/segment_manual_llm.py --file conv.txt
  uv run python scripts/segment_manual_llm.py --model openrouter/openai/gpt-4o-mini
  uv run python scripts/segment_manual_llm.py --peak-k 0.5 --ctx-k 0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hi_ontop.embedding import make_encoder      # noqa: E402
from hi_ontop.hi_ontop import HiOnTop            # noqa: E402
from segment_manual_common import print_segments, read_turns  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
# 후보 생성 인코더 무관(러너 검증) → 한국어 지원 다국어 인코더 기본 (segment_manual.py 와 통일)
DEFAULT_ENCODER = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
JUDGE_PROMPT = (REPO / "scripts/segmentation_prompts/judge_binary_neutral_v1.md").read_text(
    encoding="utf-8")
# δ_eff canonical config (run_encoder_comparison: M, RHO, A = 2, 0.7, 0.5)
M, RHO, A = 2, 0.7, 0.5


def delta_eff_seq(emb: np.ndarray) -> list[float]:
    """턴별 δ_eff 시퀀스 (HiOnTop 신호, delta_star=1.0 = 임계 미적용, 신호만 추출)."""
    seg = HiOnTop(dim=emb.shape[1], delta_star=1.0, ctx_window=M, ctx_decay=RHO, ctx_blend_a=A)
    for s in emb:
        seg.assign(s.astype(np.float64))
    return [float(h["delta_eff"]) for h in seg.history()]


def gen_peak(emb: np.ndarray, k: float = 0.5, W: int = 20) -> tuple[list[int], list[float]]:
    """Stage 1 후보 생성 — δ_eff peak+prominence (llm_judge_universal.gen_peak 와 동일).

    s_t 가 직전 상승 AND 최근 W턴 robust baseline(med + k·1.4826·MAD) 초과 → 후보. 0-lag.
    """
    s = np.asarray(delta_eff_seq(emb))
    n = len(emb)
    out: list[int] = []
    for t in range(2, n - 1):
        past = s[max(1, t - W):t]
        if len(past) < 3:
            continue
        med = float(np.median(past))
        mad = float(np.median(np.abs(past - med))) + 1e-9
        if s[t] > s[t - 1] and (s[t] - med) > k * 1.4826 * mad:
            out.append(t)
    return out, list(s)


def build_prompt(lines: list[str], t: int, k: int = 0) -> str:
    """과거 맥락(0-lag) + 현재 발화 강조. k<=0=full-past, k>0=최근 k턴만."""
    lo = 0 if k <= 0 else max(0, t - k)
    ctx = "\n".join(lines[lo:t])
    return (f"Conversation so far (most recent last):\n{ctx}\n\n"
            f">>> CURRENT utterance to judge:\n{lines[t]}\n\n"
            f"Does the CURRENT utterance begin a NEW topic, or continue the SAME topic? "
            f"Answer one word: NEW or SAME.")


def make_client():
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(REPO / ".env")
    if not os.environ.get("OPENAI_API_KEY") or not os.environ.get("OPENAI_BASE_URL"):
        sys.exit("[에러] .env 에 OPENAI_API_KEY / OPENAI_BASE_URL 가 필요합니다 "
                 "(.env.example 참조). LLM 없는 버전은 scripts/segment_manual.py 사용.")
    cf = None
    if os.environ.get("CF_ACCESS_CLIENT_ID") and os.environ.get("CF_ACCESS_CLIENT_SECRET"):
        cf = {"CF-Access-Client-Id": os.environ["CF_ACCESS_CLIENT_ID"],
              "CF-Access-Client-Secret": os.environ["CF_ACCESS_CLIENT_SECRET"]}
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"],
                  base_url=os.environ["OPENAI_BASE_URL"], default_headers=cf)


def judge(cli, model: str, prompt: str) -> tuple[int, str]:
    """LLM 이진 판정 → (1=NEW / 0=SAME, raw). temperature 0, 캐시."""
    cache_dir = REPO / "outputs/runs/_misc/llm_judge_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cp = cache_dir / (hashlib.md5(f"{model}|{prompt}".encode()).hexdigest() + ".json")
    if cp.exists():
        o = json.loads(cp.read_text())
        return o["v"], o["raw"]
    r = cli.chat.completions.create(
        model=model, temperature=0.0, max_tokens=4,
        messages=[{"role": "system", "content": JUDGE_PROMPT},
                  {"role": "user", "content": prompt}])
    raw = (r.choices[0].message.content or "").strip().upper()
    v = 1 if raw.startswith("NEW") else 0
    cp.write_text(json.dumps({"v": v, "raw": raw}))
    return v, raw


def main() -> int:
    ap = argparse.ArgumentParser(description="two-stage spike-gated LLM judge (우리 방법론+LLM)")
    ap.add_argument("--file", help="발화 파일 경로(줄당 1턴). 없으면 stdin/대화형.")
    ap.add_argument("--encoder", default=DEFAULT_ENCODER, help="후보생성 인코더 (기본 다국어)")
    ap.add_argument("--model", default="openrouter/openai/gpt-4o",
                    help="judge LLM 모델 id (기본 gpt-4o)")
    ap.add_argument("--peak-k", type=float, default=0.5,
                    help="후보 임계 = med + k·1.4826·MAD 의 k (기본 0.5, 클수록 후보 적음)")
    ap.add_argument("--ctx-k", type=int, default=0,
                    help="judge 맥락 턴수 (0=full-past, >0=최근 k턴만 = 토큰 절감)")
    args = ap.parse_args()

    turns = read_turns(args.file)
    if len(turns) < 3:
        print("발화가 3턴 이상 필요합니다.", file=sys.stderr)
        return 1
    lines = [f"[{t['speaker']}] {t['text']}" for t in turns]

    print(f"[인코더 로딩] {args.encoder} ...", file=sys.stderr)
    enc = make_encoder(model=args.encoder)
    embs = np.asarray(enc.encode([t["text"] for t in turns]))

    # Stage 1 — 우리 신호로 후보 생성 (0-lag, LLM 0콜)
    cands, deff = gen_peak(embs, k=args.peak_k)
    print(f"[Stage1] δ_eff 후보 {len(cands)}개 / {len(turns)}턴 "
          f"(rate {len(cands)/len(turns):.2f}) — 후보 turn: {cands}", file=sys.stderr)

    # Stage 2 — 후보에서만 LLM NEW/SAME 판정
    boundaries = [False] * len(turns)
    verdicts: dict[int, str] = {}
    if cands:
        print(f"[Stage2] LLM judge ({args.model}) — 후보 {len(cands)}개 호출 ...", file=sys.stderr)
        cli = make_client()
        for t in cands:
            v, raw = judge(cli, args.model, build_prompt(lines, t, args.ctx_k))
            verdicts[t] = raw or ("NEW" if v else "SAME")
            if v == 1:
                boundaries[t] = True

    extra = []
    for i in range(len(turns)):
        tag = f"δ={deff[i]:.3f}"
        if i in verdicts:
            tag += f"  [LLM:{verdicts[i]}]"
        extra.append(tag)

    n_new = sum(boundaries)
    print(f"\n[방법] two-stage spike-gated LLM judge · 후보 {len(cands)} → NEW {n_new}",
          file=sys.stderr)
    print_segments(turns, boundaries, extra)
    print(f"우리 신호 후보 {len(cands)}개 중 LLM 이 {n_new}개를 경계로 확정 "
          f"(LLM 호출 = 후보 수만큼, 전체 turn 아님). δ=δ_eff · ▶=화제 경계")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
