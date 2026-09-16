#!/usr/bin/env python3
"""④ cross-model 비교 — 프롬프트 고정, 모델을 나란히 (어느 라벨러가 최고 Score?).

deepseek/GLM/Qwen32B/Qwen3-Next × {v2, v3}. deployable Score(5축 t*=3). API 0(reset_matrix 재사용).
v3 표는 전부 no-think(deepseek=v3-OFF)라 공정. 산출: outputs/experiments/2026-07-23_s5_redesign/CROSSMODEL.md
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402

FROZEN = ["protagonist", "space", "time", "causality", "intentionality"]
DOCS = ["pieman", "tunnel", "monkey"]
OUT = ROOT / "outputs/experiments/2026-07-23_s5_redesign"


def sc(suf: str, doc: str, t=3):
    f = ROOT / f"outputs/eacl_phase0/{doc}_michelmann_s5_{suf}/reset_matrix.jsonl"
    if not f.exists():
        return None
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if str(r["doc_id"]) == doc]
    if not rows:
        return None
    rows.sort(key=lambda r: int(r["transition"]))
    n = len(rows) + 1
    gold = [0] * n
    for r in rows:
        if int(r["boundary"]):
            gold[int(r["transition"]) - 1] = 1
    cnt = {int(r["transition"]): sum(int(r[a]) for a in FROZEN) for r in rows}
    bt = [i for i in cnt if cnt[i] >= t]
    pred = [0] * n
    for i in bt:
        pred[i - 1] = 1
    return score_dialogues([gold], [pred])["score"]


def table(title, rows):
    L = [f"## {title}", "", "| 모델 | pieman | tunnel | monkey | 평균 |", "|---|---|---|---|---|"]
    data = []
    for name, suf in rows:
        ss = [sc(suf, d) for d in DOCS]
        if any(s is None for s in ss):
            L.append(f"| {name} | _pending_ | | | |")
            continue
        data.append((name, ss, sum(ss) / 3))
    best = max((a for _, _, a in data), default=None)
    for name, ss, a in data:
        star = " ★" if a == best else ""
        L.append(f"| {name} | {ss[0]:.3f} | {ss[1]:.3f} | {ss[2]:.3f} | **{a:.3f}**{star} |")
    L.append("")
    return L


RUNS = ("", "_r2", "_r3")   # D-B47: k=3. 배치 비결정을 노이즈로 흡수한다.


def sc_runs(base: str, doc: str) -> list[float]:
    """base 접미사의 r1/r2/r3 중 존재하는 run 들의 Score."""
    return [s for s in (sc(base + r, doc) for r in RUNS) if s is not None]


def table_k(title: str, rows: list[tuple[str, str]]) -> tuple[list[str], dict]:
    """k-run 평균±SD 표. rows = [(표시명, base_suffix)]. 판정용 집계도 함께 반환."""
    L = [f"## {title}", "", "| 모델 | k | pieman | tunnel | monkey | 평균 |", "|---|---|---|---|---|---|"]
    agg: dict[str, dict] = {}
    for name, base in rows:
        per_doc = {d: sc_runs(base, d) for d in DOCS}
        ks = {len(v) for v in per_doc.values()}
        if not all(per_doc.values()):
            L.append(f"| {name} | — | _pending_ | | | |")
            continue
        means = {d: mean(v) for d, v in per_doc.items()}
        sds = {d: (stdev(v) if len(v) > 1 else 0.0) for d, v in per_doc.items()}
        avg = mean(means.values())
        cells = " | ".join(f"{means[d]:.3f} ±{sds[d]:.3f}" for d in DOCS)
        L.append(f"| {name} | {min(ks)} | {cells} | **{avg:.3f}** |")
        agg[name] = {"per_doc": means, "sd": sds, "avg": avg, "k": min(ks)}
    L.append("")
    return L, agg


BASELINE = "Qwen3.6-35B-A3B-v3.1 (기준선)"
CAND_GPTOSS = "gpt-oss-120b-v3.1"
CAND_DSAWQ = "DeepSeek-V3.1-AWQ(4bit)-v3.1"
CAND_GEMMA = "Gemma-4-26B-A4B-v3.1"
CAND_GEMMA31 = "Gemma-4-31B dense-v3.1"
CAND_MM_ON = "MiniMax-M2.7-v3.1 (추론 ON)"
CAND_MM_OFF = "MiniMax-M2.7-v3.1 (추론 OFF)"


def verdict(agg: dict, cand: str, tag: str, note: str) -> list[str]:
    """사전등록 기준(+0.03 AND 2/3 무하락)으로 판정. 기준은 결과를 보고 바꾸지 않는다."""
    base = BASELINE
    if base not in agg or cand not in agg:
        return [f"> _{tag} 판정 보류 — {cand} 아직 미완료._", ""]
    d = agg[cand]["avg"] - agg[base]["avg"]
    kept = [x for x in DOCS if agg[cand]["per_doc"][x] >= agg[base]["per_doc"][x]]
    ok_margin, ok_docs = d >= 0.03, len(kept) >= 2
    L = [f"## 판정 — {cand} ({tag} 사전등록 기준)", "",
         f"- ΔScore(평균) = **{d:+.3f}** — 기준 **+0.03 이상** → {'충족' if ok_margin else '**미달**'}",
         f"- 하락 없는 문서 = **{len(kept)}/3** ({', '.join(kept) or '없음'}) — 기준 **2편 이상** "
         f"→ {'충족' if ok_docs else '**미달**'}", ""]
    if ok_margin and ok_docs:
        L += [f"**→ 채택 후보.** 앵커 재생성 여부는 별도 결정({tag} §4).", ""]
    else:
        L += ["**→ 노이즈 내 동등 또는 열위. 동결 유지, 라벨러 교체 없음.**",
              f"Qwen3.6-35B-A3B / v3.1 그대로 간다. {cand} 수치는 기록으로만 남는다.", ""]
    L += [f"> {note}", ""]
    return L


def build():
    L = ["# ④ cross-model Score 비교 (프롬프트 고정, dev-3, deployable t\\*=3)", "",
         "> 각 1 run(배치 비결정 ±0.01~0.05 노이즈). v3=전부 no-think(공정). deepseek v2=reasoning(교란).", ""]
    L += table("v2 프롬프트 (⚠️ deepseek=reasoning, 오픈=no-think)",
               [("deepseek-v2 (reason)", "eismv2"), ("GLM-v2", "glm_v2"),
                ("Qwen32B-v2", "qwen32b_v2"), ("Qwen3-Next-v2", "qwennext_v2"), ("deepseek-chat-v2", "dschat_v2")])
    L += table("v3 프롬프트 — 전부 no-think (공정 비교)",
               [("deepseek-v3-OFF", "eismv3_noreason"), ("GLM-v3", "glm_v3"),
                ("Qwen32B-v3", "qwen32b_v3"), ("Qwen3-Next-v3", "qwennext_v3"), ("deepseek-chat-v3", "dschat_v3")])
    L += ["> 참고 deepseek-v3-ON(reason) 평균 = 0.327 (v3에서 reasoning은 해로움).", ""]
    return "\n".join(L) + "\n"


def build_v31():
    """D-B47·D-B49 — 동결 프롬프트 v3.1 에서의 라벨러 후보 비교 (k=3)."""
    L = ["# D-B47·D-B49 — S5 라벨러 v3.1 dev-3 비교 (후보: gpt-oss-120b · DeepSeek-V3.2-AWQ)", "",
         "## 실험 setup", "",
         "- **목적:** 로컬 호스팅 가능한 더 나은 S5 라벨러가 있는지. DeepSeek 계열은 671B MoE·FP8 ~670GB 라",
         "  A100-80GB×4(320GB)에 안 올라가 탈락 → `openai/gpt-oss-120b`(117B MoE, MXFP4 65.2GB)가 후보.",
         "- **데이터:** dev-3 = pieman / tunnel / monkey (`outputs/eacl_phase0/{doc}_michelmann_master.jsonl`).",
         "  **held-out(Tier1 6편) 미개봉.** gold = `reset_matrix.jsonl` 의 `boundary`(Michelmann, author-filtered).",
         "- **방법:** `llm_s5_states.py --prompt-version v3.1 --engine vllm`, temp 0, batch 20, seed 20260711,",
         "  vLLM 0.25.1 (VESSL A100×1 job, 문서 1편 = job 1개, 5축 동시). 5축 전부",
         "  (protagonist·space·time·causality·intentionality), 축 병렬 후 full 패스로 reset_matrix 조립.",
         "- **⚠️ 추론 설정이 두 모델에서 대칭이 아니다.** Qwen = `--no-reasoning`",
         "  (`chat_template_kwargs={'enable_thinking':False}`) 로 **완전 off**. gpt-oss 는 **off 스위치가 없어**",
         "  `reasoning_effort='low'` + `--reasoning-parser openai_gptoss` 로 최저값만 줬다 — **추론이 켜져 있다.**",
         "  ledger 근거상 이 과제에서 추론은 해롭다(v3: deepseek reason-ON 0.327 vs OFF 0.364).",
         "  ⇒ **gpt-oss 에 불리한 교란**이며, 아래 열세를 '모델이 약함' 으로 읽으면 안 된다.",
         "- **가중치:** `openai/gpt-oss-120b` MXFP4 65.2GB. A100(sm80)에서 `Mxfp4Backend=MARLIN` 으로",
         "  **네이티브 적재**(bf16 디양자화 아님) — A100 1장에 올라간다. 적재 2분 35초(runai_streamer).",
         "- **후보 2 (D-B49): `QuantTrio/DeepSeek-V3.2-AWQ` — 4-bit AWQ 361.9GB, a100x8(TP=8).**",
         "  네이티브 FP8 원본은 689.5GB 라 640GB 에 안 들어가 **양자화본만 가능**하다.",
         "  🔴 **원본이 아니다** — 열세가 나와도 *모델 능력* 인지 *양자화 손실* 인지 분리 안 됨.",
         "  🟢 추론은 `chat_template_kwargs={'thinking':false}` 로 **완전 off** — Qwen 과 대칭이고",
         "  기존 `dschat_v3`(프록시 no-think) 조건과도 일치한다. gpt-oss 의 비대칭 문제는 여기 없다.",
         "  ⚠️ k=3 구조가 다르다: gpt-oss/Qwen 은 run 마다 새 서버, DeepSeek 은 **같은 인스턴스에서 3회**",
         "  (362GB 를 한 번만 적재). 배치 비결정은 동일하게 잡히나 적재-시점 변동은 안 잡힌다.",
         "- **지표:** deployable Score(t\\*=3) = 0.5·F1 + 0.25·(1−Pk) + 0.25·(1−WD), `dts_scoring.score_dialogues`.",
         "  gold 경계 개수를 방법에 주지 않는다(matched@gold 아님).",
         "- **반복:** 각 모델 k=3 run(접미사 `_v31`, `_v31_r2`, `_v31_r3`). 배치 비결정이 노이즈원.", ""]
    rows, agg = table_k("결과 — v3.1 (동결 프롬프트), 평균 ±SD over k run", [
        (BASELINE, "q36_35b_v31"),
        (CAND_GEMMA, "gemma4a4b_v31"),
        (CAND_GEMMA31, "gemma431b_v31"),
        (CAND_GPTOSS, "gptoss120b_v31"),
        (CAND_MM_OFF, "mm27off_v31"),
        (CAND_MM_ON, "mm27on_v31"),
        (CAND_DSAWQ, "ds31awq_v31"),
    ])
    L += rows
    L += verdict(agg, CAND_GEMMA, "D-B51 §2",
                 "🟢 **교란 없음** — 원본 가중치, 템플릿 무개조, `enable_thinking:False` 가 Qwen 과 동일 키. "
                 "이 판정만은 주석 없이 인용 가능하다.")
    L += verdict(agg, CAND_GPTOSS, "D-B47 §2",
                 "교란 = **추론 off 불가**(reasoning_effort=low 가 최저). gpt-oss 에 불리 → 이 값은 하한.")
    L += verdict(agg, CAND_MM_OFF, "D-B50 §3",
                 "교란 = **템플릿 개조**(원본은 `<think>` 강제). 채택 판정은 OFF 만 쓴다. "
                 "ON−OFF 차이는 *이 과제에서 추론이 Score 를 얼마나 깎는가* 의 직접 측정값이다.")
    L += verdict(agg, CAND_DSAWQ, "D-B49 §1",
                 "교란 = **4-bit AWQ 양자화**(원본 아님). 추론은 `thinking:false` 로 대칭. "
                 "⚠️ V3.2 는 DSA 가 Hopper/Blackwell 전용이라 A100 불가 → **V3.1 로 교체**(D-B49 §6).")
    if agg.get(CAND_MM_ON) and agg.get(CAND_MM_OFF):
        d = agg[CAND_MM_ON]["avg"] - agg[CAND_MM_OFF]["avg"]
        L += ["## 부수 측정 — 추론이 Score 에 미치는 영향 (MiniMax ON vs OFF, 동일 모델)", "",
              f"- ΔScore(ON − OFF) = **{d:+.3f}**",
              "- 같은 가중치·같은 프롬프트에서 추론 스위치만 바꾼 값이므로 **교란 없는 추론 효과 추정치**다.",
              "- 이 값이 음수면 D-B48 §4 의 gpt-oss 열세(−0.003)가 **모델 능력이 아니라 강제 추론 탓**일",
              "  가능성이 정량적으로 뒷받침된다. 양수면 그 방어 논리는 약해진다.", ""]
    if agg.get("gpt-oss-120b-v3.1") and agg.get("Qwen3.6-35B-A3B-v3.1 (기준선)"):
        g, q = agg["gpt-oss-120b-v3.1"], agg["Qwen3.6-35B-A3B-v3.1 (기준선)"]
        L += ["## 해석", "",
              "**평균은 동률인데 문서별로는 정반대다.** 문서별 Δ(gpt-oss − Qwen) = " +
              ", ".join(f"{d} {g['per_doc'][d]-q['per_doc'][d]:+.3f}" for d in DOCS) + ".",
              "pieman 에서 gpt-oss 가 크게 앞서고 tunnel 에서 크게 뒤진다 — 평균이 상쇄해 0 이 된 것이지",
              "'두 모델이 같은 라벨을 만든다'는 뜻이 아니다. 교차모델 표에서 이미 관측된",
              "**문서별 지배 도메인 상이** 패턴(D-B14 era: tunnel=space·monkey=protag/intent)과 같은 종류다.",
              "pieman 의 gpt-oss SD(±0.056)가 다른 셀의 2~3배라 이 우위 자체도 안정적이지 않다.", "",
              "**Qwen v3.1 기준선(0.382)은 기존 v3 값(0.387)과 사실상 같다** — 프롬프트 v3→v3.1 전환이",
              "dev-3 Score 를 흔들지 않았다는 뜻이고, 이번에 처음 만든 v3.1 dev-3 기준선의 sanity check 가 된다.", ""]
    L += ["## 한계 / 검증 미해결", "",
          "- 🔴 **추론 비대칭(위 setup 참조)이 최대 한계다.** gpt-oss 는 추론을 끄지 못한 채 겨뤘으므로",
          "  이 결과는 **gpt-oss 의 하한**이다. '열위' 가 아니라 '이 조건에서 우위 없음' 까지만 말할 수 있다.",
          "- dev-3 는 D-M5 로 소진된 연습장이다. 여기 수치는 **탐색적**이며 held-out 성능이 아니다.",
          "- k=3 은 배치 비결정만 흡수한다. 문서 수 3편이라 doc-level bootstrap CI 는 불가.",
          "- gpt-oss-120b 는 v3.1 프롬프트에 맞춰 튜닝된 적이 없다 — deepseek 기각(D-M13⑥)이",
          "  '능력이 아니라 스펙 해석 차이'였던 것과 같은 교란이 남아 있을 수 있다.",
          "- 🔴 **재실행 시 주의:** gpt-oss 는 `--quantization` 을 주지 않는다(체크포인트 config 가",
          "  `gpt_oss_mxfp4` 를 선언한다). Qwen 처럼 `--quantization fp8` 을 붙이면 안 된다.", ""]
    return "\n".join(L) + "\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rep = build()
    (OUT / "CROSSMODEL.md").write_text(rep, encoding="utf-8")
    print(rep)

    exp = ROOT / "outputs/experiments/2026-07-31_s5_labeler_v31"
    exp.mkdir(parents=True, exist_ok=True)
    rep31 = build_v31()
    (exp / "REPORT.md").write_text(rep31, encoding="utf-8")
    print(rep31)


if __name__ == "__main__":
    main()
