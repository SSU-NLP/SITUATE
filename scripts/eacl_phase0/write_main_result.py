#!/usr/bin/env python3
"""`EACL/main_result.md` 생성 — 6지표 전면 결과표 (dev-3 · DTS-3 · 시험지).

🔒 **수치를 손으로 옮기지 않는다.** 전부 `preds_table.json`(= `preds/*.json` 원자료에서
`preds_table.py` 가 재계산한 것)과 게이트/검증 산출 JSON 에서 읽어 렌더링한다.

    .venv/bin/python scripts/eacl_phase0/preds_table.py --datasets dev3,dts_full_tiage,\
dts_full_dialseg711,dts_full_superseg,test3,test3_goldauthor
    .venv/bin/python scripts/eacl_phase0/write_main_result.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "outputs/experiments/2026-07-29_baseline_compare"
OUT = REPO / "EACL/main_result.md"

# 표시 이름 · 유형 · 온라인성. 여기 없는 행은 표에서 제외된다(M1·M4·고정분위수 M5 = 사용자 지시).
SPEC: dict[str, tuple[str, str, str]] = {
    "M5-adaptive": ("**M5-FactoredSlot**", "우리 · 학습(서사 앵커) · LLM 호출 0",
                    "온라인 — 적응 임계(과거 통계만)"),
    "B3event-online-r10_gpt-4.1": ("LLM 직접질의 / GPT-4.1", "외부 · LLM 프롬프트",
                                   "온라인 — full-past, 미래 미사용"),
    "B3event-online-r10_gpt-5": ("LLM 직접질의 / GPT-5", "외부 · LLM 프롬프트",
                                 "온라인 — full-past, 미래 미사용"),
    "B3event-online-r10_gpt-oss-120b": ("LLM 직접질의 / gpt-oss-120b", "외부 · LLM 프롬프트",
                                        "온라인 — full-past, 미래 미사용"),
    "SeCom-binary-stateful_gpt-4o": ("SeCom-binary 상태전달 / GPT-4o", "외부 · LLM 프롬프트",
                                     "온라인 — full-past, 미래 미사용"),
    "SeCom-binary_gpt-4o": ("SeCom-binary 무상태 / GPT-4o", "외부 · LLM 프롬프트",
                            "온라인 — full-past, 미래 미사용"),
    "Membox-TopicLoom": ("Membox-TopicLoom", "외부 · LLM 프롬프트(우리 이식판)",
                         "온라인 — 직전 2문장 창"),
    "TextTiling": ("TextTiling", "외부 · 무감독", "온라인 — 우리 스트리밍 개조(원논문 오프라인)"),
    "GraphSeg": ("GraphSeg", "외부 · 무감독", "온라인 — 우리 스트리밍 개조(원논문 오프라인)"),
    "GreedySeg": ("GreedySeg", "외부 · 무감독", "온라인 — 우리 스트리밍 개조(원논문 오프라인)"),
    "CSM-offverified": ("**CSM (오프라인 검증 ckpt)**", "외부 · 무감독(학습형)",
                        "온라인 — 우리 스트리밍 개조(원논문 오프라인)"),
    "CSM": ("CSM (최종 ckpt)", "외부 · 무감독(학습형)",
            "온라인 — 우리 스트리밍 개조(원논문 오프라인)"),
    "CSM-devbest": ("CSM (검증셋 최고 ckpt)", "외부 · 무감독(학습형)",
                    "온라인 — 우리 스트리밍 개조(원논문 오프라인)"),
    "RoBERTa": ("RoBERTa", "외부 · **지도학습**(SuperDialseg gold)", "온라인 — strict-causal 추론"),
}
STAGES = [("dev3", "dev 3편 (서사 · 연습장)"),
          ("test3", "🔒 시험지 3편 (서사 · *Born a Crime*)"),
          ("dts_full_tiage", "TIAGE (대화)"),
          ("dts_full_dialseg711", "DialSeg711 (대화)"),
          ("dts_full_superseg", "SuperDialseg (대화)")]
APPENDIX = [("test3_goldauthor", "부록 — 시험지, 저자 규칙 gold (게이트 탈락판)")]


def table(ds: str, blk: dict, title: str) -> list[str]:
    """한 무대의 6지표 표 (Score 내림차순, 하한 두 개를 제자리에 끼워 넣는다)."""
    rows = [r for r in blk["rows"] if r["method"] in SPEC]
    items = []
    for r in rows:
        l1 = "—" if r["L1"] != r["L1"] else f"{r['L1']:.2f}"     # 경계 0개면 lift 가 정의되지 않는다
        deg = "  🔴 경계 0개" if r["n_pred"] == 0 else ""
        items.append((r["score"], SPEC[r["method"]][0] + deg,
                      f"{r['score']:.4f} | {r['bor']:.2f} | {r['purity']:.3f} | "
                      f"{r['coverage']:.3f} | {r['f_pc']:.3f} | {l1}"))
    for name, s in blk["floors"].items():
        items.append((s, f"*{name}*", f"*{s:.4f}* | — | — | — | — | —"))
    items.sort(key=lambda x: -x[0])
    L = [f"**{title}** · 정답 경계 {blk['n_gold']}개",
         "", "| 방법 | Score | BOR | purity | coverage | F(p,c) | L1 |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    L += [f"| {n} | {c} |" for _, n, c in items]
    L.append("")
    return L


def main() -> None:
    tab = json.loads((EXP / "preds_table.json").read_text())
    gate = json.loads((REPO / "outputs/eacl_phase0/eventrecall_gold_gate.json").read_text())
    csm = json.loads((REPO / "outputs/experiments/2026-08-02_csm_offline_verify/summary.json")
                     .read_text())
    body: list[str] = []
    for ds, title in STAGES:
        body += table(ds, tab[ds], title) if ds in tab else [f"**{title}** — 🔴 예측 없음.", ""]
    apx: list[str] = []
    for ds, title in APPENDIX:
        apx += table(ds, tab[ds], title) if ds in tab else [f"**{title}** — 🔴 예측 없음.", ""]

    onl = ["| 방법 | 유형 | 온라인성 |", "|---|---|---|"]
    seen = set()
    for ds, _ in STAGES:
        for r in tab.get(ds, {}).get("rows", []):
            m = r["method"]
            if m in SPEC and m not in seen:
                seen.add(m)
                n, t, o = SPEC[m]
                onl.append(f"| {n} | {t} | {o} |")

    g = {d["doc_id"]: d for d in gate["docs"]}
    gl = ["| 문서 | 문장 | 전이 | 주석자 | 1인 평균 경계 | 저자 규칙(≥1명) | 배수 | 채택 gold(top-N) |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for d in gate["docs"]:
        gl.append(f"| {d['doc_id']} | {d['n_sent']} | {d['n_trans']} | {d['n_subjects']} | "
                  f"{d['mean_subj']:.1f} | {len(d['gold_author'])} | "
                  f"{d['rate_author'] / d['subj_rate']:.2f}× | {d['n_top']} |")

    paper = {"tiage": 0.509, "dialseg711": 0.660, "superseg": 0.458}
    cl = ["| ckpt (step) | tiage | dialseg711 | superseg | 평균 | vs 논문 (%) |",
          "|---|---:|---:|---:|---:|---|"]
    for k, v in sorted(csm.items(), key=lambda x: int(x[0])):
        mean = sum(v.values()) / 3
        mark = " ← **선택**" if k == "175335" else ""
        cl.append(f"| {k}{mark} | {v['tiage']:.4f} | {v['dialseg711']:.4f} | {v['superseg']:.4f} | "
                  f"{mean:.4f} | " + " / ".join(f"{v[d] / paper[d] * 100:.1f}" for d in paper) + " |")

    fam = json.loads((EXP / "m5_family_test3.json").read_text())["rows"]
    fl = []
    for mode in ("adaptive", "quantile"):
        xs = sorted((r[mode]["score"] for r in fam), reverse=True)
        d0 = [r for r in fam if r["ckpt"].startswith("m5_default")][0][mode]["score"]
        med = xs[len(xs) // 2]
        fl.append(f"| {mode} | {xs[-1]:.4f} | {med:.4f} | **{xs[0]:.4f}** | {d0:.4f} "
                  f"| **{xs.index(d0) + 1}위 / {len(xs)}** |")
    fam_tbl = (["| 작동점 | 최소 | 중앙값 | 최대 | 동결본 | 동결본 순위 |",
                "|---|---:|---:|---:|---:|---:|"] + fl)

    doc = (HEADER + "\n".join(body) + MIDDLE + "\n".join(onl) + GOLD + "\n".join(gl)
           + HELDOUT + "\n".join(apx) + CSMSEC + "\n".join(cl) + CSMNOTES
           + FAMILY + "\n".join(fam_tbl) + TAIL)
    OUT.write_text(doc, encoding="utf-8")
    print(f"-> {OUT}")


HEADER = """# EACL main result — 6지표 전면 결과표

> 🔒 **이 문서는 생성물이다.** `scripts/eacl_phase0/write_main_result.py` 가
> `outputs/experiments/2026-07-29_baseline_compare/preds_table.json`(= 전이별 0/1 예측 원자료에서
> `preds_table.py` 가 재계산) 에서 렌더링한다. 수치를 손으로 고치지 말 것.
> 결정 기록 = `phase0-decisions-and-runs.md` **D-B62** · 요약본 = `pipeline-overview-ko.md` §2.4.1.

## 0. 결론 먼저

1. 🔴 **시험지에서 우리 방법은 1위가 아니다.** dev 3편에서는 M5-FactoredSlot 이 외부 방법을
   전부 앞섰지만(0.4049, 최고 외부 0.3212), **봉인 시험지에서는 3위**다 —
   LLM 직접질의/GPT-4.1(0.4010) · TextTiling(0.3994) · **M5(0.3888)**. 무경계 하한(0.3092)은
   +0.080 으로 넘지만 **"외부 방법을 이긴다"는 dev 관찰은 시험지에서 재현되지 않았다.**
2. **대화(DTS)로 옮기면 크게 뒤진다** — 지도학습 RoBERTa·Membox 아래이고, 세 데이터셋 모두에서
   M5 의 L1 은 1.1~2.2 로 신호는 있으나 밀도 규칙이 도메인에 맞지 않는다.
3. **CSM 이 하한 근처인 것은 학습 실패가 아니라 온라인화 손실이다** — 같은 체크포인트가
   오프라인에서는 논문 수준(tiage 105% · superseg 102%)을 낸다(§5).
4. **Score 하나로는 안 보인다.** 시험지에서 M5 의 F(p,c)=0.774 는 표 최고이고 BOR 1.67 로
   밀도도 가장 정직한데, 상위 두 방법은 BOR 0.42/1.30 이다. 반대로 GPT-5·gpt-oss-120b 는
   **경계를 0개** 찍어 무경계 하한을 그대로 받았다(성능이 아니라 퇴화 구성).
5. 🔴 **그 순위 자체가 노이즈 안이다.** M5 체크포인트 132개를 같은 시험지에서 돌리면
   0.3092~0.4505 로 흩어진다(§6). **변형 선택의 폭이 방법 간 격차보다 한 자릿수 크므로**
   1~3위 순서로는 어느 방향의 결론도 세울 수 없다 — 우리에게 유리한 쪽으로도, 불리한 쪽으로도.

## 1. 결과표

전 행이 **온라인**이고(§3), 채점기는 **1종**(하네스 A = `hi_ontop.dts_scoring`), 평가 모드는
**deployable**(어떤 방법에도 정답 경계 개수를 주지 않는다)이다.

"""

MIDDLE = """
## 2. 지표 읽는 법

| 지표 | 뜻 | 읽는 법 |
|---|---|---|
| **Score** | `0.5·F1 + 0.25·(1−Pk) + 0.25·(1−WindowDiff)` | **주지표.** 자명한 하한과 비교해야 의미가 있다 |
| BOR | 예측 경계 수 ÷ 정답 경계 수 | 1.0 = 밀도 일치 · >1 과분절 · <1 과소분절 |
| purity | 예측 구간이 정답 구간 하나에 얼마나 순수하게 담기나 | 🔴 **잘게 쪼갤수록 자동으로 올라간다 — 단독 인용 금지** |
| coverage | 정답 구간이 예측 구간 하나에 얼마나 덮이나 | 과분절에 벌점 |
| F(p,c) | purity·coverage 의 조화평균 | purity 단독의 착시를 즉시 드러낸다 |
| **L1** | `P(경계\\|예측) ÷ P(경계)` = 표준 lift | **1.0 미만이면 역신호** — 찍은 자리가 오히려 정답일 확률이 낮다 |

## 3. 온라인성 — 전부 온라인이다, 단 4종은 우리가 개조한 것

"""

GOLD = """

- **오프라인은 표에 없다.** SeCom 원본 partition(입력 전체를 한 번에 봄)·오프라인 B3 topline·
  오프라인 CSM 은 **경쟁 대상이 아니므로 제외**했다. 유일한 예외적 사용처가 §4 의 검증이다.
- 🔴 **TextTiling·GraphSeg·GreedySeg·CSM 4종은 원논문이 오프라인**이다(전 문서 depth + 전역 임계).
  온라인 비교를 하려면 스트리밍화가 불가피하지만 **그 결과 이 점수는 원논문 성능이 아니다.**
  스트리밍 노브(윈도 D · delay 2 · warmup · min_gap)는 우리가 정한 것이며, 나머지 상수는 발표값이다.
- **LLM 직접질의 프롬프트는 event 문구로 통일**했다(`baseline_segment_binary_event*_v1.md`).
  대화 데이터에도 *"large events"* 를 묻는다 — M5 역시 서사 사건 분절기인 채로 대화 화제 gold 에
  채점받으므로, LLM 에만 *topic* 힌트를 주면 비대칭이기 때문이다.
  🔴 **대신 이 선택은 DTS 에서 LLM 에 불리한 방향이다.** 대화 도메인 event 프롬프트는
  우리가 만든 변형이며(SeCom 원문에서 기준 문구만 교체) 원 논문에 없다.
  ⚠️ **예외: SeCom 행은 원문(topic) 그대로다** — SeCom 의 발표 구성개념 자체가 topic 이라
  event 로 바꾸면 그건 SeCom 이 아니다.
- 🔴 **event 통일의 대가가 실측됐다.** TIAGE 에서 gpt-oss-120b 는 **경계를 하나도 찍지 않았다**
  (1,464 전이 전부 SAME). 대화에서 *"하나의 사건이 끝나고 다른 사건이 시작되는가"* 를 물으면
  모델이 "한 통화 = 한 사건"으로 답하기 때문이다. **이 칸은 모델 성능이 아니라 구성개념 불일치의
  기록으로 읽어야 한다.** 같은 무대에서 topic 문구를 쓰면 다른 값이 나온다(D-B56 의 SeCom 행 참조).

## 4. 시험지(held-out) — 개봉 기록

🔴 **봉인 시험지를 이번에 열었다.** 원장 규약은 "개봉 1회"였고 진입 조건은
*1만편 k 재설정 → 앵커 재생성 → 재학습 후 확인적 평가*였다. **그 조건 전에 열었다**(사용자 지시).
따라서 이 표는 **확인적(confirmatory) 결과가 아니라 현재 체크포인트의 1회 측정**이다.

**gold 규칙 = 채점 전에 못박았다.** 원 저장소(EventRecall)에는 주석자 20명의 word_number 누름만
있고 합의 gold 추출 코드가 없다. 저자 코드의 "peak" 정의는 `avg_responses != 0`
= **20명 중 1명이라도 누른 지점**이라 그대로 쓰면 밀도가 사람 1인의 4~6배가 된다.
**게이트(사전 선언: 1인 평균 밀도의 2배 초과 시 사용 불가) → FAIL** ⇒ 사전 확정된 대체 규칙인
**밀도 정합 top-N**(N = 주석자 1인 평균 경계 수)을 main gold 로 쓴다. 판정은 **어떤 방법의 점수도
보기 전에** 내렸다.

"""

HELDOUT = """

- ⚠️ **시험지 유효 표본 n ≈ 1** — 3편 전부 같은 작품·저자·화자·주석자 풀(*Born a Crime*)이다.
  문서 단위 bootstrap 을 독립 3개로 취급할 수 없다. 격차 주장을 여기서 하지 않는다.
- 🔴 **gold 규칙이 순위를 바꾼다.** 아래 부록(저자 규칙 gold, 경계 217개)에서는 순위가 거의
  뒤집힌다 — 밀도가 5배로 오르면 과분절 방법이 유리해지기 때문이다. **main 은 게이트를 통과한
  밀도 정합 gold 하나로만 읽는다.** 두 판을 섞어 인용하지 말 것.

"""

CSMSEC = """
## 5. CSM — 오프라인에서는 논문을 재현한다

기존 표에서 CSM 이 자명한 하한 근처였던 것이 **학습 실패로 오해될 수 있었다.** 우리 학습본을
**원논문 오프라인 추론**으로 다시 돌려 확인했다(러너 `methods/CSM/offline/whole_dialogue.py`,
하이퍼 전부 발표값). 선택 규칙 = **3셋 평균 오프라인 Score 최대**, 선택셋 = **CSM 자기 논문 벤치의
공식 test**(우리 dev-3·시험지 아님 — baseline 에 유리한 방향).

"""

FAMILY = """

## 6. 시험지 결과의 상한 — M5 계열 132개를 전부 돌려보면

우리 M5 체크포인트 **132개 전부**를 같은 시험지에서 채점했다
(`outputs/experiments/2026-08-02_m5_family_test3/REPORT.md`).
🔒 **실행 전에 선언했다: 어느 변형이 높든 위 표의 우리 행은 동결본으로 고정한다.**

"""

CSMNOTES = """

논문 오프라인(하네스 C 인용): tiage **0.509** · dialseg711 **0.660** · superseg **0.458**.

- ✅ tiage 84~105% · superseg 100~112% ⇒ **재현된다.**
- 🔴 dialseg711 은 최대 76.6% 로 재현되지 않는다. 5월의 **배포 체크포인트**도 같은 자리에서 72%
  였으므로(D-B41 §5) 체크포인트가 아니라 **데이터·전처리 계보 차이**로 보인다. 미해명.
- ⇒ **CSM 이 우리 표에서 낮은 것은 온라인화 손실이다.** `CSM (오프라인 검증 ckpt)` 행이 그
  손실의 크기를 보여준다. 상세 = `outputs/experiments/2026-08-02_csm_offline_verify/REPORT.md`.
- 🔴 하네스 주의: 위 % 는 **하네스 B(우리 오프라인 러너) vs C(논문 인용)** 의 비율이다.
  main 표는 이것과 무관하게 **하네스 A** 단일로 채점했다. 세 수치를 나란히 두지 말 것.
"""

TAIL = """

- 🔴 **변형 선택의 폭(0.141)이 방법 간 격차(0.01~0.02)보다 한 자릿수 크다.**
  ⇒ 시험지에서 *"M5 가 GPT-4.1·TextTiling 보다 낮다"* 는 순위는 **변형 선택 노이즈 안에 있다.**
  우리에게 유리한 쪽으로도, 불리한 쪽으로도 이 순위로 결론을 세울 수 없다.
- 🔴 **동결본은 계열 중앙값보다 낮다**(85위/132). 상위 변형을 골랐다면 0.4505 로 표 1위가 됐다.
  **그래서 고르지 않는다** — 그 숫자는 "이길 수도 있었다"가 아니라 **selection-on-test 로 부풀릴 수
  있는 크기**의 기록이다.
- **현행 계보(phase1_i2, 중앙값 0.3797)가 구세대(phase0, 0.4033)보다 시험지에서 낮다.**
  dev 에서의 선택이 시험지로 전이되지 않았다. 미해명이며 여기서 계보를 바꾸지 않는다.
- 전체 132행 = `outputs/experiments/2026-08-02_m5_family_test3/REPORT.md`.

## 7. 표에서 뺀 것 · 안 돌린 것 (조용한 절단 금지)

| 항목 | 사유 |
|---|---|
| **M1-CosDrift** | 논문 베이스라인에서 제외(D-M12, 사용자 결정). 참고용 sanity check 였다 |
| **M4-MonoAnchor** | 사용자 지시로 이번 표에서 제외. M5−M4 분리 대조는 별도 절에서 다룬다 |
| **M5 고정분위수판** | 폐기된 작동점(`r=0.0642`, D-B61 §1). 현행 진입점은 적응 임계뿐이다 |
| **GPT-4.1 · GPT-5 의 DTS 칸** | 미실행 — reps=10 × 36,109 전이 = 모델당 199M 입력 토큰(≈ $250~400). 비용 승인 대기 |
| **SeCom 의 DTS 칸** | 미실행 — 같은 사유(gpt-4o ≈ $500) |
| **CSM 240 체크포인트 전수** | 미실행 — 에폭 말 11개만 오프라인 채점 |
| **오프라인 방법 전반** | 경쟁 대상 아님(§3) |

## 8. 한계 (제거 금지)

- **시험지 1회 개봉을 탐색적 체크포인트에 소진했다.** 확인적 평가로 쓸 판이 더는 없다.
- **시험지 유효 n ≈ 1** (같은 작품·저자·주석자 풀).
- **dev-3 는 문서 3편.** Score 는 문서 단위 평균이라 실질 표본 n=3.
- **M5 는 seed 0 단일.** 격차 주장은 3-seed 값으로만 한다.
- **스트리밍화 손실을 방법별로 분리하지 않았다.** CSM 한 종에서만 측정했다(§5).
- **Membox-TopicLoom 은 우리 이식판**이며 원 논문에 이 무대의 수치가 아예 없다(D-B59).
- **시험지 gold 는 우리가 정한 규칙**이다(저자 규칙이 게이트에서 탈락). 부록 병기판
  `test3_goldauthor` 와 순위가 다를 수 있다.
- **LLM 직접질의는 reps=10 다수결**이고 dev-3·시험지·DTS 가 같은 조건이다. 단 모델별 가용 칸이
  다르다(§7).
"""

if __name__ == "__main__":
    main()
