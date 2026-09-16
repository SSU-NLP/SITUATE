#!/usr/bin/env python3
"""`EACL_Paper.tex` 의 주 표(`tab:main`) 본문을 생성한다 — 우리 3모델 + 베이스라인 전부.

## 배열
사용자 지시대로 **태스크 계보 × 지도 방식**으로 묶는다:

    Ours
    Narrative event segmentation (NES) — unsupervised / supervised / LLM-based
    Dialogue topic segmentation (DTS) — unsupervised / supervised / LLM-based
    Controls (밀도일치 하한)

## 지표 (기존 표와 **같은 정의**를 쓴다 — 재정의 금지)
- Score = `hi_ontop.dts_scoring.score_dialogues` 의 `score` (문서 1편씩 계산)
- BOR   = 예측 경계 수 / gold 경계 수 (1.0 = 밀도 일치, >1 과분절). 굵게 표시 안 함
- Lift  = `full_comparison_dev3.lift_l1` = P(경계|예측) / P(경계)
- AUC   = 연속 점수의 순위 분리력. 🔴 **예측 덤프는 이진값이라 계산할 수 없다** → 베이스라인은 `--`
- Avg   = 6편 단순 평균 (기존 표의 M5 행으로 검산 완료)

## 기존 행은 건드리지 않는다
`M5` · `M4` · `M8` · `No boundary` 행은 **4 seed 평균**으로 다른 파이프라인에서 나온 값이다.
여기서 재계산하지 않고 `.tex` 에서 파싱해 그대로 옮긴다(굵게 표시만 다시 계산).

    env -C /tmp /root/Hi-OnTop/.venv/bin/python \\
        /root/Hi-OnTop/scripts/eacl_phase0/make_main_table.py --write
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))

PRED = REPO / "outputs/experiments/2026-07-29_baseline_compare/preds"
TEX = REPO / "EACL/EACL_Paper.tex"
OUT = REPO / "outputs/experiments/2026-08-04_online_baselines_close"

WORKS = [("pieman", "dev3"), ("tunnel", "dev3"), ("monkey", "dev3"),
         ("run", "test3"), ("gohitler", "test3"), ("mymotherslife", "test3")]

# (Setting, Supervision, [(덤프 이름, 표시 이름), ...])
# Setting = 태스크 계보, Supervision = 지도 방식. 등급 d 는 `audit_lookahead.py` 실측값(D-10).
# 🔴 Ours(M5/M4/M8) 는 사용자 템플릿대로 **NES / Unsupervised** 안에 둔다. 학습에 사람 경계
#    라벨을 안 쓰기 때문이다(앵커는 LLM 생성 = distant supervision) — 캡션에 명시.
GROUPS: list[tuple[str, str, list[tuple[str, str]]]] = [
    ("NES", "Unsupervised", [
        ("EmbeddingDelta-online-zero", "EmbeddingDelta-style Online$^{\\dagger}$"),
    ]),
    ("NES", "Weakly supervised", [
        ("__OURS__", ""),                       # SITUATE / Direct-Boundary 삽입 지점
    ]),
    ("NES", "Supervised", [
        # 🔴 발표된 모델이 아니라 **우리 재구현 ablation** — 원본 3종은 전부 독일어 전용이라
        #    영어 서사에 적용할 경로가 없다. 구조만 가져와 우리 앵커로 학습했다.
        ("SSC-anchor-online", "SSC-style Online (our anchors)$^{\\ddagger}$"),
    ]),
    ("NES", "LLM-based", [
        ("EM-LLM-SM", "EM-LLM"),                             # 원본이 이미 온라인
        ("BayesSurprise", "BayesSurprise"),                  # surprise 가 이미 인과적
        ("B3event-online-r10_gpt-4.1", "EventRecall-style Online (GPT-4.1)"),
        # 🔴 test3 세 편에서 경계를 하나도 안 찍는다(퇴화). 표에 그대로 싣고 캡션에 적는다
        ("B3event-online-r10_gpt-5", "EventRecall-style Online (GPT-5)"),
    ]),
    ("DTS", "Unsupervised", [
        ("TextTiling", "TextTiling"),
        ("TextTiling-online", "TextTiling-style Online"),
        ("DialSTART-online-zero", "DialSTART-style Online$^{\\dagger}$"),
    ]),
    ("DTS", "Supervised", [
        ("RoBERTa", "RoBERTa-style Online"),
        ("CSM-online-zero", "CSM-style Online"),
    ]),
    ("Controls", "---", [
        ("FixedInterval", "Fixed interval (density-matched)"),
        ("RandomRate", "Random (density-matched)"),
        ("__NOBND__", ""),
    ]),
]

# 🔒 명명 규칙 (사용자 지시 2026-08-04)
#   - 원본이 **오프라인**이라 우리가 온라인화한 것 → `XXX-style Online`
#   - 원본이 **이미 온라인**인 것            → 원래 이름 그대로 (EM-LLM · BayesSurprise · Membox)
#   - 오프라인 원본을 그대로 돌린 행         → 그냥 `XXX` (TextTiling)
#   - `Online` 은 **zero-lookahead 전용**이다. lookahead 가 있는 변형은 이 표에 넣지 않는다
#     (사용자 지시: "online 이면 d 가 있으면 안 된다"). 전 행의 실측 지연이 0 임을
#     `audit_lookahead.py` / 각 러너의 max-delay 로 확인했다.
#   - `†` = 원본 **방법 정의**에서 벗어난 개조 (DialSTART top-k readout, EmbeddingDelta 대칭 창).
#   - GraphSeg · GreedySeg 는 창 그래프/후보 누적이 알고리즘 본체라 zero 로 만들면 원본이
#     남지 않는다 ⇒ 사용자 결정으로 **표에서 제외**.

# 재현불가 항목 — 실행하지 않은 이유를 표 안에 남긴다 (빈칸으로 두면 '안 해봤다' 로 읽힌다)
# 🔴 이 주석 행은 **NES/Supervised 전용**이다. 다른 그룹이 비면 헤더째 생략한다
# (엉뚱한 그룹에 "재현불가" 가 붙으면 사실과 다른 말이 된다).
UNREPRO_AT = ("NES", "Supervised")
UNREPRO = ("BERT-SSC / half-stride, BERT-Context, TSE",
           "German-only: no English checkpoint or corpus")

METRICS = ("Score", "BOR", "Lift")   # 논문 §Evaluation Metrics — AUC 는 정의하지 않는다
NCOL = 3 + 7 * len(METRICS)          # Setting + Supervision + Method + (6편+Avg)×3

# 🔴 Lift 는 **예측한 위치에서만** 계산되는 비율이라 예측이 몇 개 없으면 값이 폭주한다
# (실측: SSC 가 tunnel 에서 예측 2개로 Lift 13.97 을 찍어 전 열의 굵게 표시를 가져갔다).
# BOR(=예측/gold)가 이 값 미만이면 그 칸의 Lift 는 굵게 표시 후보에서 뺀다. 값 자체는 싣는다.
LIFT_BOLD_MIN_BOR = 0.5

# 굵게 표시에서 제외 (하한·통제군)
NO_BOLD_ROWS = {"No boundary", "Fixed interval (density-matched)",
                "Random (density-matched)"}


def load_data():
    from baseline_compare import load_dataset  # noqa: PLC0415
    d = {}
    for ds in ("dev3", "test3"):
        for doc, _s, gold in load_dataset(ds):
            d[doc] = np.asarray(gold, np.int8)
    return d


def metrics(name: str, gold: dict) -> dict[str, list[float | None]] | None:
    """방법 1개 → {지표: [6편 값] + [Avg]}. 덤프가 없으면 None."""
    from full_comparison_dev3 import lift_l1               # noqa: PLC0415  🔒 동결 정의
    from hi_ontop.dts_scoring import score_dialogues       # noqa: PLC0415

    preds = {}
    for ds in ("dev3", "test3"):
        f = PRED / f"{name}_{ds}.json"
        if not f.exists():
            return None
        preds.update(json.loads(f.read_text()))
    if any(w not in preds for w, _ in WORKS):
        return None

    sc, bor, lift = [], [], []
    for w, _ds in WORKS:
        g = gold[w]
        p = np.asarray(preds[w][: len(g)], np.int8)
        y = np.concatenate([g, [0]]).tolist()
        q = np.concatenate([p, [0]]).tolist()
        sc.append(float(score_dialogues([y], [q])["score"]))
        ng = int(g.sum())
        bor.append(float(p.sum()) / ng if ng else float("nan"))
        lift.append(lift_l1([p], [g]))
    return {"Score": sc + [float(np.mean(sc))],
            "BOR": bor + [float(np.mean(bor))],
            "Lift": lift + [float(np.nanmean(lift))],
            }


# 🔒 Ours 행의 출처 = **4 seed 러너 산출** (ep100, personkey_i2). `.tex` 에 박혀 있던
# 값은 구세대라 어떤 산출물로도 재현되지 않았다(실측: M5 tunnel 0.269 vs 러너 0.405).
# 이제 `.tex` 를 파싱하지 않고 이 파일에서 직접 읽는다 — 표 전체가 한 파이프라인이 된다.
#
# 🔴 2026-08-04 작동점 정정: 이전 경로 `2026-08-04_i2_all_models/` 는 **문서 전역 분위수**
#    (top 6.60%) 산출이다. 그 규칙은 전이 t 를 판정하는 데 t+1..T 가 필요하므로 **오프라인**이고
#    (`eval_all_models_alldocs.py` 의 `metrics` docstring 참조), 논문이 주장하는
#    "Every online row is strictly zero-lookahead" 와 어긋난다. 판별법: 분위수는 개수를 고정하므로
#    **BOR 이 네 모델·네 seed 에서 비트 단위로 동일**하다. 적응임계면 모델마다 달라야 한다.
#    ⇒ 적응임계(`--mode online_nf --c 1.0`) 산출로 교체했다. 이 교체로 Ours Avg 가
#    SITUATE 0.405→0.376, Direct-Boundary 0.408→0.394 로 내려가고 순위가 바뀐다(사용자 승인).
OURS_SRC = REPO / "outputs/experiments/2026-08-04_all_models_adaptive/all_models_alldocs.json"
OURS_KEYS = [("M5", "SITUATE"), ("M8-pair", "Direct-Boundary")]
RUNNER_WORKS = ["dev-3/pieman", "dev-3/tunnel", "dev-3/monkey",
                "test3/run", "test3/gohitler", "test3/mymotherslife"]
SNAPSHOT = OUT / "ours_rows_snapshot.json"


def load_ours() -> dict[str, dict[str, list[float | None]]]:
    """러너 산출 → {표시이름: {지표: [6편] + [Avg]}}. seed 축은 평균한다.

    `No boundary` 행은 같은 파일의 `floor`(무경계 Score) 에서 만든다 — 별도 출처를 두면
    또 세대가 어긋난다.
    """
    d = json.loads(OURS_SRC.read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for key, disp in OURS_KEYS:
        row = d[key]
        m = {}
        for metric, field in (("Score", "score"), ("BOR", "bor"), ("Lift", "lift")):
            vals = [float(np.mean([x[field] for x in row[w]])) for w in RUNNER_WORKS]
            m[metric] = vals + [float(np.mean(vals))]
        out[disp] = m
    fl = [float(np.mean([x["floor"] for x in d["M5"][w]])) for w in RUNNER_WORKS]
    out["No boundary"] = {"Score": fl + [float(np.mean(fl))],
                          "BOR": [None] * 7, "Lift": [None] * 7}
    return out


def parse_tex_rows(tex: str) -> dict[str, dict[str, list[float | None]]]:
    """`No boundary` · `M5` · `M4` · `M8` 행의 값을 얻는다.

    이 값들은 **4 seed 평균**이라 `preds/` 덤프에서 재계산할 수 없다. 원래 `.tex` 에서
    파싱했는데, 이 러너가 `.tex` 를 다시 쓰면 표 포맷이 바뀌어(열 수·라벨) 파서가 깨진다.
    그래서 한 번 파싱한 값을 `ours_rows_snapshot.json` 에 굳혀 두고 **스냅샷을 우선**한다.
    """
    need = ("No boundary", "M5", "M4", "M8")
    if SNAPSHOT.exists():
        rows = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
        if all(k in rows for k in need):
            return {k: {m: v[m] for m in METRICS if m in v} for k, v in rows.items()}

    body = tex[tex.index("\\midrule", tex.index("\\label{tab:")):]
    body = body[: body.index("\\bottomrule")]
    rows = {}
    for chunk in body.split("\\\\"):
        chunk = re.sub(r"\\(midrule|addlinespace(\[[^\]]*\])?|cmidrule(\(l?r?\))?\{[^}]*\})",
                       "", chunk.strip()).strip()
        if not chunk:
            continue
        cells = [c.strip() for c in chunk.split("&")]
        # 옛 4지표 포맷만 파싱한다 (29 = 라벨1+7×4, 31 = 라벨3+7×4)
        if len(cells) == 29:
            label, nums, stride = cells[0], cells[1:], 4
        elif len(cells) == 31:
            label, nums, stride = cells[2], cells[3:], 4
        else:
            continue
        m = re.fullmatch(r"Ours \((M\d)\)", label)
        label = m.group(1) if m else label
        if label not in need:
            continue
        vals = []
        for c in nums:
            c = re.sub(r"\\textbf\{([^}]*)\}", r"\1", c).strip()
            vals.append(None if c in ("--", "---", "") else float(c))
        # 옛 순서 = Score, BOR, Lift, AUC → AUC 는 버린다 (논문에 정의 없음)
        rows[label] = {mm: vals[i::stride] for i, mm in enumerate(("Score", "BOR", "Lift", "AUC"))
                       if mm in METRICS}

    if all(k in rows for k in need):
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def fmt(v: float | None, metric: str, bold: bool) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    s = f"{v:.3f}" if metric == "Score" else f"{v:.2f}"
    return f"\\textbf{{{s}}}" if bold else s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help=".tex 를 실제로 갱신")
    args = ap.parse_args()

    gold = load_data()
    tex = TEX.read_text(encoding="utf-8")
    existing = load_ours()
    print(f"Ours 출처: {OURS_SRC.relative_to(REPO)} (4 seed 평균, ep100)")

    # 각 (Setting, Supervision) 블록의 행을 모은다. Ours 와 No boundary 는 .tex 파싱값이다.
    grouped: list[tuple[str, str, list[tuple[str, dict]]]] = []
    missing = []
    for setting, sup, members in GROUPS:
        rows: list[tuple[str, dict]] = []
        for key, disp in members:
            if key == "__OURS__":
                rows += [(disp_, existing[disp_]) for _k, disp_ in OURS_KEYS]
                continue
            if key == "__NOBND__":
                rows.append(("No boundary", existing["No boundary"]))
                continue
            r = metrics(key, gold)
            if r is None:
                missing.append(key)
                continue
            rows.append((disp, r))
        grouped.append((setting, sup, rows))
    ordered = [(s_, d, r) for s_, _sup, rs in grouped for d, r in rs]
    if missing:
        print(f"⚠️ 덤프 없음(표에서 제외): {missing}")

    # ── 열별 최고값 (Score·Lift·AUC 는 클수록 좋음, BOR 은 굵게 표시 안 함) ──
    best: dict[tuple[str, int], float] = {}
    for _g, disp, r in ordered:
        if disp in NO_BOLD_ROWS:
            continue
        for m in ("Score", "Lift"):
            for i, v in enumerate(r[m]):
                if v is None or np.isnan(v):
                    continue
                if m == "Lift":
                    b = r["BOR"][i]
                    if b is None or np.isnan(b) or b < LIFT_BOLD_MIN_BOR:
                        continue
                d = 3 if m == "Score" else 2
                v = round(v, d)
                if (m, i) not in best or v > best[(m, i)]:
                    best[(m, i)] = v

    # ── tabular 전체(열 정의 + 머리글 + 본문)를 새로 만든다 ────────────────────
    grp = " @{\\hspace{6pt}} ".join(["rrr"] * 7)
    lines = [f"\\begin{{tabular}}{{@{{}}lll {grp}@{{}}}}", "\\toprule"]
    works = ["Pieman", "Tunnel", "Monkey", "Run", "GoHitler", "MyMothersLife", "Avg"]
    lines.append("& & & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{\\textbf{{{w}}}}}"
                                       for w in works) + " \\\\")
    lines.append(" ".join(f"\\cmidrule(lr){{{4+3*i}-{6+3*i}}}" for i in range(7)))
    lines.append("\\textbf{Setting} & \\textbf{Supervision} & \\textbf{Method}\n"
                 + "\n".join("& Score$\\uparrow$ & BOR$\\,{\\to}\\,1$ & Lift$\\uparrow$" for _ in works) + " \\\\")
    lines.append("\\midrule")

    prev_setting = None
    for gi, (setting, sup, rows) in enumerate(grouped):
        if not rows and (setting, sup) != UNREPRO_AT:
            continue          # 빈 그룹은 헤더째 생략 (재현불가 주석은 NES/Supervised 전용)
        n = max(len(rows), 1)
        if setting != prev_setting:
            if prev_setting is not None:
                lines.append("\\midrule")
            # Setting 은 그 계보의 전 블록을 걸쳐 한 번만 쓴다 (빈 그룹은 제외)
            span = sum(max(len(r), 1) for s_, p_, r in grouped
                       if s_ == setting and (r or (s_, p_) == UNREPRO_AT))
            c1 = f"\\multirow{{{span}}}{{*}}{{\\textbf{{{setting}}}}}"
            prev_setting = setting
        else:
            lines.append(f"\\addlinespace\n\\cmidrule(l){{2-{NCOL}}}")
            c1 = ""
        c2 = f"\\multirow{{{n}}}{{*}}{{\\textit{{{sup}}}}}" if sup != "---" else ""
        if not rows:                       # NES/Supervised — 전부 재현불가
            lines.append(f"{c1}\n& {c2}\n& {UNREPRO[0]} & "
                         f"\\multicolumn{{{NCOL-3}}}{{l}}{{{UNREPRO[1]}}} \\\\")
            continue
        for ri, (disp, r) in enumerate(rows):
            # Ours 앞에 가는 줄 하나 (템플릿의 \cmidrule(l){3-17} 자리)
            if disp.startswith("Ours (") and ri > 0 and not rows[ri-1][0].startswith("Ours ("):
                lines.append(f"\\cmidrule(l){{3-{NCOL}}}")
            a = c1 if ri == 0 else ""
            b = c2 if ri == 0 else ""
            lines.append(f"{a}\n& {b}\n& " + _row(disp, r, best))

    lines += ["\\bottomrule", "\\end{tabular}"]
    body = "\n".join(lines)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "main_table_body.tex").write_text(body, encoding="utf-8")
    print(f"→ {OUT/'main_table_body.tex'} ({len(ordered)} 행)")

    if args.write:
        i = tex.index("\\label{tab:")
        h = tex.index("\\begin{tabular}", i)
        t = tex.index("\\end{tabular}", h) + len("\\end{tabular}")
        tex = tex[:h] + body + tex[t:]
        tex = tex.replace("\\label{tab:per-work}", "\\label{tab:main}")
        TEX.write_text(tex, encoding="utf-8")
        print(f"→ {TEX} 갱신")
    return 0


def _row(disp: str, r: dict, best: dict) -> str:
    """지표 dict → LaTeX 행 1개 (6편 + Avg × 4지표 = 28칸)."""
    cells = []
    for i in range(7):
        for m in METRICS:
            v = r[m][i]
            d = 3 if m == "Score" else 2
            b = (disp not in NO_BOLD_ROWS and m != "BOR" and v is not None
                 and not (isinstance(v, float) and np.isnan(v))
                 and round(v, d) == best.get((m, i)))
            if b and m == "Lift":
                bo = r["BOR"][i]
                b = bo is not None and not np.isnan(bo) and bo >= LIFT_BOLD_MIN_BOR
            cells.append(fmt(v, m, b))
    return f"{disp}\n& " + "\n& ".join(
        " & ".join(cells[j:j + 3]) for j in range(0, 7 * len(METRICS), len(METRICS))) + " \\\\"


if __name__ == "__main__":
    raise SystemExit(main())
