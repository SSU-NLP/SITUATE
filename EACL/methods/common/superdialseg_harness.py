#!/usr/bin/env python3
"""하네스 C — Jiang et al. (2023) SuperDialseg 논문과 **동일 조건**의 평가 하네스.

## 하네스 3종 구분 (섞으면 안 된다)
`EACL/main_result.md` §5 가 못박은 분리:

| | 데이터 | 채점기 | 용도 |
|---|---|---|---|
| A | 우리 dev-3 / DTS 팩 | `hi_ontop.dts_scoring` deployable Score | main table. **논문 대조 불가** |
| B | Def-DTS 번들 | 우리 segeval 래퍼 | 우리 offline↔online 내부 비교 |
| **C** | **SuperDialseg 공식 배포본** | **저자 `SegmentationEvaluation`** | **논문 수치 대조 전용** ← 이 모듈 |

실측으로 A/B/C 가 얼마나 갈리는지는 `../texttiling/REPRO.md` §2 를 볼 것 —
**같은 TextTiling 인데 B→C 로 바꾸면 tiage Score 가 0.3081 → 0.3628 로 움직인다.**

## 🔴 업스트림 버그 2건 우회 (`benchmarks/` read-only 규약 유지)
1. `examples/reproduce/main.py:18` — `TextTilingCLSSegmenter` import(실제 이름은
   `TexttilingCLSSegmenter`). 게다가 `main.py:36` 은 저장소에 없는 `secret_config` 를 import.
   → main.py 를 쓰지 않고 그 **평가 루프만** 이식했다(`evaluate`).
2. `utils/data/data_collator.py:8` — `from .utils.data.load_dialseg import ...` 가
   relative 를 한 단계 더 써서 ModuleNotFoundError.
   → `sys.modules` 자기참조 shim(`shim_broken_relative_import`).

실행되는 알고리즘·채점 코드는 저자 원본 그대로다.

## 사용법
    PYTHONPATH=benchmarks/superdialseg/src .venv-superdialseg/bin/python <러너>
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "benchmarks/superdialseg_data"
DATASETS = ("superseg", "tiage", "dialseg711")

# Jiang et al. (2023) EMNLP, SuperDialseg **Table 3** — 비지도 섹션
# 출처 대조: outputs/reports/dts_result.md (LaTeX 표, † 표시 행)
PAPER_TABLE3 = {
    "texttiling": {
        "tiage":      {"pk": 0.469, "windowdiff": 0.488, "f1(binary)": 0.204, "total_score": 0.363},
        "dialseg711": {"pk": 0.470, "windowdiff": 0.493, "f1(binary)": 0.245, "total_score": 0.382},
        "superseg":   {"pk": 0.441, "windowdiff": 0.453, "f1(binary)": 0.388, "total_score": 0.471},
    },
    "graphseg": {
        "tiage":      {"pk": 0.496, "windowdiff": 0.515, "f1(binary)": 0.238, "total_score": 0.366},
        "dialseg711": {"pk": 0.412, "windowdiff": 0.442, "f1(binary)": 0.392, "total_score": 0.483},
        "superseg":   {"pk": 0.450, "windowdiff": 0.454, "f1(binary)": 0.249, "total_score": 0.398},
    },
    "greedyseg": {
        "tiage":      {"pk": 0.469, "windowdiff": 0.506, "f1(binary)": 0.181, "total_score": 0.341},
        "dialseg711": {"pk": 0.381, "windowdiff": 0.410, "f1(binary)": 0.445, "total_score": 0.525},
        "superseg":   {"pk": 0.490, "windowdiff": 0.494, "f1(binary)": 0.365, "total_score": 0.437},
    },
    "csm": {
        "tiage":      {"pk": 0.400, "windowdiff": 0.420, "f1(binary)": 0.427, "total_score": 0.509},
        "dialseg711": {"pk": 0.278, "windowdiff": 0.302, "f1(binary)": 0.610, "total_score": 0.660},
        "superseg":   {"pk": 0.462, "windowdiff": 0.467, "f1(binary)": 0.381, "total_score": 0.458},
    },
}


def shim_broken_relative_import() -> None:
    """`super_dialseg.utils.data.data_collator` 의 relative import 오타를 우회한다.

    그 파일의 `.` 은 이미 `super_dialseg.utils.data` 인데 `.utils.data.load_dialseg` 를
    import 한다. `...utils.data.utils` 라는 이름에 **패키지 자신**을 꽂아 경로가
    자기 자신으로 되돌아오게 만든다. 코드는 수정하지 않는다.
    """
    import sys

    import super_dialseg.utils.data as sdd

    sys.modules.setdefault("super_dialseg.utils.data.utils", sdd)
    sys.modules.setdefault("super_dialseg.utils.data.utils.data", sdd)


def segmentation_file(dataset: str) -> Path:
    return DATA / dataset / "segmentation_file_test.json"


def evaluate(segmenter, dataset: str, *, batch_size: int = 1, num_workers: int = 4,
             ignore_index: int = -100, plain_text: bool = True) -> dict:
    """저자 `ReproducePipeline.easy_evaluate` (main.py:73-121) 를 그대로 이식.

    `plain_text=True` → `DataCollatorPlainText` (utterances 를 넘긴다).
    저자 main.py 는 texttiling·bayesseg·greedyseg·csm·texttiling_* 전부 이 collator 를 쓴다.
    """
    import torch
    from torch.utils.data import DataLoader, SequentialSampler
    from tqdm import tqdm

    shim_broken_relative_import()
    from super_dialseg.utils.data.data_collator import (DataCollatorForSegmentationLabels,
                                                        DataCollatorPlainText)
    from super_dialseg.utils.data.load_dialseg import DialsegDataset

    f = segmentation_file(dataset)
    if not f.exists():
        raise FileNotFoundError(f"{f} 없음")

    dataset_obj = DialsegDataset(segmentation_file=str(f))
    loader = DataLoader(
        dataset_obj, sampler=SequentialSampler(dataset_obj), batch_size=batch_size,
        num_workers=num_workers, persistent_workers=num_workers > 0, drop_last=False,
        pin_memory=torch.cuda.is_available(),
        collate_fn=DataCollatorPlainText() if plain_text else DataCollatorForSegmentationLabels(),
    )

    with tqdm(total=len(loader), desc=dataset) as pbar:
        for batch in loader:
            for j, labels in enumerate(batch["labels"]):
                inputs = {k: v[j] for k, v in batch.items()}
                try:
                    labels = labels.cpu().numpy().tolist()
                except AttributeError:
                    pass
                seg_labels = [x for x in labels if x != ignore_index]
                seg_preds = segmenter.forward(inputs)
                # 저자 주석 그대로: 값 범위를 [0,1] 로 유지하려 마지막을 0 으로 강제
                seg_labels[-1] = 0
                seg_preds[-1] = 0
                segmenter.evaluation.add(seg_labels, seg_preds)
            pbar.update(1)

    return {k: float(v) for k, v in segmenter.evaluation.compute().items()}


def evaluate_online(segmenter, dataset: str, *, batch_size: int = 1, num_workers: int = 4,
                    ignore_index: int = -100, max_dialogues: int | None = None,
                    progress: bool = True) -> dict:
    """**온라인(prefix-causal)** 평가 — 오프라인과 **같은 데이터·같은 채점기**로 잰다.

    ## 변환 규칙: prefix-recompute
    턴 t 의 경계 판정을 `u_1..u_t` **만으로** 내린다. 구체적으로 오프라인 분절기를
    prefix 에 그대로 돌려 **최전선 전이(t−1 → t)** 의 판정만 채택하고, 그보다 앞선
    전이에 대한 이번 호출의 판정은 **버린다**(이미 확정됐으므로 재수정 없음).

    ```
    for t in 2..n:  flags = segmenter.forward(u_1..u_t);  pred[t-1] = flags[t-2]
    ```
    (`methods/texttiling/online/prefix.py` → `scripts/run_texttiling_prefix.py` 가 검증한
    규약과 동일하다. 코드가 아니라 규약을 따온 것 — 그쪽은 하네스 B 전용이다.)

    ## 왜 prefix-recompute 인가
    **바뀌는 변수가 '미래 문맥의 유무' 하나뿐이다.** 모델·가중치·하이퍼·데이터·채점기가
    전부 오프라인과 같으므로 `Δ = online − offline` 이 곧 **미래 문맥의 기여**다.
    스트리밍 근사(running threshold, delay=k 등)를 쓰면 근사 오차가 섞여 Δ 의 해석이 흐려진다.

    ## 비용
    문서당 O(n²) 회 forward. TextTiling 같은 경량 방법은 괜찮지만 BERT 계열은 무겁다.
    `max_dialogues` 로 표본을 줄일 수 있고, 줄이면 **반드시 보고에 명시**한다.
    """
    import torch
    from torch.utils.data import DataLoader, SequentialSampler
    from tqdm import tqdm

    shim_broken_relative_import()
    from super_dialseg.utils.data.data_collator import DataCollatorPlainText
    from super_dialseg.utils.data.load_dialseg import DialsegDataset

    f = segmentation_file(dataset)
    dataset_obj = DialsegDataset(segmentation_file=str(f))
    loader = DataLoader(
        dataset_obj, sampler=SequentialSampler(dataset_obj), batch_size=batch_size,
        num_workers=num_workers, persistent_workers=num_workers > 0, drop_last=False,
        pin_memory=torch.cuda.is_available(), collate_fn=DataCollatorPlainText(),
    )

    n_done, n_miss = 0, 0
    bar = tqdm(total=(max_dialogues or len(dataset_obj)), desc=f"{dataset}(online)",
               disable=not progress)
    for batch in loader:
        for j, labels in enumerate(batch["labels"]):
            if max_dialogues is not None and n_done >= max_dialogues:
                break
            try:
                labels = labels.cpu().numpy().tolist()
            except AttributeError:
                pass
            seg_labels = [x for x in labels if x != ignore_index]
            utts = batch["utterances"][j]
            n = len(seg_labels)

            preds = [0] * n
            for t in range(2, n + 1):
                sub = {k: (v[j][:t] if k == "utterances" else v[j]) for k, v in batch.items()}
                # 🔴 짧은 prefix 에서는 분절기가 예외를 던진다(예: nltk TextTiling 의
                # "No paragraph breaks were found"). 기존 러너 규약과 동일하게
                # **경계 없음(0) + miss 카운트** 로 처리한다 — 그 턴을 건너뛰지 않는다.
                try:
                    flags = segmenter.forward(sub)
                except Exception:  # noqa: BLE001 — 분절기마다 예외 종류가 다르다
                    flags = None
                if flags is None or len(flags) < t - 1:
                    n_miss += 1
                    continue
                # 최전선 전이 = u_{t-1} 이 세그먼트를 끝냈는가
                preds[t - 2] = int(flags[t - 2])
            seg_labels[-1] = 0
            preds[-1] = 0
            segmenter.evaluation.add(seg_labels, preds)
            n_done += 1
            bar.update(1)
        if max_dialogues is not None and n_done >= max_dialogues:
            break
    bar.close()

    out = {k: float(v) for k, v in segmenter.evaluation.compute().items()}
    out["_n_dialogues"] = n_done
    out["_n_frontier_miss"] = n_miss
    return out


METRICS = ("pk", "windowdiff", "f1(binary)", "total_score")


def print_comparison(method: str, rows: dict[str, dict], tolerance: float = 0.02) -> dict[str, bool]:
    """우리 값 vs 논문 Table 3 대조표를 찍고 데이터셋별 판정을 돌려준다."""
    paper = PAPER_TABLE3[method]
    print("\n" + "=" * 78)
    print(f"{method} — 논문(Jiang et al. 2023, SuperDialseg Table 3) 대조 · 하네스 C")
    print("=" * 78)
    print(f"{'데이터셋':<12}{'지표':<14}{'우리':>9}{'논문':>9}{'차이':>9}   판정")
    verdicts = {}
    for ds, res in rows.items():
        ok_all = True
        for m in METRICS:
            if m not in res:
                continue
            d = res[m] - paper[ds][m]
            ok = abs(d) <= tolerance
            ok_all &= ok
            print(f"{ds:<12}{m:<14}{res[m]:>9.4f}{paper[ds][m]:>9.3f}{d:>+9.4f}   {'✅' if ok else '🔴'}")
        verdicts[ds] = ok_all
        print(f"{'':<12}{'→ 판정':<14}{'':>9}{'':>9}{'':>9}   {'✅ 재현' if ok_all else '🔴 미재현'}")
    n = sum(verdicts.values())
    print(f"\n종합: {n}/{len(verdicts)} 데이터셋 재현 (허용오차 ±{tolerance})")
    return verdicts
