#!/usr/bin/env python3
"""TextTiling 논문 수치 재현 — Jiang et al. (2023) SuperDialseg Table 3.

## 왜 이 파일이 필요했나
저장소에 TextTiling 실행 기록은 있지만(`outputs/experiments/2026-05-20_texttiling_offline/`)
**논문 수치와의 대조표도, 재현 판정도 없다.** REPORT 는 우리 값만 표로 두고
논문값은 "한계" 절에 문장으로만 적은 뒤 *"정확 일치 아님, 방향·정상동작 검증용"* 이라고
넘겼다. 이 스크립트가 그 구멍을 닫는다.

## 하네스 = C (논문과 동일 조건)
`EACL/main_result.md` §5 가 못박은 3종 분리를 지킨다:
  A = main table 채점기(우리 deployable Score) — 논문 대조와 무관
  B = 우리 오프라인 러너(Def-DTS 데이터 + segeval) — 2026-05-20 REPORT 가 쓴 것
  **C = 논문 조건** ← 이 스크립트

C 를 성립시키는 세 요소를 전부 저자 것으로 맞춘다:
  - 데이터  : `benchmarks/superdialseg_data/{superseg,tiage,dialseg711}/segmentation_file_test.json`
              (SuperDialseg 공식 배포본. Def-DTS 번들이 아니다)
  - 모델    : `super_dialseg.models.texttiling.TexttilingSegmenter` (nltk w=10 · k=6, 무수정 호출)
  - 채점기  : `super_dialseg.metrics.SegmentationEvaluation` (Pk · WindowDiff · F1 · total_score)

## 🔴 업스트림 버그 때문에 저자 스크립트를 그대로 못 돌린다
`benchmarks/superdialseg/examples/reproduce/main.py:18` 이

    from ...modeling_texttiling import TexttilingSegmenter, TextTilingCLSSegmenter

를 import 하는데, 실제 클래스명은 `TexttilingCLSSegmenter` 다(`modeling_texttiling.py:37`,
'tiling' 의 t 가 소문자). **저자 재현 스크립트가 ImportError 로 죽는다.**
또 `main.py:36` 은 저장소에 없는 `secret_config` 모듈을 import 한다.

`benchmarks/` 는 read-only 규약이므로 고치지 않고, **저자 `ReproducePipeline` 의 평가 루프를
그대로 옮겨** 여기서 돈다(배치 순회 → 문서별 forward → 마지막 라벨/예측 0 강제 → evaluation.add).
알고리즘·채점 로직은 한 줄도 바꾸지 않았다.

    PYTHONPATH=benchmarks/superdialseg/src \\
      .venv-superdialseg/bin/python EACL/methods/texttiling/run_paper_repro.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "benchmarks/superdialseg_data"
OUT = REPO / "outputs/experiments/2026-08-03_texttiling_paper_repro"

# Jiang et al. (2023) EMNLP, SuperDialseg Table 3 · TextTiling 행
PAPER = {
    "superseg":   {"pk": 0.441, "windowdiff": 0.453, "f1(binary)": 0.388, "total_score": 0.471},
    "tiage":      {"pk": 0.469, "windowdiff": 0.488, "f1(binary)": 0.204, "total_score": 0.363},
    "dialseg711": {"pk": 0.470, "windowdiff": 0.493, "f1(binary)": 0.245, "total_score": 0.382},
}
DATASETS = ("superseg", "tiage", "dialseg711")


def _shim_broken_relative_import() -> None:
    """🔴 업스트림 버그 우회 (코드 수정 없이).

    `super_dialseg/utils/data/data_collator.py:8` 이

        from .utils.data.load_dialseg import DialsegSample
        from .utils.data import ROLE2LABEL, DOC2DIAL_DA2LABEL

    를 하는데, 그 파일의 `.` 은 이미 `super_dialseg.utils.data` 다. 즉
    `super_dialseg.utils.data.utils.data.load_dialseg` 를 찾아 ModuleNotFoundError 가 난다
    (relative import 를 한 단계 더 쓴 오타). `benchmarks/` 는 read-only 규약이라 고치지 않고,
    `...utils.data.utils` 라는 이름에 **패키지 자신**을 꽂아 경로가 자기 자신으로 되돌아오게 한다.
    실행되는 코드는 저자 원본 그대로다.
    """
    import sys

    import super_dialseg.utils.data as sdd

    sys.modules.setdefault("super_dialseg.utils.data.utils", sdd)
    sys.modules.setdefault("super_dialseg.utils.data.utils.data", sdd)


def evaluate(segmenter, segmentation_file: Path, batch_size: int, num_workers: int,
             ignore_index: int = -100) -> dict:
    """저자 `ReproducePipeline.easy_evaluate` 를 그대로 옮긴 것 (main.py:73-121)."""
    import torch
    from torch.utils.data import DataLoader, SequentialSampler
    from tqdm import tqdm

    _shim_broken_relative_import()
    from super_dialseg.utils.data.data_collator import DataCollatorPlainText
    from super_dialseg.utils.data.load_dialseg import DialsegDataset

    # 🔒 texttiling 의 collator 는 `DataCollatorPlainText` 다 (저자 main.py:185).
    #    기본값 `DataCollatorForSegmentationLabels` 는 labels 만 넘겨서
    #    `TexttilingSegmenter.forward` 가 `inputs['utterances']` 에서 KeyError 를 낸다.
    dataset = DialsegDataset(segmentation_file=str(segmentation_file))
    loader = DataLoader(
        dataset, sampler=SequentialSampler(dataset), batch_size=batch_size,
        num_workers=num_workers, persistent_workers=num_workers > 0, drop_last=False,
        pin_memory=torch.cuda.is_available(),
        collate_fn=DataCollatorPlainText(),
    )

    with tqdm(total=len(loader)) as pbar:
        for batch in loader:
            for j, labels in enumerate(batch["labels"]):
                inputs = {k: v[j] for k, v in batch.items()}
                try:
                    labels = labels.cpu().numpy().tolist()
                except AttributeError:
                    pass
                seg_labels = [x for x in labels if x != ignore_index]
                seg_preds = segmenter.forward(inputs)
                # 저자 주석 그대로: 마지막 값을 0 으로 강제해 [0,1] 범위를 지킨다
                seg_labels[-1] = 0
                seg_preds[-1] = 0
                segmenter.evaluation.add(seg_labels, seg_preds)
            pbar.update(1)

    return segmenter.evaluation.compute()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-batch-size", type=int, default=128, help="저자 스크립트 기본값")
    ap.add_argument("--num-workers", type=int, default=8, help="저자 스크립트 기본값")
    ap.add_argument("--tolerance", type=float, default=0.02,
                    help="✅ 판정 임계 — |우리 − 논문| 이 이 값 이하면 재현으로 본다")
    args = ap.parse_args()

    from super_dialseg.models.texttiling.modeling_texttiling import TexttilingSegmenter

    rows = {}
    for ds in DATASETS:
        f = DATA / ds / "segmentation_file_test.json"
        if not f.exists():
            print(f"🔴 {ds}: {f} 없음 — 건너뜀")
            continue
        print(f"\n=== {ds} ===")
        seg = TexttilingSegmenter()          # nltk w=10 · k=6 (저자 기본값)
        res = evaluate(seg, f, args.test_batch_size, args.num_workers)
        rows[ds] = {k: float(v) for k, v in res.items()}

    print("\n" + "=" * 78)
    print("TextTiling — 논문(Jiang et al. 2023, SuperDialseg Table 3) 대조 · 하네스 C")
    print("=" * 78)
    hdr = f"{'데이터셋':<12}{'지표':<14}{'우리':>9}{'논문':>9}{'차이':>9}   판정"
    print(hdr)
    verdicts = {}
    for ds, res in rows.items():
        ok_all = True
        for metric in ("pk", "windowdiff", "f1(binary)", "total_score"):
            ours = res.get(metric)
            paper = PAPER[ds][metric]
            if ours is None:
                continue
            d = ours - paper
            ok = abs(d) <= args.tolerance
            ok_all &= ok
            print(f"{ds:<12}{metric:<14}{ours:>9.4f}{paper:>9.3f}{d:>+9.4f}   {'✅' if ok else '🔴'}")
        verdicts[ds] = ok_all
        print(f"{'':<12}{'→ 판정':<14}{'':>9}{'':>9}{'':>9}   "
              f"{'✅ 재현' if ok_all else '🔴 미재현'}")

    n_ok = sum(verdicts.values())
    print(f"\n종합: {n_ok}/{len(verdicts)} 데이터셋 재현 "
          f"(허용오차 ±{args.tolerance})")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(
        {"ours": rows, "paper": PAPER, "tolerance": args.tolerance, "verdicts": verdicts},
        ensure_ascii=False, indent=2))
    print(f"저장: {OUT / 'results.json'}")
    return 0 if n_ok == len(verdicts) else 1


if __name__ == "__main__":
    raise SystemExit(main())
