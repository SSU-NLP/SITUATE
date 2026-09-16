#!/usr/bin/env python3
"""STSS-Test-2 데이터 조립 — 저작권 없이 端到端 실행이 되는 유일한 무대.

## 무엇을 조립하는가
STSS 코퍼스는 저작권 자료라 전문이 공개되지 않는다. 예외가 **STSS-Test-2 2편**
(*Effi Briest*, *Aus guter Familie*, 둘 다 퍼블릭 도메인)이고, Zehe et al. (2025) 레포가
UIMA XMI 전문을 동봉했다. 이 모듈은 그 XMI 에서 원문을 뽑아 **공식 STSS JSON 포맷**
(`{"text", "sentences":[{begin,end}], "scenes":[{begin,end,type}]}`)으로 만든다.

## 🔴 gold 가 두 판본이다 — 반드시 구분할 것
| 판본 | 출처 | 경계 수 | 용도 |
|---|---|---|---|
| **2021 공식** | `embedding-delta/training/stss_results/t2/` | 219 · 219 | **STSS 2021 공식 순위표와 대조할 때** |
| 2025 확장 | `scene-segmentation/data/standoff/stss_test_2/` | 218 · 226 | Zehe et al. 2025 표와 대조할 때 |

두 판본은 같은 텍스트를 가리키지만 **주석이 다시 매겨졌다**. 정확 일치는
*Aus guter Familie* 107/219, *Effi Briest* 23/219 뿐이다.

## 🔴 오프셋 관례 함정 (이걸 모르면 F1 이 0 으로 나온다)
2021 gold 는 장면 시작을 **문장 앞 공백 위치**에 찍는다. 2025 문장 경계는 공백 뒤에서
시작한다. 실측 드리프트:

    Aus guter Familie : 0자(109개) · +1자(72개)
    Effi Briest       : +31자(70개, = 줄바꿈+들여쓰기) · +1자(44개) · 0자(23개)

즉 같은 경계인데 문자 오프셋이 다르다. 공식 채점기는 **문자 오프셋 완전일치**를 요구하므로
그대로 채점하면 전부 오답이 된다. 그래서 이 모듈은 모든 경계를 **문장 인덱스로 스냅**해
같은 격자에 올린다. 스냅이 점수를 바꾸지 않음은 `verify_grid.py` 가 검증한다
(저자의 공식 제출 예측을 스냅 후 재채점 → 발표값 재현).
"""
from __future__ import annotations

import bisect
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path

METHODS = Path(__file__).resolve().parents[1]
UPSTREAM = METHODS / "_upstream"
XMI_ZIP_DIR = UPSTREAM / "scene-segmentation/data/full/stss_test_2"
STANDOFF_DIR = UPSTREAM / "scene-segmentation/data/standoff/stss_test_2"
DUMP_DIR = UPSTREAM / "embedding-delta/training/stss_results/t2"

# 덤프 파일명 → 작품. 텍스트 길이로 확정했다(450,085 vs 453,780 / 664,060 vs 666,423).
DUMP_TO_DOC = {"1": "Aus guter Familie", "2": "Effi Briest"}

# 2025 standoff 의 독일어 타입 → 공식 포맷의 영어 타입
TYPE_MAP = {"Szene Ebene 1": "Scene", "Nonscene": "Nonscene", "Nicht-Szene": "Nonscene"}

_SOFA_RE = re.compile(r'<cas:Sofa[^>]*sofaString="(.*?)"\s*/>', re.S)


@dataclass
class Doc:
    name: str
    text: str
    sent_starts: list[int]          # 문장 시작 문자 오프셋 (오름차순)
    sent_ends: list[int]
    gold2025: list[tuple[int, str]]  # (문장 인덱스, 전이 타입)
    gold2021: list[tuple[int, str]]

    def sentence_index(self, offset: int) -> int:
        """문자 오프셋을 문장 인덱스로 스냅.

        규칙: offset 이 어떤 문장 안이면 그 문장, 문장 사이 공백이면 **다음 문장**.
        (경계는 '여기서 새 장면이 시작한다'는 뜻이므로 뒤쪽 문장에 붙이는 것이 맞다.)
        """
        i = bisect.bisect_right(self.sent_starts, offset) - 1
        if i < 0:
            return 0
        if offset < self.sent_ends[i]:
            return i
        return min(i + 1, len(self.sent_starts) - 1)


def _read_xmi_text(name: str) -> str:
    """XMI(zip) 에서 sofaString 을 꺼낸다. md5 는 standoff 와 대조해 검증한다."""
    import hashlib
    import zipfile

    zp = XMI_ZIP_DIR / f"{name}.xmi.zip"
    with zipfile.ZipFile(zp) as z:
        raw = z.read(f"{name}.xmi").decode("utf-8")
    m = _SOFA_RE.search(raw)
    if not m:
        raise RuntimeError(f"{name}: XMI 에서 sofaString 을 찾지 못했다")
    text = html.unescape(m.group(1))
    ref = json.loads((STANDOFF_DIR / f"{name}.json").read_text())["md5"]
    got = hashlib.md5(text.encode("utf-8")).hexdigest()
    if got != ref:
        raise RuntimeError(f"{name}: md5 불일치 {got} != {ref} — 텍스트 추출이 틀렸다")
    return text


def _boundaries_from_typed_scenes(scenes: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """[(시작오프셋, 장면타입)] → [(경계오프셋, '이전-to-현재')] (공식 규칙: 첫 장면 제외)."""
    out = []
    for i in range(1, len(scenes)):
        out.append((scenes[i][0], f"{scenes[i-1][1]}-to-{scenes[i][1]}"))
    return out


def load_doc(name: str) -> Doc:
    from common.stss_eval import parse_official_dump

    text = _read_xmi_text(name)
    st = json.loads((STANDOFF_DIR / f"{name}.json").read_text())
    sents = sorted(((s["start"], s["end"]) for s in st["sentences"]))
    sent_starts = [a for a, _ in sents]
    sent_ends = [b for _, b in sents]

    doc = Doc(name, text, sent_starts, sent_ends, [], [])

    scenes25 = [(s["start"], TYPE_MAP[s["type"]]) for s in st["scenes"]]
    doc.gold2025 = [(doc.sentence_index(o), t) for o, t in _boundaries_from_typed_scenes(scenes25)]

    dump_key = next(k for k, v in DUMP_TO_DOC.items() if v == name)
    d = parse_official_dump(DUMP_DIR / dump_key)["gold"]
    doc.gold2021 = [(doc.sentence_index(o), t) for o, t in zip(d.offsets, d.types)]
    return doc


def load_official_pred(name: str) -> list[tuple[int, str]]:
    """저자(Schneider et al.)의 **2021 공식 제출 예측**을 문장 격자로 스냅해 돌려준다."""
    from common.stss_eval import parse_official_dump

    doc = load_doc(name)
    dump_key = next(k for k, v in DUMP_TO_DOC.items() if v == name)
    d = parse_official_dump(DUMP_DIR / dump_key)["pred"]
    return [(doc.sentence_index(o), t) for o, t in zip(d.offsets, d.types)]


def load_test2() -> list[Doc]:
    return [load_doc(n) for n in DUMP_TO_DOC.values()]


def write_official_json(doc: Doc, out_path: Path, gold: str = "2021") -> None:
    """참가 시스템이 그대로 먹을 수 있는 **공식 STSS 입력 JSON** 을 쓴다.

    `scenes` 에는 gold 를 넣는다(공식 입력 포맷이 그렇게 생겼고, 참가 코드들은
    `scenes` 를 읽지 않고 `text`/`sentences` 만 쓴다 — 예측 시 누출 없음).
    """
    boundaries = doc.gold2021 if gold == "2021" else doc.gold2025
    scenes, prev_end, prev_type = [], 0, "Scene"
    starts = [0] + [doc.sent_starts[i] for i, _ in boundaries]
    types = ["Scene"] + [t.split("-to-")[1] for _, t in boundaries]
    for i, (s, t) in enumerate(zip(starts, types)):
        end = starts[i + 1] if i + 1 < len(starts) else len(doc.text)
        scenes.append({"begin": s, "end": end, "type": t})
    out = {
        "text": doc.text,
        "sentences": [{"begin": a, "end": b} for a, b in zip(doc.sent_starts, doc.sent_ends)],
        "scenes": scenes,
    }
    out_path.write_text(json.dumps(out, ensure_ascii=False))
