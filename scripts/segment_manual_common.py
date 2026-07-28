#!/usr/bin/env python3
"""사용자 직접-입력 화제분절 CLI 공용 유틸 (턴 파싱 + 출력 포매팅).

두 진입점이 공유한다:
- ``segment_manual.py``     — LLM 없는 최고버전 (Hi-OnTop δ_eff, 로컬 인코더)
- ``segment_manual_llm.py`` — LLM 있는 최고버전 (SeCom v1 프롬프트, full-context 1콜)

턴 입력 규약 (두 스크립트 공통):
- 한 줄 = 한 턴(발화). 빈 줄은 무시.
- ``Speaker: text`` 형태면 콜론 앞을 화자로 파싱(화자명은 공백 포함 가능, 30자 이내).
  콜론이 없으면 화자 = ``S`` (익명 단일 화자).
- ``#`` 로 시작하는 줄은 주석으로 무시.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# "Alice: 안녕" / "Speaker 1: ..." → (speaker, text). URL(http://)·시간(12:30) 오파싱 방지로
# 콜론 앞이 30자 이내이고 http 가 아닐 때만 화자로 취급.
_SPK = re.compile(r"^([^:]{1,30}):\s+(.*)$")


def parse_turns(lines: list[str]) -> list[dict[str, str]]:
    """텍스트 줄들 → 턴 dict 리스트 ``[{'speaker','text'}, ...]``."""
    turns: list[dict[str, str]] = []
    for raw in lines:
        line = raw.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = _SPK.match(line.strip())
        if m and not m.group(1).lower().startswith(("http", "https")):
            speaker, text = m.group(1).strip(), m.group(2).strip()
        else:
            speaker, text = "S", line.strip()
        if text:
            turns.append({"speaker": speaker, "text": text})
    return turns


def read_turns(file: str | None) -> list[dict[str, str]]:
    """``--file`` 이 있으면 파일에서, 없으면 stdin(대화형)에서 턴을 읽는다."""
    if file:
        lines = Path(file).read_text(encoding="utf-8").splitlines()
        return parse_turns(lines)
    if not sys.stdin.isatty():
        return parse_turns(sys.stdin.read().splitlines())
    # 대화형 입력
    print("── 발화를 한 줄에 하나씩 입력하세요. (형식: `화자: 내용` 또는 그냥 `내용`)")
    print("── 입력을 마치려면 빈 줄에서 Ctrl-D (Windows: Ctrl-Z) 를 누르세요.\n")
    lines: list[str] = []
    try:
        while True:
            lines.append(input())
    except EOFError:
        pass
    return parse_turns(lines)


def print_segments(turns: list[dict[str, str]], boundaries: list[bool],
                   extra: list[str] | None = None) -> None:
    """턴 + 경계(불리언 리스트) → 사람이 읽기 좋은 세그먼트 출력.

    ``boundaries[i]`` = True 면 턴 i 에서 새 화제 시작. ``extra[i]`` 는 턴별
    부가정보(예: δ_eff / graded score) 문자열(옵션).
    """
    seg_id = -1
    n_boundaries = 0
    print("\n" + "═" * 64)
    for i, t in enumerate(turns):
        is_b = boundaries[i]
        if is_b or seg_id < 0:
            seg_id += 1
            if seg_id > 0:
                n_boundaries += 1
            print(f"\n┏━ 화제 #{seg_id}")
        tag = extra[i] if extra else ""
        marker = "▶" if is_b and i > 0 else " "
        print(f"┃ {marker} [{i:>3}] ({t['speaker']}) {t['text']}   {tag}")
    print("\n" + "═" * 64)
    print(f"총 {len(turns)}턴 · 화제 {seg_id + 1}개 · 경계 {n_boundaries}개")
