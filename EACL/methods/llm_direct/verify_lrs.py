#!/usr/bin/env python3
"""B2 의 `min_check_length` 계산이 저자 구현과 같은 값을 내는지 검사한다.

저자 `longestRepeatedSubstring` 은 O(n²) DP 라 수십만 자 원문에서 못 쓴다. 우리는
빠른 구현으로 갈아끼웠는데, **값이 1이라도 다르면 재정렬 탐색 길이가 달라져 분절이
통째로 바뀐다.** 그래서 무작위 문자열로 두 구현을 직접 대조한다.

🔴 주의: 저자 DP 는 `(j-i) > LCSRe[i-1][j-1]` 조건으로 **겹치는 반복을 배제**한다.
즉 "가장 긴 반복 부분문자열"이 아니라 "가장 긴 **비중첩** 반복 부분문자열"이다.
`aaaa` 에서 저자 답은 `aa`(2)이지 `aaa`(3)이 아니다. 이 차이를 놓치면 조용히 어긋난다.

    uv run python EACL/methods/llm_direct/verify_lrs.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_direct.segment import longest_repeated_substring_len  # noqa: E402


def author_lrs(s: str) -> str:
    """저자 `longestRepeatedSubstring` 원문 이식 (O(n²) DP, 비중첩)."""
    n = len(s)
    lcsre = [[0] * (n + 1) for _ in range(n + 1)]
    res_length = 0
    index = 0
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            if s[i - 1] == s[j - 1] and lcsre[i - 1][j - 1] < (j - i):
                lcsre[i][j] = lcsre[i - 1][j - 1] + 1
                if lcsre[i][j] > res_length:
                    res_length = lcsre[i][j]
                    index = max(i, index)
            else:
                lcsre[i][j] = 0
    return s[index - res_length:index] if res_length > 0 else ""


def main() -> int:
    rng = random.Random(20260803)
    bad = 0
    cases = 400
    for _ in range(cases):
        alpha = rng.choice(["ab", "abc", "abcdefg", "ab cd ef "])
        n = rng.randint(1, 90)
        s = "".join(rng.choice(alpha) for _ in range(n))
        a = len(author_lrs(s))
        b = longest_repeated_substring_len(s)
        if a != b:
            bad += 1
            if bad <= 3:
                print(f"  🔴 불일치 {s!r}: 저자={a} 우리={b}")
    print(f"랜덤 {cases}케이스: {'✅ 전부 일치' if not bad else f'🔴 {bad}건 불일치'}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
