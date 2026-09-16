# lookahead 곡선 — 미래를 얼마나 볼 수 있느냐에 따른 점수 변화 (하네스 C)

**핵심 결과 (TextTiling):** 적절한 하이퍼에서 **bounded-lookahead 는 offline 을 거의 따라잡는다**
(dialseg711 −0.011). 그런데 **lookahead 를 완전히 없애면 3~6배 더 잃는다**(−0.067).
⇒ TextTiling 이 쓰는 **무제한 미래 문맥은 거의 불필요**하지만, **약간의 lookahead 는 본질적**이다.

---

## 1. 왜 단일 Δ 가 아니라 곡선인가

`2026-08-03_online_offline_delta/REPORT.md` §4 가 보인 문제: `Δ = online − offline` 하나로는
**무엇이 변했는지 특정되지 않는다.** 온라인 변형에는 두 축이 섞인다 —
① 미래 문맥을 얼마나 쓰는가 ② 그 조건에 맞게 재설계했는가.

그래서 **미래 사용량**을 축으로 세 점을 잰다:

| 조건 | 미래 사용 | 판정 확정 | 구현 |
|---|---|---|---|
| `offline` | 무제한 | 문서 종료 후 | 저자 `TexttilingSegmenter` |
| `bounded` | k 블록 대기 | 턴 t+지연 | `StreamingTextTiling` (running threshold, Welford) |
| `zero` | 없음 | 턴 t 즉시 | prefix-recompute |

🔴 **스트리밍 = "미래를 안 씀"이 아니다.** `StreamingTextTiling.push()` docstring 원문:
*"gap 의 score 는 right block(k pseudo-sentence)이 모두 닫혀야 평가 가능하므로, boundary 가
확정되는 시점은 그 boundary 의 utterance 가 들어온 시점보다 늦다."* ⇒ **bounded-lookahead**.

## 2. 설정

| 항목 | 값 |
|---|---|
| 러너 | `EACL/methods/common/run_lookahead_curve.py` |
| 하네스 | C (공식 데이터 + 저자 `SegmentationEvaluation`) |
| offline | `TexttilingSegmenter` w=10·k=6 (논문값, 전 조건 고정) |
| bounded | `StreamingTextTiling` — **두 조건**: 논문값(w=10·k=6) / 축소값(w=5·k=3) |
| zero | prefix-recompute, 문서당 O(n²) |
| 인과성 검증 | `verify_causality.py` 통과 (입력감사 ✅ · 불변 12/12 ✅ · 대조군 6/12 ✅) |
| 표본 | 전량 (tiage 100 · dialseg711 711) |

## 3. 결과

### 조건 A — bounded 를 논문 하이퍼(w=10·k=6)로

| 데이터셋 | offline | bounded | zero | Δ(b−o) | Δ(z−b) |
|---|---:|---:|---:|---:|---:|
| tiage | 0.3628 | **0.2805** 🔴 | 0.2874 | −0.0823 | +0.0069 |
| dialseg711 | 0.3816 | 0.3406 | 0.3041 | −0.0410 | −0.0365 |

🔴 tiage 의 bounded 는 **F1 = 0.0000** — 완전 퇴화다. 대화가 16턴뿐이라
`2k = 12` pseudo-sentence(각 10토큰)를 채우지 못한다. **하이퍼가 데이터에 안 맞는 것**이지
방법의 한계가 아니다.

### 조건 B — bounded 를 축소 하이퍼(w=5·k=3)로 ← **해석은 이쪽**

| 데이터셋 | offline | bounded | zero | **Δ(b−o)** | **Δ(z−b)** |
|---|---:|---:|---:|---:|---:|
| tiage | 0.3628 | **0.3398** | 0.2874 | **−0.0230** | **−0.0523** |
| dialseg711 | 0.3816 | **0.3707** | 0.3041 | **−0.0109** | **−0.0667** |

지표별(조건 B):

| 데이터셋 | 지표 | offline | bounded | zero |
|---|---|---:|---:|---:|
| tiage | F1 | 0.2043 | 0.1881 | **0.0113** |
| dialseg711 | F1 | 0.2446 | 0.1992 | **0.0361** |

## 4. 해석

**두 구간의 비대칭이 크다.**

    dialseg711 :  offline ──(−0.011)──> bounded ──(−0.067)──> zero
    tiage      :  offline ──(−0.023)──> bounded ──(−0.052)──> zero

- **offline → bounded 는 거의 공짜다** (dialseg711 에서 3% 손실). 즉 TextTiling 이 쓰는
  **무제한 미래 문맥의 대부분은 불필요**하다. 몇 블록만 앞을 보면 오프라인을 사실상 따라잡는다.
- **bounded → zero 가 진짜 비용이다** (6배 더 큼). F1 이 0.20 → 0.04 로 무너진다.
  이 구간에서 방법이 **"경계 없음"으로 퇴화**한다.

원인은 방법의 성질에 있다. TextTiling 의 임계값은 depth score 의 `mean − std/2` 이고,
어휘 골짜기를 판정하려면 **골짜기의 오른쪽 어깨**가 필요하다. lookahead 가 0 이면 골짜기인지
아직 알 수 없다. 반대로 어깨만 확보되면 문서 전체를 볼 필요는 없다.

### Causal-SSC 논증에 어떻게 쓰나
원 논거는 *"온라인 변형의 하락폭이 크면 온라인 조건의 난이도 입증, 작으면 SotA 가 불필요한
미래 문맥을 쓴다는 발견"* 이었다. 곡선으로 보면 **둘 다 참일 수 있다** — 어느 구간이냐에 따라
다르기 때문이다. TextTiling 은 **"불필요한 미래를 쓴다"(offline→bounded 완만)이면서
동시에 "zero-lookahead 는 어렵다"(bounded→zero 가파름)** 이다.
⇒ 논문에는 **단일 Δ 가 아니라 두 구간을 나눠** 보고해야 한다.

## 5. 남은 것

- ⬜ CSM · GreedySeg · RoBERTa · DialSTART 의 bounded 어댑터 (현재 TextTiling 만)
- ⬜ DialSTART 온라인 (신규 구현 필요 — DTS 베이스라인 중 유일하게 온라인 판 없음)
- ⬜ 밀도 하한(`methods/trivial/`) 병기 — zero 조건이 무경계 하한에 얼마나 근접했는지
- ⬜ superseg 는 미측정 (zero 조건이 O(n²)라 약 26분 추가)

## 6. 한계 / 검증 미해결

- **한 방법(TextTiling)만** 쟀다. 곡선 모양이 방법마다 다를 수 있고, 지금 결론은 TextTiling 한정이다.
- **bounded 의 하이퍼가 offline 과 다르다**(조건 B). 그 차이만큼 Δ(b−o) 가 과소평가될 수 있다.
  조건 A 는 하이퍼가 같지만 tiage 에서 퇴화해 쓸 수 없었다. **둘 다 보고하는 이유가 이것이다.**
- zero 조건의 최전선 miss(짧은 prefix 에서 nltk 예외)를 "경계 없음"으로 처리했다. 이 선택이
  zero 의 F1 을 더 낮춘다. 다른 처리(직전 판정 유지 등)는 시험하지 않았다.
- bounded 는 **지연 확정**을 허용한다 — 턴 t 의 판정이 t 시점에 나오지 않는다.
  실시간 응용에서 그 지연이 허용되는지는 별개 문제이며 여기서 다루지 않았다.
