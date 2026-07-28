# dsv4pro 실패의 축별 귀속 (재설계가 과녁을 맞추는가) — 예산0

> 원자료 `outputs/experiments/2026-07-20_labeler_swap_dsv4pro/agreement.json`(§2-2 Gate B).
> 값 = deepseek↔fable5 **리셋 A↔B 일치 F1**(낮을수록 두 라벨러 해석 발산 큼). Gate C는 vs 사람이나,
> 이 발산 F1이 "재설계 프롬프트 문구가 어느 축에서 가장 중요한가"의 진단이다(fable5=인간통과 기준).

## 🔴 최상위 발견: 축별 F1은 ΔScore를 예측하지 못한다 (앞으로 판정 시 항상 걸릴 함정)

표를 세로로 보면 **완전히 거꾸로**다 — 축별 라벨이 가장 잘 맞은 문서가 가장 크게 망가졌다.

| doc | 축별 A↔B F1 | ΔScore | count≥3 비율 (fable5→dsv4pro) |
|---|---|---|---|
| pieman | **최고** (0.80/0.89/0.78/0.73/0.44) | **−0.253 (최악)** | 0.158 → **0.105** (−33%; gold 0.189) |
| monkey | **최악** (space 0.47 + intent 0.38) | **−0.052 (최선)** | 0.034 → 0.022 (gold 0.074) |

원인 둘:
1. **동시리셋(co-reset) 구조 붕괴.** 경계 = `count ≥ t*=3`이라 **개별 축 정확도가 아니라 축들이 같은 전이에서 함께 터지느냐**가 결정. pieman `protagonist+goal` 동시리셋 11→4로 무너지며 count≥3이 0.158→0.105로 급락 → 경계를 놓침. 축 하나하나 80% 맞아도 3개 동시 패턴이 깨지면 Score가 무너진다.
2. **기준선 높이 효과.** pieman fable5 Score 0.634(3편 최고, 떨어질 여지 큼)·monkey 0.359(최저). 게다가 0.634는 **t\*를 고른 바로 그 문서(dev=Pieman)** 숫자라 **가장 오염**된 값 → pieman이 구조적으로 Gate C 통과 최난.

**⇒ 판정·스모크 지표를 바꾼다: unspecified 비율·축별 F1만으로 Gate C를 예측 못 함. `count≥3 비율` + `동시리셋 패턴`을 함께 본다**(재추출 축 + 기존 캐시 나머지 축 결합 = 추가비용 0).

## 축별 A↔B 리셋 F1

| 축 | pieman (ΔS −0.253) | tunnel (ΔS −0.151) | monkey (ΔS −0.052) | 재설계 조준 |
|---|---|---|---|---|
| protagonist_perspective | 0.800 | 0.756 | 0.848 | 실험필드(protagonist_present)만 |
| space (구 place) | 0.889 | 0.800 | **0.467** | ✅ **null-명시화(§3.3)** |
| time | 0.778 | 0.706 | 0.788 | ✗ frozen(tense 이미 제거) |
| causality | 0.730 | **0.455** | 0.686 | ⚠️ 실험필드(causal_marker)만 + 공통꼬리말 |
| **intentionality (구 goal)** | **0.437** | **0.340** | **0.381** | ✅ **null-명시화(§3.5)+Q5** |

## 판정

1. **codex "pieman=mention 오염" 은 오답(사용자 지적 정확).** REPORT §2-3의 "home in England" carry-forward 오염은 **monkey** 사례. pieman에서 place는 **최청정**(F1 0.889, false-move 0). pieman에서 무너진 축 = **intentionality 0.437** + `protagonist+goal` co-reset 붕괴(11→4).

2. **재설계는 지배적 실패를 조준한다.** intentionality(goal)가 **3편 전부 최악**(0.34~0.44) → 재설계 #1 우선(null-명시화 §3.5 + Q5)이 정확히 그 축. space는 **monkey 공동최악**(0.467) → §3.3이 조준. **⇒ "재설계가 과녁을 빗나간다"는 아니다.**

3. **🔴 잔여 갭 = causality(특히 tunnel 0.455).** tunnel의 2번째 발산 축인데 재설계는 **core event_type/aspect 프롬프트를 안 바꾸고** 실험필드(causal_marker, 다른 신호)만 추가. 유일한 core 대응 = 공통꼬리말 §4.2("interpret what this sentence depicts, do not merely string-match")의 **전역 anti-literal 지시**(REPORT §3: literal-vs-continuity 갭이 5축 전반에 전파 → 전역 꼬리말이 근원을 침). causal_marker는 event_type/aspect 발산을 직접 못 고침. → **tunnel이 Gate C에서 causality로 실패하면 이는 알려진 잔여이지, 라운드를 태워 재튜닝할 대상이 아님**(core causality 변경은 §3.4 "최저 충실도·frozen" 이탈이라 codex 정당화 필요).

## 스모크 범위 (문서마다 다른 축 — codex "space만"은 오답)

pieman에 space 재추출은 무의미(space 0.889=최고). 올바른 범위 = **붕괴한 축을 그 붕괴한 doc에서**:

| doc | 재추출 축 | 검증 | before(dsv4pro→목표 fable5) |
|---|---|---|---|
| monkey | space | over-silence 수리(§3.3) | unspecified 0.963 → 0.824 |
| pieman | intentionality | null-명시화 수리(§3.5) | none-rate 0.760 → 0.594 |

- **각 1문서×1축 = 3×5의 2/15.**
- **지표(F1≠Score 반영):** (1) 해당 축 unspecified/none-rate 하락, (2) **재추출 축을 dsv4pro 캐시 나머지 4축과 합쳐 `count≥3 비율`이 fable5/gold 방향으로 회복하는가**(구조 회복 = Score 예측), (3) 동시리셋 패턴. 추가비용 0.
- **Q5 선결:** pieman intentionality 스모크는 **Q5 최종 정의**를 검증해야 하므로 Q5를 스모크 前에 결정.

## 남은 위험 (관찰만, 지금 미대응)

1. **causality @ tunnel = 0.455**(2번째 나쁜 칸). 재설계는 core event_type/aspect 미변경, causal_marker(동결 count 밖)만 추가 → **사실상 무대응**. tunnel(ΔS −0.151)은 나쁜 축이 causality 0.455 + intentionality 0.340 **둘** → intentionality만 고쳐도 tunnel Gate C 실패 가능. **tunnel 실패 시 2회차 후보(core causality). 지금은 관찰만.**
2. **time = 전 문서 0.71~0.79, frozen 미대응.** 최악은 아니나 좋지도 않음. tunnel 실패 시 2번째 후보.
3. **pieman 기준선 caveat:** fable5 0.634 = 3편 최고 + t\* 선택 문서라 가장 오염 → **구조적으로 Gate C 최난**. Gate C 실패가 pieman에 집중되면 라벨러 능력이 아니라 기준선 문제일 수 있음.
