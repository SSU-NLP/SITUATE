# Qwen3-1.7B LoRA 공동 학습 — 1 epoch, seed 1

VESSL JOB `job-acqx6oddvn3l`, A100 80GB 1장. 학습 및 평가 결과는 아직 미확정이다.
원격 출력: `/shared/hiontop/qwen17b-lora-e1-s1-20260926/run/`.

기준은 최근 동결 Qwen3-1.7B 실험과 논문 결과표용 체크포인트의 실제 실행 설정이다.
논문 수식에 기술된 슬롯 예측 레시피와는 구분한다. 백본 revision은 기존 Qwen 임베딩
manifest에서 읽어 동일하게 고정한다.

- 학습 10,430편 / 검증 1,406편, 기존 작가 단위 분할과 split seed 20260728 유지.
- 5슬롯 × 128차원, 입력 bottleneck 8, head dropout 0.4.
- dynamics per_slot, head_feat delta, pred_space obs, pred_weight 1, pos_weight clip.
- FactoredSlot: Adam lr 3e-4, 문서 배치 48, 무작위 초기화.
- LoRA: q_proj/v_proj, rank 16, alpha 32, dropout 0, Adam lr 2e-5. 본체 가중치는 동결.
- 문장별 마지막 유효 토큰의 최종 hidden state를 FP32 L2 정규화. 채팅 템플릿 없음.
- 1 epoch, seed 1, 조기종료 없음. 종료 시 adapter와 FactoredSlot을 각각 저장.
- 학습 후 사람 주석 6편을 online_nf c=1.0으로 평가하고 문서별 및 6편 평균 SCORE/BOR/LIFT 저장.
- 6편 결과를 학습이나 체크포인트 선택에 사용하지 않는다.

동결 백본 실험과 달라지는 구현 조건:

1. LoRA와 FactoredSlot을 함께 업데이트한다. 미리 저장한 임베딩으로 학습하지 않는다.
2. 관측 예측의 정답은 현재 인코더가 만든 다음 문장 벡터이며 정답 쪽은 stop-gradient한다.
   기존 동결 백본에서는 원래 정답 벡터에 기울기가 없었다. LoRA에서는 이 정책을 명시한다.
3. 기존 FP16 캐시 대신 온라인 FP32 정규화 벡터를 head에 전달한다.
4. 문서 배치의 임베딩을 no-grad로 계산한 뒤 head의 입력 기울기를 얻고,
   문장 microbatch를 재계산해 LoRA로 역전파한다. 문서 상태나 문장 간 BPTT를 끊지 않는다.
   encoder dropout을 0으로 두어 두 계산이 일치하도록 한다.

검증: CPU에서 분할 역전파와 직접 공동 역전파의 모든 파라미터 기울기가 일치했다.
기존 `test_factored_slot_batched.py` 18개 테스트 통과. GPU JOB에서도 초기 adapter가
동결 본체와 같은 출력을 내는지, replay가 일치하는지, LoRA 기울기와 실제 업데이트가
발생하는지 검사한다. GPU 검증 결과와 진행 상태는 각각 `gpu_preflight.json`,
`progress.json`에 저장된다. 최종 결과는 `metrics.json`이다.

## 처리속도 최적화

기존 JOB `job-acqx6oddvn3l`을 종료하고 `job-sju15o0asx7t`로 교체했다.
기존 설정·로그·산출물은 `pre_optimization_backup/`에 보존했다. 기존 JOB은 epoch 완료 전이므로
학습 가중치 저장본이 없었으며, 새 JOB은 같은 seed 1의 초기 상태부터 학습한다.
새 원격 출력은 `/shared/hiontop/qwen17b-lora-e1-s1-20260926/fast-run/`이다.

학습 데이터에서 추출한 동일 1,024문장으로 인코더 캐시+역전파 시간을 비교했다.
32문장/4096토큰·활성값 재계산 on은 10.592초, 128문장/8192토큰·off는 2.751초였다.
선택 설정은 후자이며 측정 최대 메모리는 26.048GB였다. 256문장은 2.752초/48.699GB로
추가 이득이 없었고, 512문장/32768토큰·off는 OOM으로 제외했다.
이는 인코더 구간 벤치마크이며 전체 학습의 속도 향상은 실제 배치 완료 로그로 따로 확인한다.
학습 배치 48문서, 순서, 손실, seed, optimizer와 학습률은 유지했다.
첫 step 및 10 step마다 adapter/head/optimizer/RNG 상태를 `training_latest.pt`에 저장한다.

첫 전체 학습 배치(41,838문장)의 실측 시간은 351.342초 → 84.704초로 약 4.15배 개선됐다.
최대 메모리는 27.187GB, LoRA 기울기 및 가중치 업데이트 검증 통과.
첫 배치 loss는 6.514630 → 6.514614로, 배치 계산에 따른 수치 차이 수준이다.
