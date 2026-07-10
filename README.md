# Validity-Gated Counterfactual Consistency Regularization for Korean Hate/Offensive Language Detection

> Korea University COSE461 (자연어처리) Final Project
> 데이터: [K-HATERS](https://huggingface.co/datasets/humane-lab/K-HATERS) · 모델: [`klue/roberta-base`](https://huggingface.co/klue/roberta-base)

**한 줄 요약: 한국어 혐오/공격 표현 탐지 모델이 문맥이 아니라 집단 지칭 단어 자체에 반응하는 shortcut 문제를, "타당성이 검증된" counterfactual 쌍에만 일관성 제약을 거는 방식(validity-gated CCR)으로 완화했습니다. 핵심 발견 — identity 치환은 label-preserving 여부를 검증한 뒤에만 정규화 신호로 써야 robustness가 개선됩니다. 동일한 λ에서 gate만 추가해도 robustness가 개선되고(Strict-Gated), coverage 감소를 보정한 Strict-Matched는 23% 적은 쌍(5,964개)만으로 비교한 5개 학습 조건 중 최고 robustness(StrictPairAcc 0.8295, FlipRate 0.0205)를 Macro-F1 손실 없이 달성했습니다.**

---

## 1. 문제 정의 — Identity-Term Shortcut

한국어 혐오 표현 탐지는 보통 일반 텍스트 분류로 다뤄지지만, **높은 aggregate 성능이 systematic한 약점을 가립니다.** K-HATERS에서 실제로 관찰된 flip 사례:

| 입력 (동일 문맥, gold label = offensive) | Baseline 예측 |
|---|---|
| `여가부 · 여성단체. 왈왈왈 #@이름# 미워할거양` | offensive ✅ |
| `여가부 · `**`남성단체`**`. 왈왈왈 #@이름# 미워할거양` — 집단 단어만 교체 | not offensive ❌ |

같은 문장, 같은 의도인데 예측이 뒤집힙니다. 이것이 의미하는 것:

- **Identity-term shortcut** — 모델이 실제 abusive intent가 아니라 성별·종교·인종·나이·성적지향·장애를 가리키는 토큰 자체를 offensiveness의 강한 단서로 학습
- **중립 언급 오탐** — 단지 집단을 언급하기만 한 문장을 hateful로 오분류
- **Macro-F1로는 안 드러남** — aggregate 정확도는 높게 유지된 채로 이런 실패가 밑에 숨음

> **Research Question**: validity-gated counterfactual consistency가 base-task Macro-F1을 유지하면서 한국어 혐오/공격 표현 robustness를 개선하는가?

> **가설**: counterfactual 쌍은 "많이" 쓰는 것보다 "교체 후에도 의미와 label이 보존되는 쌍만 골라" 쓰는 것이 중요하다. 정규화 강도가 같다면, 검증된 쌍만 쓴 모델이 더 robust할 것이다.

이 가설이 맞다면, 같은 λ에서 쌍을 23% 걸러내기만 해도 robustness가 더 좋아야 합니다. 아래 검증 설계(§4)가 정확히 이것을 확인하도록 짜여 있습니다.

---

## 2. Method

**생성 → validity gate → CCR** 3단계입니다. 생성은 후보를 만들 뿐이고, 학습에 안전한지는 게이트가 결정합니다.

![Pipeline Overview](assets/pipeline_overview.png)

*파이프라인 개요: 공통 생성 단계에서 만든 candidate pair를 Naive Swap(비교군)은 전부 사용하고, Strict-Gated branch(제안 방법)는 7가지 strict 조건의 validity gate를 통과한 쌍만 사용합니다. 이후 shared KLUE-RoBERTa에서 일관성 목적함수로 학습하며, Strict-Matched는 gate로 인한 coverage 감소를 보정하기 위해 λ를 상향한 조건입니다 (보정의 성격은 §2.3 참조).*

### 2.1 Counterfactual Pair 생성 (`dataset.py`)

- **Identity lexicon**: 6개 범주 — gender / religion / ethnicity / age / sexuality / disability (`SWAP_PAIRS_BY_CAT`, 31개 swap terms)
- **Kiwi 형태소 분석**으로 identity term 탐지 — term이 독립 토큰으로 등장해야 하고, 서로 다른 identity term이 2개 이상이면 제외 (`find_swap`)
- **Token-aware 치환 + 조사(josa) 자동 교정** — 받침 조건이 바뀌면 post-positional particle 조정 (`make_swap`, `_adjust_josa`)

### 2.2 Validity Gate — 왜 필요한가 (핵심 기여)

문제는 **모든 identity 교체가 label-preserving하지 않다**는 것입니다. 한국어는 조사·형태소·슬랭·맥락 의미가 얽혀 있어 단어 하나를 바꾸면 문장의 의미 자체가 깨질 수 있습니다.

![Validity Gate: Which Counterfactuals Become Training Signal?](assets/validity_gate_cases.png)

*왼쪽(Case 1): 교체 후에도 의미·label이 유지되는 strict-valid 쌍 → CCR 학습 신호로 사용. 오른쪽(Case 2): `임신` 맥락이 있는 gender swap처럼 교체가 사실적 의미를 바꾸는 쌍 → gate가 거절. Naive Swap은 이런 쌍도 정규화에 사용합니다. Strict-Gated는 거절된 쌍을 일관성 항에서만 제외하며, 원본 문장은 분류 loss로 정상 학습에 참여합니다 — 샘플을 버리는 것이 아니라 정규화 신호에서만 빼는 것입니다 (§2.3).*

나쁜 쌍은 단순 노이즈가 아니라 모델에 **잘못된 invariance**를 적극적으로 가르칩니다. 그래서 **생성(generation)과 타당성 검증(validity assessment)을 분리**하고, 걸러내는 것 자체가 성능이 됩니다.

**7가지 strict 조건**(`compute_validity_strict`)으로 검증합니다. 대표적으로 (1) **semantic blacklist** — gender swap 문장에 `임신` 같은 맥락이 있으면 교체 시 사실적 비대칭이 생겨 거절, (2) **grammar correctness** — 치환 후 조사 결합이 ill-formed면 거절하는 식입니다.

**Gate output**: 학습 split의 swappable 후보 **7,735쌍 → 5,964쌍 통과 (약 77% retained, 23% 거절)**

<details>
<summary>7가지 조건 전체 목록</summary>

| # | 조건 | 막아내는 실패 모드 |
|---|---|---|
| 1 | Semantic blacklist | 교체 시 사실·생물학적 비대칭이 생기는 맥락 (gender swap의 `임신`, religion swap의 `지하드`) |
| 2 | Asymmetric-pair exclusion | 사회적으로 label 보존이 성립하지 않는 방향 (`트랜스젠더 ↔ 이성애자`) |
| 3 | Comparison / relation filter | 이미 두 집단을 비교하는 문장 (`보다`, `반면` 등) |
| 4 | Harmful-object filter | identity term이 사건 키워드(`폭행` 등)와 목적어로 함께 등장 → 다른 사건 함의 |
| 5 | Age-decade filter | 명시적 연령대 표현(`60대`)이 youth term과 교체되면 의미 모순 |
| 6 | Grammar correctness | 치환 후 조사 결합이 ill-formed면 폐기 |
| 7 | Same-category constraint | 원본·치환 term이 같은 범주여야 함 (by-construction 만족, 완전성 위해 기록) |

</details>

### 2.3 CCR Objective

원본 예측 분포 `p_o`와 counterfactual 예측 분포 `p_c` 사이의 **대칭 KL divergence**를 일관성 항으로 사용합니다 (`run_exp.py`의 `sym_kl()`).

```
L = L_cls + λ · ½ [ KL(p_o ‖ p_c) + KL(p_c ‖ p_o) ]
```

일관성 항은 gate를 통과한 쌍에만 적용되며, 원본 logits는 classification forward의 것을 anchor로 재사용합니다 (train-time dropout noise 감소).

**Gate에서 거절된 샘플은 학습에서 제외되지 않습니다.** 해당 샘플의 원본 문장은 `L_cls`로 정상 학습에 참여하며, counterfactual 쌍만 일관성 항 계산에서 빠집니다. 즉 gate는 "데이터 필터"가 아니라 **"정규화 신호 필터"**입니다.

**일관성 항의 실제 집계 방식** (`run_exp.py` 학습 루프): 배치 안에서 valid 쌍만 불리언 마스크로 골라낸 뒤, 그 쌍들에 대한 **평균** sym-KL(`reduction='batchmean'`, 분모 = 해당 배치의 valid 쌍 수)을 계산해 λ 그대로 더합니다. valid 쌍이 하나도 없는 배치에서는 일관성 항이 통째로 꺼집니다. batch 64 기준 strict 계열에서 valid 쌍이 1개 이상인 배치는 약 90%(평균 2.2쌍/배치, 학습 로그 기준)이므로, coverage가 정규화에 작용하는 채널은 "예시 평균의 희석"이 아니라 이 **배치 단위 on/off**입니다.

**Coverage-aware λ control (Strict-Matched)**: gate를 쓰면 valid 쌍 비율 c가 줄어들기 때문에(4.49% → 3.46%), Strict-Gated가 이기더라도 "정규화가 약해져서 좋아진 것 아니냐"는 반론이 가능합니다. 이를 보정하기 위해 **Strict-Matched**는 `λ_eff = λ × c`가 Naive Swap과 같아지도록 λ를 0.1297로 상향합니다. 단, 위 집계 방식 때문에 이 보정은 gradient 수준의 정규화 압력을 정확히 등화하는 것이 아니라 **coverage 감소에 대한 heuristic 보정**입니다 — 배치 on/off 기준으로는 Naive와 Strict 모두 대부분의 배치에서 항이 켜지므로, λ를 키운 Strict-Matched는 오히려 Naive보다 더 강하게 정규화됐을 가능성도 있습니다. 그래서 "같은 정규화 조건에서의 품질 효과"를 가장 깨끗하게 보여주는 비교는 **동일 λ=0.1의 Naive Swap vs Strict-Gated**이고(§5.1), Strict-Matched는 coverage 반론까지 보정한 대표 조건으로 보고합니다.

<details>
<summary>Coverage & λ_eff 수치</summary>

| Condition | Pairs | c | λ | λ_eff |
|---|---|---|---|---|
| Naive Swap | 7,735 | 0.04493 | 0.100 | 0.00449 |
| Strict-Gated | 5,964 | 0.03464 | 0.100 | 0.00346 |
| **Strict-Matched** | **5,964** | **0.03464** | **0.1297** | **0.00449** |

c가 4~5% 수준으로 낮은 것은 전체 학습 데이터 중 identity term이 독립 토큰으로 등장하는 문장 비율 자체가 낮기 때문입니다. λ_eff는 loss 식에서 기계적으로 유도되는 값이 아니라, 로깅과 Strict-Matched의 λ 산정에 쓰기 위해 `effective_lambda = λ × train_valid_cf_ratio`로 **정의된 요약값**입니다.

</details>

---

## 3. Data & Task

| 항목 | 내용 |
|---|---|
| 데이터 | K-HATERS repository split — **172,157 train / 10,000 val / 10,000 test** |
| Label 매핑 | `offensive`, `l1_hate`, `l2_hate` → 1 / `clean`, `exclude` → 0 (binary) |
| 원래 task | K-HATERS 댓글에 대한 binary hate/offensive 탐지 |
| Counterfactual task | valid identity swap 후에도 예측이 정답을 유지하는지 평가 |
| 학습 CF 쌍 | 생성 7,735쌍 → strict-valid 5,964쌍 |
| 평가 CF 쌍 | robustness 평가용 **455쌍**, 그중 strict-valid subset **350쌍** |

---

## 4. 검증 설계

### 학습 조건 5개 — 각 조건이 답하는 질문

모든 조건은 동일한 base 모델 셋업을 사용합니다. 차이는 **어떤 counterfactual 신호를 쓰는가**뿐이며, 각 조건은 가설 검증에 필요한 질문 하나씩을 담당합니다.

| 조건 | 답하려는 질문 | 학습 신호 | 쌍 수 | λ |
|---|---|---|---|---|
| Baseline | 아무것도 안 하면 어디까지인가 (출발점) | cross-entropy만 | — | 0 |
| Masking Cons Reg | 일관성 penalty를 아무렇게나 걸어도 좋아지는가 | 원본 vs identity term을 `[MASK]`로 가린 문장 간 동일 penalty | — | 0.1 |
| Naive Swap | 쌍을 거르지 않고 전부 쓰면 어떻게 되는가 | 생성된 **모든** identity swap에 CCR | 7,735 | 0.1 |
| Strict-Gated | gate로 거르면 어떻게 되는가 | **strict-valid** swap에만 CCR | 5,964 | 0.1 |
| **Strict-Matched** | 거르면서 **coverage 감소까지 보정**하면? | strict-valid swap + coverage-matched λ (heuristic, §2.3) | 5,964 | 0.1297 |

이 중 '쌍의 품질' 효과를 가장 깨끗하게 분리하는 비교는 **동일 λ=0.1의 Naive Swap vs Strict-Gated**입니다 — 다른 조건은 전부 같고 쌍만 걸렀기 때문입니다. **Strict-Matched**는 여기에 "gate로 coverage가 줄어 정규화가 약해진 것 아니냐"는 반론까지 λ_eff 보정으로 막은 대표 조건입니다 (보정의 성격과 한계는 §2.3 참조).

### 평가 지표 — "고집"과 "맞는 고집"

| 지표 | 역할 | 계산 대상 |
|---|---|---|
| **Macro-F1** | base-task guardrail (성능 희생 여부 확인) | test set (10,000) |
| **PairAcc** | 원본·counterfactual **둘 다 정답**일 확률 | 455 robustness 쌍 |
| **StrictPairAcc** | 주 robustness 지표 (strict-valid 쌍 한정 PairAcc) | 350 strict-valid 쌍 |
| **FlipRate** | 교체 시 예측이 바뀌는 비율 (보조 진단, ↓) | 쌍 집합 |
| **ProbGap** | 원본·counterfactual 간 confidence 차이 (↓) | 쌍 집합 |

FlipRate는 "예측이 안 바뀌는가", 즉 **고집**만 재고 정답 여부는 보지 않습니다 — 둘 다 일관되게 틀리는 모델도 FlipRate는 좋아 보입니다. 우리가 원하는 것은 "안 바뀌면서 정답까지 맞는" 모델이므로, **맞는 고집을 재는 PairAcc/StrictPairAcc를 주 지표**로 씁니다.

<details>
<summary>모델/학습 셋업 상세</summary>

- `klue/roberta-base`, CLS 토큰 위 단일 linear classifier
- AdamW, lr `3e-5`, batch `64`, max seq len `128`, weight decay `0.01`, epochs `3`
- Seeds: 42, 123, 456 (결과는 3-seed 평균, seed별 best checkpoint는 validation Macro-F1로 선택)
- λ ablation (Strict-Gated family): λ ∈ {0.05, 0.10, 0.1297, 0.15, 0.25}

</details>

---

## 5. Results & Analysis

### 5.1 메인 비교 — 품질이 총량을 이긴다

3-seed 평균. FlipRate·ProbGap은 낮을수록, 나머지는 높을수록 좋습니다.

| Method | Macro-F1 | FlipRate | ProbGap | PairAcc | StrictPairAcc |
|---|---|---|---|---|---|
| Baseline | 0.7882 | 0.0549 | 0.0440 | 0.8029 | 0.8076 |
| Masking Cons Reg | 0.7901 | 0.0491 | 0.0435 | 0.8110 | 0.8133 |
| Naive Swap | 0.7916 | 0.0212 | 0.0168 | 0.8168 | 0.8171 |
| Strict-Gated | 0.7907 | 0.0234 | 0.0198 | 0.8220 | 0.8248 |
| **Strict-Matched** | 0.7906 | **0.0205** | 0.0176 | **0.8264** | **0.8295** |

**읽는 순서: StrictPairAcc 열부터.** Baseline 0.8076 → Naive Swap 0.8171 → **Strict-Matched 0.8295**. 이 궤적이 결과의 핵심이며, 아래 네 가지가 그 해석입니다.

1. **쌍의 품질이 양보다 중요합니다.** 1차 근거는 동일 λ=0.1 비교입니다: Strict-Gated는 Naive Swap보다 쌍을 23% 덜 쓰고도(5,964 vs 7,735) PairAcc·StrictPairAcc에서 앞섭니다 (0.8220/0.8248 vs 0.8168/0.8171). 같은 λ에서 쌍만 걸렀는데 좋아졌으므로, 이 차이는 gate가 고른 쌍의 품질로 귀속됩니다. coverage 감소까지 보정한 Strict-Matched는 두 지표 모두 최고(0.8264/0.8295)를 기록합니다. 엄밀한 통계적 유의성 검정은 수행하지 않았습니다 (한계 참조). 대신 개선의 방향이 3-seed 평균 기준 단조적이고(0.8076 → 0.8171 → 0.8295), §5.2에서 λ 5개 전 구간에 걸쳐 gate 조건이 Naive Swap을 일관되게 상회한다는 점이 이 결론을 지지합니다.
2. **Naive Swap이 FlipRate·ProbGap에서 앞서는 것은 모순이 아닙니다.** FlipRate는 고집만 재는 지표라, 잘못된 쌍에까지 "예측 바꾸지 마"를 강제하는 Naive Swap이 유리할 수 있습니다. 그러나 정답 유지까지 요구하는 PairAcc/StrictPairAcc에서는 순위가 뒤집힙니다 — 나쁜 쌍이 가르친 invariance가 correctness에는 오히려 해가 된다는 신호입니다.
3. **Macro-F1은 유지됩니다.** 전 조건 0.788~0.792의 좁은 범위 → robustness 개선이 분류 성능을 희생시키지 않습니다.
4. **penalty 자체가 아니라 쌍의 구조가 효과의 원천입니다.** Masking Cons Reg는 PairAcc를 조금만 올리고 FlipRate는 baseline 근처에 머뭅니다. 아무 대조나 일관성을 강제한다고 되는 게 아니라, semantic하게 grounded된 swap 쌍이어야 효과가 납니다.

### 5.2 λ Sweep — 결론이 λ 선택에 의존하지 않는지 확인

이 sweep의 목적은 **최고 λ를 찾는 것이 아니라**, 두 가지를 확인하는 것입니다.

- **결론의 안정성**: Strict-Gated가 5개 λ 전부에서 Naive Swap을 상회합니다 (Δ ∈ [+0.0009, +0.0123]) → gate의 이점은 특정 λ 튜닝의 산물이 아닙니다.
- **두 효과의 분리**: ProbGap은 단조 감소하고 (0.0231 → 0.0163: 정규화가 강할수록 confidence는 안정화), StrictPairAcc는 비단조입니다 (λ=0.10에서 dip 후 λ=0.25에서 sweep 최고치 0.8381) → calibration 안정성과 correctness-aware robustness는 관련 있지만 동일하지 않습니다.

**왜 sweep 최고치(λ=0.25, 0.8381)가 아니라 Strict-Matched(λ=0.1297, 0.8295)를 대표 숫자로 보고하는가.** λ=0.1297은 결과를 보기 전에 규칙(coverage-matching)이 정해준 값이고, λ=0.25는 결과표를 보고 나서 고르는 값이기 때문입니다. 결과를 보고 고른 최고점을 대표로 쓰면 그것은 검증이 아니라 평가 세트에 대한 튜닝이 됩니다.

<details>
<summary>λ sweep 전체 수치</summary>

| λ | Macro-F1 | PairAcc | StrictPairAcc | ProbGap |
|---|---|---|---|---|
| 0.05 | 0.7917 | 0.8293 | 0.8352 | 0.0231 |
| 0.10 | 0.7907 | 0.8220 | 0.8248 | 0.0198 |
| 0.1297† | 0.7906 | 0.8264 | 0.8295 | 0.0176 |
| 0.15 | 0.7917 | 0.8249 | 0.8305 | 0.0168 |
| 0.25 | 0.7898 | 0.8300 | **0.8381** | 0.0163 |

`†` = Strict-Matched (coverage-matched λ)

350쌍 기준으로 쌍 1개가 StrictPairAcc 약 0.003에 해당하므로, λ 간 0.01 수준의 차이는 쌍 3~4개 차이입니다. 개별 λ 간 우열보다는 위의 두 가지 경향(전 구간 Naive 상회, ProbGap 단조/StrictPairAcc 비단조)을 읽는 것이 적절합니다.

</details>

### 5.3 정성 분석 — 무엇이 고쳐지고, 무엇이 남았나

Gate는 테스트 455쌍 중 **105쌍(23.1%)을 거절** → 350 strict-valid 쌍. 350쌍에서 flip은 Baseline **19쌍(5.4%) → Strict-Matched 7쌍(2.0%)**으로 감소. (⚠️ 이 정성 분석 수치는 seed 42 기준 — 메인 표의 3-seed 평균 FlipRate와는 별개의 수치)

| 유형 | 예시 | 결과 |
|---|---|---|
| **Gate accepts** | `여성단체 → 남성단체` (동일 abusive intent) | ✅ CCR에 사용 — Baseline의 flip을 모든 CCR 조건이 교정 |
| **Gate rejects** | `트랜스젠더인데 자연임신?` → `이성애자인데 자연임신?` (`자연임신`이 gender-identity-specific이라 교체 시 의미 변화) | ❌ Gate가 거절 — Naive Swap은 이 쌍을 학습에 사용 |
| **Residual failure** | `#@이름#란 뭐하는 여자지 → 남자지` (결정 경계 근처: p=0.493 → 0.576) | ⚠️ Strict-Matched도 margin에서 flip |

---

## 6. Conclusion

> 한국어 혐오/공격 표현 탐지에서 counterfactual regularization은 **identity 치환이 실제로 label-preserving인지를 반드시 고려해야 합니다.**

1. Identity swap은 정규화 전에 **label 보존 여부를 검증**해야 함 — 나쁜 쌍은 잘못된 invariance를 가르침
2. **gate 조건은 Macro-F1을 유지하면서 robustness를 개선** — 동일 λ에서 쌍만 걸러도 이겼고(Strict-Gated vs Naive Swap), λ 5개 전 구간에서 일관되므로 품질이 양을 이긴다는 가설을 지지함
3. Correctness가 중요할 때는 **FlipRate(고집)보다 PairAcc 계열(맞는 고집)이 더 유용한 지표**

### 한계 & Future Work

- **Rule-based gate**라 미묘한 pragmatic validity 실패를 놓칠 수 있음 → learned validity gating (NLI 기반 label-preservation 검사)
- **테스트 쌍도 동일 rule-based 시스템으로 생성** → OOD 치환에 대한 robustness를 과대평가할 여지
- **StrictPairAcc는 gate가 승인한 쌍에서 계산되므로, gate로 학습한 모델에 유리한 지표라는 우려가 가능** — gate와 무관한 전체 455쌍 PairAcc에서도 Strict-Matched가 최고(0.8264)라는 점이 이 우려를 완화하지만, 독립적으로 구성된 평가 쌍으로의 검증이 남은 과제
- **λ_eff = λ × c 보정은 heuristic** — 일관성 항이 배치 내 valid 쌍의 평균으로 집계되기 때문에, λ_eff를 맞추는 것이 gradient 수준의 정규화 압력을 정확히 등화하지는 않음 (Strict-Matched가 Naive Swap보다 더 강하게 정규화됐을 가능성 존재). 품질 효과의 1차 근거를 동일 λ 비교에 두는 이유이며, per-example 가중 등 정확한 등화 설계가 future work
- **단일 데이터셋·단일 encoder family, seed 3개** → 차이의 방향은 조건·λ 전반에서 일관되지만 크기는 신중히 해석해야 함 (1 std 이내 차이 포함); paired bootstrap 등 엄밀한 검정, 그리고 더 큰/multilingual/generative 모델로의 확장이 future work
- **결정 경계 근처 residual flip** → confidence-weighted consistency objective가 다음 방향

---

## Repository

```
nlp-korean/
├── README.md                        # 이 문서
├── assets/                          # README figure
├── docs/                            # 개발 과정 문서 (제안서·방향 결정·ablation 계획 등)
├── results/                         # report-grade 결과물
│   ├── raw/                         #   최종 실험 JSON
│   ├── tables/                      #   CSV 요약
│   └── logs/                        #   메인 run 학습 로그
├── tests/                           # 유틸리티 스크립트 단위 테스트
└── validity_gated_exp/              # 실험 코드
    ├── run_exp.py                   # 메인 실험 러너 (학습·평가·결과 저장)
    ├── dataset.py                   # K-HATERS 로딩, identity swap 생성, validity gate 구현
    ├── experiment_utils.py          # coverage-matched λ, 결과 병합, error example 수집
    ├── compare_results.py           # 결과 JSON 비교표
    └── RUNNING.md                   # 실행 runbook (환경 → 검증 → 실험 → 비교)
```

<details>
<summary>재현하기 (커맨드 요약)</summary>

전체 절차는 [`validity_gated_exp/RUNNING.md`](validity_gated_exp/RUNNING.md)를 따르세요.

```bash
# 1) 환경 검사
python validity_gated_exp/env_check.py --require_cuda --min_free_gb 15

# 2) 데이터 & CF pair 생성 확인
python validity_gated_exp/check_data.py

# 3) 본 실험
python validity_gated_exp/run_exp.py --exp Baseline "Naive Swap" Strict-Gated Strict-Matched \
  --seeds 42 123 456 --epochs 3 --batch_size 64 \
  --result_path validity_gated_exp/results_core_followup.json

# 4) 결과 비교
python validity_gated_exp/compare_results.py results/raw/results_core_followup.json
```

README의 모든 수치는 `results/raw/`의 최종 JSON과 대조 확인된 값입니다.

</details>
