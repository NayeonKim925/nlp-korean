# Validity-Gated Counterfactual Consistency Regularization for Korean Hate/Offensive Language Detection

> Korea University COSE461 (자연어처리) Final Project
> 데이터: [K-HATERS](https://huggingface.co/datasets/humane-lab/K-HATERS) · 모델: [`klue/roberta-base`](https://huggingface.co/klue/roberta-base)

## Abstract

한국어 혐오·공격 표현 탐지 모델은 문맥이 아니라 집단 지칭 표현 자체에 반응하는 shortcut을 학습하는 경향을 보인다. 본 연구는 이 문제를 완화하기 위해, 타당성이 검증된 counterfactual 쌍에만 일관성 제약을 부여하는 validity-gated counterfactual consistency regularization(CCR)을 제안한다. 핵심 발견은 identity 치환을 label-preserving 여부에 대한 검증 없이 정규화 신호로 사용해서는 안 된다는 것이다. 동일한 λ 조건에서 validity gate를 추가하는 것만으로 robustness가 개선되며(Strict-Gated), gate로 인한 coverage 감소를 보정한 Strict-Matched 조건은 23% 적은 쌍(5,964개)만을 사용하여 비교 대상 5개 학습 조건 중 최고 robustness(StrictPairAcc 0.8295, FlipRate 0.0205)를 Macro-F1의 손실 없이 달성한다.

---

## 1. 문제 정의 — Identity-Term Shortcut

한국어 혐오 표현 탐지는 일반적인 텍스트 분류 문제로 다루어져 왔으나, 높은 aggregate 성능이 체계적인 취약성을 은폐할 수 있다. K-HATERS에서 관찰된 대표적인 flip 사례는 다음과 같다.

| 입력 (동일 문맥, gold label = offensive) | Baseline 예측 |
|---|---|
| `여가부 · 여성단체. 왈왈왈 #@이름# 미워할거양` | offensive (정답) |
| `여가부 · `**`남성단체`**`. 왈왈왈 #@이름# 미워할거양` — 집단 지칭 표현만 교체 | not offensive (오분류) |

동일한 문장 구조와 동일한 발화 의도에도 불구하고 예측이 반전된다. 이는 다음 세 가지를 시사한다.

- **Identity-term shortcut** — 모델이 실제 abusive intent가 아니라 성별·종교·인종·연령·성적지향·장애를 지시하는 토큰 자체를 offensiveness의 강한 단서로 학습한다.
- **중립적 언급에 대한 오탐** — 특정 집단을 단순히 언급하기만 한 문장을 hateful로 오분류한다.
- **Aggregate 지표의 한계** — Macro-F1은 높게 유지되므로 이러한 실패 양상이 지표상 드러나지 않는다.

> **Research Question.** validity-gated counterfactual consistency는 base-task Macro-F1을 유지하면서 한국어 혐오·공격 표현 탐지의 robustness를 개선하는가?

> **가설.** counterfactual 쌍은 사용량보다 타당성이 중요하다. 정규화 강도가 동일하다면, 치환 이후에도 의미와 label이 보존되는 쌍만을 사용한 모델이 더 높은 robustness를 보일 것이다.

이 가설이 성립한다면 동일한 λ에서 쌍의 23%를 배제하는 것만으로 robustness가 향상되어야 한다. §4의 검증 설계는 이 예측을 직접 확인하도록 구성하였다.

---

## 2. Method

본 방법은 **생성(generation) → validity gate → CCR**의 3단계로 구성된다. 생성 단계는 후보 쌍을 산출할 뿐이며, 해당 쌍이 학습 신호로 적합한지는 gate가 판정한다.

![Pipeline Overview](assets/pipeline_overview.png)

*그림 1. 파이프라인 개요. 공통 생성 단계에서 산출된 candidate pair에 대해, 비교군인 Naive Swap은 전량을 사용하고 제안 방법인 Strict-Gated branch는 7가지 strict 조건으로 구성된 validity gate를 통과한 쌍만을 사용한다. 이후 shared KLUE-RoBERTa 상에서 일관성 목적함수로 학습하며, Strict-Matched는 gate로 인한 coverage 감소를 보정하기 위해 λ를 상향한 조건이다(보정의 성격은 §2.3 참조).*

### 2.1 Counterfactual Pair 생성 (`dataset.py`)

- **Identity lexicon** — gender, religion, ethnicity, age, sexuality, disability의 6개 범주로 구성된다 (`SWAP_PAIRS_BY_CAT`, 31개 swap term).
- **Kiwi 형태소 분석 기반 identity term 탐지** — term이 독립 토큰으로 출현해야 하며, 서로 다른 identity term이 2개 이상 포함된 문장은 후보에서 제외한다 (`find_swap`).
- **Token-aware 치환 및 조사 자동 교정** — 받침 조건이 변화하는 경우 post-positional particle을 조정한다 (`make_swap`, `_adjust_josa`).

### 2.2 Validity Gate — 필요성과 설계 (핵심 기여)

모든 identity 치환이 label-preserving한 것은 아니다. 한국어는 조사·형태소·슬랭·맥락 의미가 긴밀히 결합되어 있어, 단일 어휘의 교체가 문장의 의미 자체를 변화시킬 수 있다.

![Validity Gate: Which Counterfactuals Become Training Signal?](assets/validity_gate_cases.png)

*그림 2. 좌측(Case 1)은 치환 이후에도 의미와 label이 유지되는 strict-valid 쌍으로, CCR의 학습 신호로 사용된다. 우측(Case 2)은 `임신` 맥락이 존재하는 gender swap과 같이 치환이 사실적 의미를 변화시키는 쌍으로, gate가 이를 배제한다. Naive Swap은 이러한 쌍 또한 정규화에 사용한다. Strict-Gated는 배제된 쌍을 일관성 항에서만 제외하며, 해당 원본 문장은 분류 loss를 통해 정상적으로 학습에 참여한다. 즉 샘플을 폐기하는 것이 아니라 정규화 신호에서만 배제하는 것이다(§2.3).*

타당하지 않은 쌍은 단순한 noise가 아니라 모델에 **잘못된 invariance를 능동적으로 학습시킨다.** 따라서 본 연구는 **생성과 타당성 검증(validity assessment)을 분리**하며, 타당하지 않은 쌍을 배제하는 절차 자체를 방법론의 핵심 기여로 둔다.

검증은 7가지 strict 조건(`compute_validity_strict`)으로 수행한다. 대표적으로 (1) **semantic blacklist** — gender swap 대상 문장에 `임신` 등의 맥락이 존재하면 치환 시 사실적 비대칭이 발생하므로 배제하고, (2) **grammar correctness** — 치환 이후 조사 결합이 ill-formed인 경우 배제한다.

**Gate output.** 학습 split의 swappable 후보 **7,735쌍 중 5,964쌍이 통과한다 (약 77% retained, 23% 배제).**

<details>
<summary>7가지 조건 전체 목록</summary>

| # | 조건 | 차단하는 실패 유형 |
|---|---|---|
| 1 | Semantic blacklist | 치환 시 사실적·생물학적 비대칭이 발생하는 맥락 (gender swap의 `임신`, religion swap의 `지하드`) |
| 2 | Asymmetric-pair exclusion | 사회적으로 label 보존이 성립하지 않는 치환 방향 (`트랜스젠더 ↔ 이성애자`) |
| 3 | Comparison / relation filter | 두 집단을 이미 비교하고 있는 문장 (`보다`, `반면` 등) |
| 4 | Harmful-object filter | identity term이 사건 키워드(`폭행` 등)와 목적어 관계로 공기하여 다른 사건 함의가 발생하는 경우 |
| 5 | Age-decade filter | 명시적 연령대 표현(`60대`)이 youth term과 교체되어 의미 모순이 발생하는 경우 |
| 6 | Grammar correctness | 치환 이후 조사 결합이 ill-formed인 경우 |
| 7 | Same-category constraint | 원본 term과 치환 term이 동일 범주에 속해야 함 (by-construction 만족하나 완전성을 위해 명시) |

</details>

### 2.3 CCR Objective

원본 예측 분포 `p_o`와 counterfactual 예측 분포 `p_c` 사이의 **대칭 KL divergence**를 일관성 항으로 사용한다 (`run_exp.py`의 `sym_kl()`).

```
L = L_cls + λ · ½ [ KL(p_o ‖ p_c) + KL(p_c ‖ p_o) ]
```

일관성 항은 gate를 통과한 쌍에 한하여 적용하며, 원본 logits는 classification forward에서 산출된 값을 anchor로 재사용하여 train-time dropout noise를 감소시킨다.

**Gate가 배제한 샘플은 학습에서 제외되지 않는다.** 해당 샘플의 원본 문장은 `L_cls`를 통해 정상적으로 학습에 참여하며, counterfactual 쌍만이 일관성 항의 계산에서 제외된다. 즉 gate는 데이터 필터가 아니라 **정규화 신호 필터**로 기능한다.

**일관성 항의 집계 방식** (`run_exp.py` 학습 루프). 배치 내에서 valid 쌍만을 불리언 마스크로 선택한 뒤, 해당 쌍들에 대한 **평균** sym-KL(`reduction='batchmean'`, 분모는 해당 배치의 valid 쌍 수)을 계산하여 λ를 곱하지 않은 채 그대로 더한다. valid 쌍이 존재하지 않는 배치에서는 일관성 항이 비활성화된다. batch size 64 기준으로 strict 계열에서 valid 쌍이 1개 이상 존재하는 배치는 약 90%이며(평균 2.2쌍/배치, 학습 로그 기준), 따라서 coverage가 정규화에 작용하는 경로는 예시 평균의 희석이 아니라 이러한 **배치 단위의 on/off**이다.

**Coverage-aware λ control (Strict-Matched).** gate를 적용하면 valid 쌍 비율 c가 감소하므로(4.49% → 3.46%), Strict-Gated의 우위에 대해 "정규화 강도가 약화되어 발생한 결과가 아닌가"라는 반론이 제기될 수 있다. 이를 보정하기 위해 **Strict-Matched**는 `λ_eff = λ × c`가 Naive Swap과 동일해지도록 λ를 0.1297로 상향한다. 다만 위의 집계 방식으로 인해 이 보정은 gradient 수준의 정규화 압력을 정확히 등화하는 것이 아니라 **coverage 감소에 대한 heuristic 보정**에 해당한다. 배치 on/off 기준으로는 Naive와 Strict 모두 대부분의 배치에서 일관성 항이 활성화되므로, λ를 상향한 Strict-Matched가 오히려 Naive보다 강하게 정규화되었을 가능성을 배제할 수 없다. 따라서 "동일한 정규화 조건에서의 쌍 품질 효과"를 가장 명확하게 분리하는 비교는 **동일 λ=0.1 하의 Naive Swap 대 Strict-Gated**이며(§5.1), Strict-Matched는 coverage 반론까지 보정한 대표 조건으로 보고한다.

<details>
<summary>Coverage 및 λ_eff 수치</summary>

| Condition | Pairs | c | λ | λ_eff |
|---|---|---|---|---|
| Naive Swap | 7,735 | 0.04493 | 0.100 | 0.00449 |
| Strict-Gated | 5,964 | 0.03464 | 0.100 | 0.00346 |
| **Strict-Matched** | **5,964** | **0.03464** | **0.1297** | **0.00449** |

c가 4~5% 수준에 머무는 것은 전체 학습 데이터 중 identity term이 독립 토큰으로 출현하는 문장의 비율 자체가 낮기 때문이다. λ_eff는 loss 식으로부터 기계적으로 유도되는 값이 아니라, 로깅 및 Strict-Matched의 λ 산정을 위해 `effective_lambda = λ × train_valid_cf_ratio`로 **정의된 요약 통계량**이다.

</details>

---

## 3. Data & Task

| 항목 | 내용 |
|---|---|
| 데이터 | K-HATERS repository split — **172,157 train / 10,000 val / 10,000 test** |
| Label 매핑 | `offensive`, `l1_hate`, `l2_hate` → 1 / `clean`, `exclude` → 0 (binary) |
| 원 task | K-HATERS 댓글에 대한 binary hate/offensive 탐지 |
| Counterfactual task | valid identity swap 이후에도 예측이 정답을 유지하는지 평가 |
| 학습 CF 쌍 | 생성 7,735쌍 → strict-valid 5,964쌍 |
| 평가 CF 쌍 | robustness 평가용 **455쌍**, 그중 strict-valid subset **350쌍** |

---

## 4. 실험 설계

### 4.1 학습 조건과 대응 연구 질문

모든 조건은 동일한 base 모델 셋업을 사용하며, 차이는 **어떤 counterfactual 신호를 사용하는가**에 국한된다. 각 조건은 가설 검증에 필요한 하나의 질문에 대응한다.

| 조건 | 대응 질문 | 학습 신호 | 쌍 수 | λ |
|---|---|---|---|---|
| Baseline | 정규화를 적용하지 않을 때의 하한은 어느 수준인가 | cross-entropy만 | — | 0 |
| Masking Cons Reg | 임의의 대조에 일관성 penalty를 부여해도 개선이 발생하는가 | 원본과, identity term을 `[MASK]`로 치환한 문장 간의 동일 penalty | — | 0.1 |
| Naive Swap | 쌍을 선별하지 않고 전량 사용하면 어떤 결과가 나타나는가 | 생성된 **모든** identity swap에 CCR 적용 | 7,735 | 0.1 |
| Strict-Gated | validity gate로 선별하면 어떤 결과가 나타나는가 | **strict-valid** swap에만 CCR 적용 | 5,964 | 0.1 |
| **Strict-Matched** | 선별과 동시에 coverage 감소를 보정하면 어떤 결과가 나타나는가 | strict-valid swap + coverage-matched λ (heuristic, §2.3) | 5,964 | 0.1297 |

이 중 쌍의 품질 효과를 가장 명확하게 분리하는 비교는 **동일 λ=0.1 하의 Naive Swap 대 Strict-Gated**이다. 다른 모든 조건이 동일하고 쌍의 선별 여부만 다르기 때문이다. **Strict-Matched**는 여기에 더해 "gate로 coverage가 감소하여 정규화가 약화된 결과가 아닌가"라는 반론을 λ_eff 보정으로 통제한 대표 조건이다(보정의 성격과 한계는 §2.3 참조).

### 4.2 평가 지표 — 예측 불변성과 정답 보존

| 지표 | 역할 | 계산 대상 |
|---|---|---|
| **Macro-F1** | base-task guardrail (성능 희생 여부 확인) | test set (10,000) |
| **PairAcc** | 원본과 counterfactual 예측이 **모두 정답**일 확률 | 455 robustness 쌍 |
| **StrictPairAcc** | 주 robustness 지표 (strict-valid 쌍에 한정한 PairAcc) | 350 strict-valid 쌍 |
| **FlipRate** | 치환 시 예측이 반전되는 비율 (보조 진단, ↓) | 쌍 집합 |
| **ProbGap** | 원본과 counterfactual 간 confidence 차이 (↓) | 쌍 집합 |

FlipRate는 예측의 **불변성만을 측정하며 정답 여부를 고려하지 않는다.** 원본과 counterfactual 모두에서 일관되게 오분류하는 모델 또한 낮은 FlipRate를 얻을 수 있다. 본 연구가 목표로 하는 것은 정답을 유지한 채 불변인 예측이므로, **correctness-aware 지표인 PairAcc 및 StrictPairAcc를 주 지표로 사용한다.**

<details>
<summary>모델 및 학습 셋업 상세</summary>

- `klue/roberta-base`, CLS 토큰 위 단일 linear classifier
- AdamW, lr `3e-5`, batch `64`, max seq len `128`, weight decay `0.01`, epochs `3`
- Seeds: 42, 123, 456 (결과는 3-seed 평균이며, seed별 best checkpoint는 validation Macro-F1 기준으로 선택)
- λ ablation (Strict-Gated family): λ ∈ {0.05, 0.10, 0.1297, 0.15, 0.25}

</details>

---

## 5. Results & Analysis

### 5.1 주 실험 결과 — 쌍의 타당성이 쌍의 수량보다 결정적이다

3-seed 평균이다. FlipRate와 ProbGap은 낮을수록, 나머지 지표는 높을수록 우수하다.

| Method | Macro-F1 | FlipRate | ProbGap | PairAcc | StrictPairAcc |
|---|---|---|---|---|---|
| Baseline | 0.7882 | 0.0549 | 0.0440 | 0.8029 | 0.8076 |
| Masking Cons Reg | 0.7901 | 0.0491 | 0.0435 | 0.8110 | 0.8133 |
| Naive Swap | 0.7916 | 0.0212 | 0.0168 | 0.8168 | 0.8171 |
| Strict-Gated | 0.7907 | 0.0234 | 0.0198 | 0.8220 | 0.8248 |
| **Strict-Matched** | 0.7906 | **0.0205** | 0.0176 | **0.8264** | **0.8295** |

결과의 핵심은 StrictPairAcc 열에 있다. Baseline 0.8076 → Naive Swap 0.8171 → **Strict-Matched 0.8295**의 궤적이 그것이며, 이에 대한 해석은 다음 네 가지이다.

1. **쌍의 타당성이 수량보다 중요하다.** 1차 근거는 동일 λ=0.1 비교이다. Strict-Gated는 Naive Swap보다 쌍을 23% 적게 사용하면서도(5,964 대 7,735) PairAcc와 StrictPairAcc에서 우위를 보인다(0.8220/0.8248 대 0.8168/0.8171). 동일한 λ 하에서 쌍만을 선별하였음에도 성능이 향상되었으므로, 이 차이는 gate가 선별한 쌍의 품질에 귀속된다. coverage 감소까지 보정한 Strict-Matched는 두 지표 모두에서 최고치(0.8264/0.8295)를 기록한다. 다만 엄밀한 통계적 유의성 검정은 수행하지 않았다(§6 한계 참조). 대신 개선의 방향이 3-seed 평균 기준으로 단조적이며(0.8076 → 0.8171 → 0.8295), §5.2에서 λ 5개 구간 전반에 걸쳐 gate 조건이 Naive Swap을 일관되게 상회한다는 점이 이 결론을 지지한다.
2. **Naive Swap이 FlipRate와 ProbGap에서 우위를 보이는 것은 본 결론과 모순되지 않는다.** FlipRate는 불변성만을 측정하는 지표이므로, 타당하지 않은 쌍에 대해서까지 예측 불변성을 강제하는 Naive Swap이 유리할 수 있다. 그러나 정답 보존까지 요구하는 PairAcc 및 StrictPairAcc에서는 순위가 역전된다. 이는 타당하지 않은 쌍이 학습시킨 invariance가 correctness에는 오히려 해가 됨을 시사한다.
3. **Macro-F1은 유지된다.** 전 조건이 0.788~0.792의 좁은 범위에 분포하며, 이는 robustness 개선이 분류 성능의 희생을 대가로 하지 않음을 의미한다.
4. **효과의 원천은 penalty 자체가 아니라 쌍의 의미적 구조이다.** Masking Cons Reg는 PairAcc를 소폭 향상시키는 데 그치며 FlipRate는 baseline 수준에 머문다. 임의의 대조에 일관성을 강제하는 것만으로는 충분하지 않으며, semantic하게 grounded된 swap 쌍이어야 효과가 발생한다.

### 5.2 λ 민감도 분석 — 결론의 λ 의존성 검증

본 sweep의 목적은 **최적 λ의 탐색이 아니라** 다음 두 가지의 확인에 있다.

- **결론의 안정성** — Strict-Gated는 5개 λ 전부에서 Naive Swap을 상회한다(Δ ∈ [+0.0009, +0.0123]). 따라서 gate의 이점은 특정 λ 튜닝의 산물이 아니다.
- **두 효과의 분리** — ProbGap은 단조 감소하는 반면(0.0231 → 0.0163, 정규화가 강할수록 confidence가 안정화됨), StrictPairAcc는 비단조적이다(λ=0.10에서 dip 후 λ=0.25에서 sweep 최고치 0.8381). 이는 calibration 안정성과 correctness-aware robustness가 상관관계를 가지되 동일한 속성은 아님을 보여준다.

**sweep 최고치(λ=0.25, 0.8381)가 아니라 Strict-Matched(λ=0.1297, 0.8295)를 대표값으로 보고하는 이유.** λ=0.1297은 결과를 관찰하기 이전에 규칙(coverage-matching)에 의해 결정된 값인 반면, λ=0.25는 결과표를 관찰한 이후에 선택되는 값이다. 결과를 보고 선택한 최고점을 대표값으로 보고하는 것은 검증이 아니라 평가 세트에 대한 튜닝에 해당한다.

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

350쌍 기준으로 쌍 1개는 StrictPairAcc 약 0.003에 해당하므로, λ 간 0.01 수준의 차이는 쌍 3~4개의 차이에 불과하다. 따라서 개별 λ 간의 우열보다는 위에서 제시한 두 가지 경향(전 구간에서의 Naive 상회, ProbGap의 단조성과 StrictPairAcc의 비단조성)을 해석하는 것이 적절하다.

</details>

### 5.3 정성 분석 — 교정된 오류와 잔존 오류

Gate는 테스트 455쌍 중 **105쌍(23.1%)을 배제**하여 350개의 strict-valid 쌍을 남긴다. 이 350쌍에서 flip은 Baseline **19쌍(5.4%)에서 Strict-Matched 7쌍(2.0%)으로 감소**한다. (주: 본 정성 분석 수치는 seed 42 기준이며, 주 실험 표의 3-seed 평균 FlipRate와는 별개의 값이다.)

| 유형 | 예시 | 결과 |
|---|---|---|
| **Gate accepts** | `여성단체 → 남성단체` (동일한 abusive intent 유지) | CCR에 사용됨 — Baseline의 flip을 모든 CCR 조건이 교정 |
| **Gate rejects** | `트랜스젠더인데 자연임신?` → `이성애자인데 자연임신?` (`자연임신`이 gender-identity-specific이므로 치환 시 의미 변화 발생) | Gate가 배제 — Naive Swap은 이 쌍을 학습에 사용 |
| **Residual failure** | `#@이름#란 뭐하는 여자지 → 남자지` (결정 경계 근방: p=0.493 → 0.576) | Strict-Matched에서도 margin 영역에서 flip 발생 |

---

## 6. Conclusion

> 한국어 혐오·공격 표현 탐지에서 counterfactual regularization은 **identity 치환이 실제로 label-preserving한지를 반드시 고려해야 한다.**

1. Identity swap은 정규화에 사용하기 전에 **label 보존 여부를 검증해야 한다.** 타당하지 않은 쌍은 잘못된 invariance를 학습시킨다.
2. **gate 조건은 Macro-F1을 유지하면서 robustness를 개선한다.** 동일한 λ 하에서 쌍만을 선별해도 우위를 보였으며(Strict-Gated 대 Naive Swap), 이 경향이 λ 5개 구간 전반에서 일관되게 나타난다는 점이 품질이 수량에 우선한다는 가설을 지지한다.
3. Correctness가 중요한 설정에서는 **불변성만을 측정하는 FlipRate보다, 정답 보존을 함께 요구하는 PairAcc 계열 지표가 더 적절하다.**

### 한계 및 향후 연구

- **Rule-based gate**이므로 미묘한 pragmatic validity 실패를 포착하지 못할 수 있다. NLI 기반 label-preservation 검사를 활용한 learned validity gating이 후속 방향이다.
- **테스트 쌍 또한 동일한 rule-based 시스템으로 생성**되었으므로, OOD 치환에 대한 robustness를 과대평가할 여지가 있다.
- **StrictPairAcc는 gate가 승인한 쌍에서 계산되므로, gate로 학습한 모델에 유리한 지표라는 우려가 제기될 수 있다.** gate와 무관하게 구성된 전체 455쌍 PairAcc에서도 Strict-Matched가 최고치(0.8264)를 기록한다는 점이 이 우려를 완화하나, 독립적으로 구성된 평가 쌍을 통한 검증이 과제로 남는다.
- **λ_eff = λ × c 보정은 heuristic이다.** 일관성 항이 배치 내 valid 쌍의 평균으로 집계되므로 λ_eff를 등화하는 것이 gradient 수준의 정규화 압력을 정확히 등화하지는 않으며, Strict-Matched가 Naive Swap보다 강하게 정규화되었을 가능성이 존재한다. 이것이 품질 효과의 1차 근거를 동일 λ 비교에 두는 이유이며, per-example 가중 등 정확한 등화 설계는 향후 연구 과제이다.
- **단일 데이터셋과 단일 encoder family, 3개 seed**에 기반한 결과이다. 차이의 방향은 조건과 λ 전반에서 일관되나 그 크기는 신중하게 해석되어야 한다(1 std 이내의 차이를 포함). paired bootstrap 등의 엄밀한 유의성 검정, 그리고 더 큰 규모의 multilingual/generative 모델로의 확장이 향후 과제이다.
- **결정 경계 근방의 residual flip**이 남아 있으며, confidence-weighted consistency objective가 다음 방향이다.

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
<summary>재현 절차 (커맨드 요약)</summary>

전체 절차는 [`validity_gated_exp/RUNNING.md`](validity_gated_exp/RUNNING.md)를 참조하라.

```bash
# 1) 환경 검사
python validity_gated_exp/env_check.py --require_cuda --min_free_gb 15

# 2) 데이터 및 CF pair 생성 확인
python validity_gated_exp/check_data.py

# 3) 본 실험
python validity_gated_exp/run_exp.py --exp Baseline "Naive Swap" Strict-Gated Strict-Matched \
  --seeds 42 123 456 --epochs 3 --batch_size 64 \
  --result_path validity_gated_exp/results_core_followup.json

# 4) 결과 비교
python validity_gated_exp/compare_results.py results/raw/results_core_followup.json
```

README에 보고된 모든 수치는 `results/raw/`의 최종 JSON과 대조하여 확인한 값이다.

</details>
