# PROJECT_CONTEXT.md

> **이 문서의 용도**
> 이 레포의 코드만 봐서는 "왜 이걸 했고, 무엇을 발견했는지"가 드러나지 않는다.
> 이 문서는 논문 · 발표 슬라이드 · 발표 스크립트에서 추출한 **검증된 사실(ground truth)**이다.
> Claude Code는 이 문서를 먼저 읽고, **코드 및 결과 파일과 대조**한 뒤 README를 작성한다.
> 이 문서에 없는 코드 세부(함수 구현, 인자 처리 등)는 **추측하지 말고 실제 파일을 열어 확인**한다.
> 수치가 코드/결과 파일과 다르면 임의로 고치지 말고 불일치를 보고한다.

---

## 0. 프로젝트 개요 (한 줄 요약)

한국어 혐오/공격 표현 탐지 모델이 **문맥이 아니라 "집단 지칭 단어"에 지름길처럼 반응하는 문제(identity-term shortcut)**를,
**타당한 교체에만 일관성 제약을 거는 validity-gated counterfactual consistency regularization**으로 완화하고,
그 과정에서 **"더 많은 쌍"보다 "검증을 통과한 더 깨끗한 쌍"이 효과적**임을 실험으로 보인 프로젝트.

| 항목 | 내용 |
|---|---|
| 과목 | Korea University COSE461 (자연어처리) Final Project |
| 팀 | Team 11 |
| 저자 | Minseo Shin · Soobin Cho · Nayeon Kim |
| 데이터셋 | K-HATERS (`humane-lab/K-HATERS`) |
| 모델 | KLUE-RoBERTa (`klue/roberta-base`) |
| 원본 레포 계정 | `angellashin` (README에는 `nlp-korean`, 논문 인용은 `261RCOSE46101` — §9 참고) |

---

## 1. 문제 정의 (Problem)

한국어 혐오 표현 탐지는 보통 일반적인 텍스트 분류로 다뤄지지만, **높은 aggregate 정확도가 systematic한 약점을 가린다.**

- **Identity-term shortcut**: 모델이 실제 abusive intent가 아니라 성별·종교·인종·나이·성적지향·장애를 가리키는 토큰 자체를 offensiveness의 강한 단서로 학습한다.
- **Flip 현상**: 문맥은 그대로 두고 집단 단어만 바꾸면 예측이 뒤집힌다.
  - 예: `여가부·여성단체 … 미워할거양` → `offensive` (정답) / 여성단체 → 남성단체로만 교체 → `not offensive` (오답). 같은 문장, 같은 의도인데 label이 flip.
- **중립 언급 오탐**: 단지 집단을 언급하기만 한 문장을 hateful로 오분류.
- **Macro-F1로는 안 드러남**: aggregate 정확도는 높게 유지되는 채로 이런 identity-term 실패가 밑에 숨는다.

**Research Question**: validity-gated counterfactual consistency가 base-task Macro-F1을 유지하면서 한국어 혐오/공격 표현 robustness를 개선하는가?

---

## 2. 핵심 아이디어 / 이론 (Method idea)

### 2.1 Counterfactual Consistency Regularization (CCR)
두 문장이 **집단 단어만 다르고 label이 유지되어야 한다면**, 모델은 두 문장에 대해 일관된 예측을 해야 한다.
원본 예측 분포 `p_o`와 counterfactual 예측 분포 `p_c`(2-class softmax) 사이의 **대칭 KL 발산**을 일관성 항으로 사용:

```
Cons(p_o, p_c) = ½ · ( KL(p_o ‖ p_c) + KL(p_c ‖ p_o) )
L = L_cls + λ · Cons(p_o, p_c)
```

- `L_cls`: cross-entropy 분류 손실
- `λ`: nominal regularization weight
- 일관성 항은 **해당 조건에서 선택된 counterfactual 쌍에만** 적용된다.

### 2.2 Validity Gate (핵심 기여)
문제는 **모든 identity 교체가 label-preserving하지 않다는 것.** 특히 한국어는 형태소·조사·슬랭·맥락 의미가 얽혀서 교체가 의미를 바꾸기 쉽다.
예) `트랜스젠더 → 이성애자`인데 문장에 `자연임신` 맥락이 있으면 의미가 깨진다. `여성인권 → 남성인권`을 역사적 불평등 문장에 넣으면 사회적 의미가 반전된다.
→ 이런 쌍으로 학습하면 모델에게 **잘못된 invariance**를 가르친다.
따라서 counterfactual **생성(generation)**과 **타당성 검증(validity assessment)**을 분리하고, **게이트를 통과한 쌍에만** 일관성 제약을 건다.

### 2.3 Coverage-Aware Regularization
일관성 항이 valid counterfactual이 있는 예시에만 적용되므로, 실제 regularization 압력은 λ뿐 아니라 **coverage** `c = #{valid CF 학습 예시} / N_train`에도 의존한다.
- 유효 강도: `λ_eff = λ × c`
- **Strict-Matched**: strict gate는 유지하되 nominal λ를 키워 `λ_matched × c_strict ≈ λ_base × c_naive`가 되게 맞춤. → Naive Swap과 **비슷한 aggregate 강도**에서 비교하므로, 남는 차이는 "게이트가 어떤 쌍을 골랐는가(pair quality)"에 더 직접 귀속된다.

---

## 3. 방법 파이프라인 (Pipeline)

**생성 → 검증 게이트 → 일관성 정규화** 3단계. 생성은 후보를 만들 뿐, 그 쌍이 학습에 안전한지는 결정하지 않는다.

### 3.1 Counterfactual Pair Generation
- 6개 identity 범주 lexicon: **gender / religion / ethnicity / age / sexuality / disability**
- **Kiwi** 형태소 분석으로 identity term 탐지 (한국어 token boundary 존중)
- token-aware 치환 + **조사(josa) 자동 교정** (받침 조건이 바뀌면 post-positional particle 조정)
- 산출물: 후보 쌍 `(x, x_cf)` — 이 단계는 모든 조건이 공유. Naive Swap은 후보를 그대로 쓰고, Strict-Gated는 여기서 §3.2 필터를 적용.

### 3.2 Validity Gate — **7개 strict conditions**
> ⚠️ 정확도 주의(§8 참고): 논문 본문·그림 캡션은 일관되게 **"7 strict conditions"**로 기술한다.

| # | 조건 | 막아내는 실패 모드 (예시) |
|---|---|---|
| 1 | Semantic blacklist | 교체 시 사실·생물학적 비대칭을 만드는 맥락 (gender swap의 `임신`, religion swap의 `지하드`) |
| 2 | Asymmetric-pair exclusion | 사회적으로 label 보존이 성립 안 되는 방향 (`트랜스젠더 ↔ 이성애자`) |
| 3 | Comparison / relation filter | 이미 두 집단을 비교하는 문장 (`보다`, `반면` 등 대조 표현) |
| 4 | Harmful-object filter | identity term이 사건 키워드(`폭행`, `강간`)와 목적어로 함께 등장 → 다른 사건 함의 |
| 5 | Age-decade filter | 명시적 연령대 표현(`60대`)이 youth term과 교체되면 의미 모순 |
| 6 | Grammar correctness (`valid_grammar`) | token-aware 치환 후 조사 결합이 ill-formed면 폐기 |
| 7 | Same-category constraint (`same_category`) | 원본·치환 term이 같은 범주여야 함. **by-construction으로 만족되어 binding filter로 작동하진 않음** (완전성 위해 기록) |

- 결과: **7,735 후보 → 5,964 통과 (약 77% retained)**
- 통과 쌍은 CCR 학습 신호로 사용, 탈락 쌍은 CCR에서 제외(Strict-Gated는 원본에 `L_cls`만 적용).

### 3.3 CCR Objective
§2.1의 대칭 KL 손실을 그대로 사용.

### 3.4 Coverage matching
§2.3의 Strict-Matched 정의.

---

## 4. 실험 셋업 (Experimental setup)

### 4.1 데이터
- K-HATERS repository split: **172,157 train / 10,000 val / 10,000 test**
- Label 매핑 (binary): `offensive`, `l1_hate`, `l2_hate` → **1 (positive)** / `clean`, `exclude` → **0 (negative)**
- 학습 split의 swappable 후보: **7,735**, 그중 strict gate 통과: **5,964**
- 테스트 쌍: robustness 평가용 **455쌍**, 그중 strict-valid subset **350쌍**

### 4.2 학습 조건 (5개 + λ ablation)
| 조건 | 학습 신호 |
|---|---|
| Baseline | cross-entropy만 (λ=0) |
| Masking Cons Reg | identity term을 `[MASK]`로 가리고 동일 penalty (semantic 치환 없음, sanity 비교용) |
| Naive Swap | 생성된 **모든** identity swap에 CCR (7,735쌍, λ=0.1) |
| Strict-Gated | **strict-valid** swap에만 CCR (5,964쌍, λ=0.1) |
| Strict-Matched | strict-valid swap + coverage-matched λ (5,964쌍, **λ=0.1297**) |

### 4.3 평가 지표 (Metrics)
| 지표 | 역할 | 계산 대상 |
|---|---|---|
| **Macro-F1** | base-task guardrail (성능이 희생되지 않았는지) | test set |
| **PairAcc** | 원본·counterfactual 둘 다 정답일 확률 | **455 robustness 쌍** |
| **StrictPairAcc** | 주 robustness 지표 (strict-valid 쌍 한정 PairAcc) | **350 strict-valid 쌍** |
| **FlipRate** | 교체 시 예측이 바뀌는 비율 (보조 진단, 낮을수록 좋음) | 쌍 집합 |
| **ProbGap** | 원본·counterfactual 간 confidence 변화 (낮을수록 안정) | 쌍 집합 |

- **분해**: `PairAcc = Origcorrect × consistency` — Origcorrect는 원본이 맞을 확률, consistency는 그 맞은 원본의 counterfactual도 맞을 확률.

### 4.4 하이퍼파라미터 / 실행
- `klue/roberta-base`, CLS 토큰 위 단일 linear classifier
- AdamW, lr `3e-5`, batch `64`, max seq len `128`, weight decay `0.01`, epochs `3`
- seeds: **42, 123, 456** (결과는 seed 평균, seed별 best checkpoint는 validation Macro-F1로 선택)
- λ ablation (Strict-Gated): λ ∈ {0.05, 0.10, 0.1297, 0.15, 0.25} (Naive Swap 동일 sweep은 Appendix 대조용)

---

## 5. 핵심 결과 (Results)

### 5.1 메인 비교 (Table 2, 3 seeds 평균)
FlipRate·ProbGap은 낮을수록, 나머지는 높을수록 좋음.

| Method | Macro-F1 | FlipRate | ProbGap | PairAcc | StrictPairAcc |
|---|---|---|---|---|---|
| Baseline | 0.7882 | 0.0549 | 0.0440 | 0.8029 | 0.8076 |
| Masking Cons Reg | 0.7901 | 0.0491 | 0.0435 | 0.8110 | 0.8133 |
| Naive Swap | 0.7916 | 0.0212 | 0.0168 | 0.8168 | 0.8171 |
| Strict-Gated | 0.7907 | 0.0234 | 0.0198 | 0.8220 | 0.8248 |
| **Strict-Matched** | **0.7906** | **0.0205** | 0.0176 | **0.8264** | **0.8295** |

- **핵심 결과**: Strict-Matched가 PairAcc·StrictPairAcc 모두 최고. Naive Swap 대비 각각 **+0.0096 / +0.0124** 개선.
- 이걸 **Naive Swap보다 23% 적은 쌍(5,964 vs 7,735)**으로 달성 → "cleaner beats more."
- **Macro-F1은 전 조건에서 0.788~0.792 좁은 범위 유지** → robustness 개선이 분류 성능을 희생시키지 않음.
- Masking Cons Reg는 PairAcc를 조금만 올리고 FlipRate는 baseline 근처에 머묾 → **semantic하게 grounded된 대조 쌍 구조가 중요**하고, 일관성 penalty 자체만으로는 부족함을 시사.

### 5.2 Coverage & 유효 강도 (Table 1)
| Condition | Pairs | c | λ | λ_eff |
|---|---|---|---|---|
| Naive Swap | 7,735 | 0.04493 | 0.100 | 0.00449 |
| Strict-Gated | 5,964 | 0.03464 | 0.100 | 0.00346 |
| Strict-Matched | 5,964 | 0.03464 | 0.1297 | 0.00449 |

→ Strict-Matched는 낮은 coverage를 보정해 λ_eff를 Naive Swap과 맞춤 → 비교 초점이 "총 정규화 양"이 아니라 "쌍 품질"에 놓임.

### 5.3 λ 민감도 (Table 3, Strict-Gated family)
| λ | Macro-F1 | PairAcc | StrictPairAcc | ProbGap |
|---|---|---|---|---|
| 0.05 | 0.7917 | 0.8293 | 0.8352 | 0.0231 |
| 0.10 | 0.7907 | 0.8220 | 0.8248 | 0.0198 |
| 0.1297† | 0.7906 | 0.8264 | 0.8295 | 0.0176 |
| 0.15 | 0.7917 | 0.8249 | 0.8305 | 0.0168 |
| 0.25 | 0.7898 | 0.8300 | **0.8381** | 0.0163 |

`†` = Strict-Matched.
- **ProbGap은 λ 증가에 따라 단조 감소** (0.0231 → 0.0163): 강한 정규화가 confidence를 꾸준히 안정화.
- **StrictPairAcc는 비단조**: λ=0.10에서 dip 후 λ=0.25에서 최고 0.8381. → calibration 안정성과 correctness-aware robustness는 관련 있지만 동일하지 않음.

### 5.4 Control check (Table 7)
동일 λ grid에서 Strict-Gated가 Naive Swap을 **5개 λ 전부에서** StrictPairAcc로 앞섬 (Δ ∈ [+0.0009, +0.0123]).
→ 게이트의 이점은 hyperparameter tuning 아티팩트가 아니라, invalid 쌍 필터링 자체의 효과.

### 5.5 PairAcc 분해 분석 (§Analysis)
- 두 CCR 변형 모두 PairAcc 개선의 대부분이 **consistency**에서 옴 (Naive +0.0332 / Strict-Matched +0.0342).
- 단, Strict-Matched가 Naive Swap을 넘어서는 추가 이점(Δ=0.0095)은 **거의 전부 Origcorrect 성분**(+0.0086, ≈90%)에서 옴.
- 해석: strict gate는 pair-level 일관성을 더 조인다기보다, **per-example 정확도를 깎아먹는 쌍을 제거**하는 방식으로 도움을 준다.

### 5.6 정성 분석 (Table 4, seed 42)
- 게이트가 **455쌍 중 105쌍(23.1%) 거절**.
- 남은 **350 strict-valid 쌍**에서 Baseline은 **19쌍(5.4%)** flip → Strict-Matched 학습 후 **7쌍(2.0%)**로 감소.
- 세 가지 case:
  - **Gate accept**: `여성단체→남성단체` — 같은 abusive intent인데 Baseline이 flip. 게이트 통과 → CCR이 flip 교정.
  - **Gate reject**: `트랜스젠더→이성애자` + `자연임신` 맥락 — 교체가 의미를 바꿈. Naive Swap은 유지하지만 게이트는 거절.
  - **Residual failure**: 결정 경계 근처 (원본 p=0.493 정답 → 교체 후 p=0.576로 경계 넘음). Strict-Matched도 margin에서는 여전히 flip.

---

## 6. 결론 (3 takeaways)
1. **Identity 교체는 정규화에 쓰기 전 label 보존 여부를 검증해야 한다.** 한국어에서 naive 치환은 invalid 학습 신호를 만든다.
2. **깨끗한 쌍 선택이 coverage 최대화보다 유용할 수 있다.** Strict-Matched가 23% 적은 쌍으로 Naive Swap을 앞섰고, 그 이점은 주로 per-example 정확도를 해치는 쌍을 제거한 데서 옴.
3. **correctness가 중요할 때 PairAcc가 FlipRate보다 유익한 진단이다.** FlipRate는 "일관되게 틀리는" 예측을 못 잡지만 PairAcc는 원본·counterfactual 둘 다 정답을 요구한다.

---

## 7. 한계 & Future Work
- **Rule-based 게이트**라 미묘한 pragmatic validity 실패를 놓칠 수 있음.
- **테스트 쌍도 동일 rule-based 시스템으로 생성** → OOD 치환에 대한 robustness를 과대평가할 여지.
- **단일 데이터셋 · 단일 encoder family**, seed 3개 → 1 std 이내 차이는 신중히 해석.
- λ_eff 균등화로도 23% coverage gap이 완전히 해소되진 않음 (gate composition이 부분적 confound).
- 방향: **learned validity gating** (NLI 기반 label-preservation 검사), 더 크거나 multilingual/generative 모델, **confidence-weighted consistency** (경계 근처 예시에 penalty를 certainty로 가중).

---

## 8. ⚠️ 정확히 지켜야 할 사실 (README 작성 시 틀리기 쉬운 지점)

이 프로젝트 자료들(논문/슬라이드/스크립트) 사이에 표기 차이가 있다. README는 **논문 기준**으로 통일한다.

1. **Validity gate 조건 수 = 7 (논문 기준).** 발표 슬라이드의 표는 6줄인데, 이는 harmful-object(incident) 필터와 age-decade 필터를 `Incident / age filters` 한 줄로 합쳤기 때문이다. `same_category`(조건 7)는 by-construction으로 만족되어 실질 binding filter는 아니므로, "실질 필터 6개"라는 서술도 가능하지만 **총 조건 수는 7로 적는다.** 6이라고만 쓰면 논문과 어긋난다.
2. **`5.4% → 2.0%`는 seed 42, 350 strict-valid 쌍에서의 flip 개수(19→7)다.** 메인 표의 FlipRate(3-seed 평균, 455쌍 기준)는 `0.0549 → 0.0205`(≈5.5%→2.1%)로 별개 수치다. 둘을 뭉뚱그리지 말 것.
3. **PairAcc는 455 robustness 쌍, StrictPairAcc는 350 strict-valid 쌍**에서 계산된다. 대상 집합을 혼동하지 말 것.
4. **6개 identity 범주**: gender / religion / ethnicity / age / sexuality / disability.
5. **77% retained = 5,964 / 7,735** (약값). 정확 비율이 필요하면 5,964/7,735로 계산.
6. **원본 레포는 `angellashin` 계정.** README 상 clone 주소는 `nlp-korean`, 논문 인용은 `261RCOSE46101` — §9 참고.

---

## 9. 레포 / 재현 메모 (Claude Code 대조용)

- 논문에 명시된 실행 설정: `klue/roberta-base`, batch 64, max len 128, epochs 3, lr 3e-5, seeds 42/123/456. **README/코드가 이와 다르면 불일치를 보고할 것.**
- 레포 이름이 두 가지로 참조됨(`nlp-korean` vs `261RCOSE46101`). fork 후 remote/README 주소를 일치시킬지 판단 필요.
- `results/`에 report-grade JSON·요약·최종 표가 있고, 생성물(checkpoint, log, 임시 JSON, CSV, 생성된 CF 쌍)은 gitignore 대상 — 공유용 최종 JSON은 `results/raw/`로 복사하는 규칙.
- 팀 기여(Appendix A.1): Minseo Shin(학습 파이프라인·main 실험: baseline/naive/strict-gated), Soobin Cho(프로젝트 방향·method 설계·CF 쌍 구성·validity-gated 학습), Nayeon Kim(평가·결과 분석·error-case 검사·figure/table·Naive Swap λ sensitivity·Results/Analysis/Conclusion 작성). — README에 개인 기여를 넣을지는 선택.
