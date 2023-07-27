본 포스팅은 제가 읽었던 논문을 간단하게 정리하는 글입니다. 논문의 모든 내용을 작성하는 것이 아닌, 일부분만 담겨 있으므로 자세한 내용은 [원본 논문](https://paperswithcode.com/paper/why-do-tree-based-models-still-outperform)을 확인해 주시기를 바랍니다. 또한, 논문을 잘못 이해한 부분이 있을 수 있으므로, 양해 바랍니다.

---

![](https://velog.velcdn.com/images/kyyle/post/a39780e5-cbec-4b08-9833-f37a5d63f3f3/image.png)


# Abstract 

이번 논문을 통해 다양한 데이터셋에 대한 standard / novel 딥러닝 모델과 XGBoost, Random Forest와 같은 트리 기반 모델의 벤치마크를 제공함. 

실험 결과, 트리 기반 모델의  우수한 속도를 고려하지 않고도, 트리 기반 모델은 ~10K samples 정도의 사이즈를 가진 데이터셋에서 가장 좋은 결과를 보여주었음.  

이러한 차이를 설명할 수 있도록, 트리 기반 모델과 NN의 귀납적 편향의 차이를 조사하였음. 

* Inductive bias란 모델이 학습하지 않은 데이터에 대해 추론할 때 참고하는 어떠한 가정/편향

조사를 통해, 정형 데이터를 위한 NN은 다음과 같은 문제를 고려해야 함을 알 수 있음

1. uninformative 특성에 강건해야 함
2. 데이터의 방향을 유지해야 함
3. 불규칙한 함수를 쉽게 학습할 수 있어야 함 

# Introduction

딥러닝은 이미지, 텍스트, 오디오 등 다양한 데이터에서 많은 발전을 이루었지만, 정형 데이터에서는 딥러닝보다 XGBoost 같은 결정 트리 기반의 앙상블 모델이 주로 사용되고 있음. 

딥러닝 아키텍처는 데이터의 공간 의존성과 불변성에 맞는 귀납적 편향을 생성하도록 설계되었음. .
이질적인 특성, 작은 샘플 크기, 극단적인 값을 가지는 정형 데이터에서는 이러한 불변성을 찾기가 어려움. 

트리 기반 모델은 미분할 수 없으므로, 딥러닝 블록으로 구성되거나 같이 학습되기 어려움. 많은 정형 데이터를 위한 딥러닝 논문에서는 해당 딥러닝 모델이 트리 기반 모델보다 성능이 우수하다고 주장하지만, 실험 결과 논문에서 언급되지 않은 새로운 데이터셋에서의 성능은 좋지 않았음.  

정형 데이터를 위한 확립된 벤치마크는 존재하지 않고, 인터넷에서 구할 수 있는 정형 데이터는 ImageNet과 같은 벤치마크에 비해 규모가 작아 평가에 있어 더 많은 노이즈가 발생함. 이런 문제에 추가로, 불균등한 하이퍼파라미터 튜닝 노력과 벤치마크의 통계적 불확실성을 고려하지 않는 것과 같은 문제 또한 존재함.

이런 문제를 완화할 수 있도록, 데이터셋 및 하이퍼파라미터 튜닝을 위한 정확한 방법론이 포함된 정형 데이터 벤치마크를 제공하고자 함. 

정형 데이터에 대한 트리 기반 모델의 우수성을 이해하기 위하여, 어떤 귀납적 편향이 이러한 데이터에 적합한지 알고자 함. 

By transforming tabular datasets to modify the performances of different models, we uncover differing biases of tree-based models and deep learning algorithms which partly explain their different performances: neural networks struggle to learn irregular patterns of the target function, and their rotation invariance hurt their performance, in particular when handling the numerous uninformative features present in tabular data.
정형 데이터셋을 변환하며 다양한 모델의 성능을 확인함으로써, 트리 기반 모델과 딥러닝 알고리즘의 서로 다른 성능을 부분적으로 설명하는 다양한 편향을 발견함. 신경망은 목표 함수의 불규칙한 패턴을 학습하는 데 어려움을 겪고, 특히 정형 데이터에 존재하는 수많은 비정보적 특징을 처리할 때 딥러닝 모델의 회전 불변성으로 인해 성능이 저하됨.

정형 데이터를 위한 새로운 벤치마크를 만들고, 하이퍼파라미터 선택 비용을 고려한 정형 데이터에서의 모델 비교를 제공함. 또한, 트리 기반 모델이 우수한 이유를 찾기 위해 정형 데이터를 변환시켜 어떤 변환이 모델 간(트리 기반 모델과 딥러닝 기반 모델) 성능 격차를 넓히거나 좁히는지 알고자 함. 이를 통해 정형 데이터를 위한 바람직한 편향을 알고자 함. 

# Related work 

## Deep learning for tabular data

딥러닝에 적합한 데이터 인코딩 기법, 트리 기반 알고리즘의 귀납적 편향은 유지하며 NN의 유연성을 활용하는 하이브리드 방법, Factorization Machines, 정형 데이터를 위한 트랜스포머 아키텍처, 정규화 기법 등 정형 데이터를 위한 딥러닝 연구는 다양하게 진행되고 있음. 이번 논문에서는 MLP와 트랜스포머에서 영감을 받은 딥러닝 모델에 초점을 맞춤. 

## Comparisons between NNs and tree-based models

Tabular Data: Deep Learning is Not All You Need 논문에서는 포괄적인 벤치마크를 만들기보다는, 딥러닝 모델은 새로운 데이터셋에서 일반화 성능이 떨어진다는 사실을 밝히는데 더 큰 목적이 있었음. 다른 연구에서도 정형 데이터를 위한 딥러닝을 검토하며 최근 알고리즘을 벤치마크 했지만, 단 3개의 데이터셋을 사용하였고 정형 데이터를 위한 통합 벤치마크의 필요성을 강조했음. 

정형 데이터를 위한 새로운 아키텍처를 소개하는 대부분의 논문은 다양한 알고리즘을 벤치마킹하지만, 평가 방법론이 매우 가변적이고 적은 데이터셋을 사용하며, 평가가 저자의 모델에 편향될 수 있음. 이번 논문은 하이퍼파라미터 튜닝 비용을 고려하여, 중형/대형/범주형 특징 포함/미포함 등의 다양한 설정을 가진 45개의 데이터셋을 사용해 보다 포괄적인 벤치마크를 제공하고자 함. 

## No standard benchmark for tabular data

컴퓨터 비전, NLP와는 다르게 정형 데이터에는 표준 벤치마크가 존재하지 않음.

## Understanding the difference between NNs and tree-based models

트리 기반 모델이 정형 데이터에서 NN보다 우수한 성능을 보이는 이유에 대한 조사는 많이 진행되지 않음. 한 논문에서는 13가지 정규화 기법을 통해 MLP를 검색하여 데이터 세트별 조합을 찾으면 좋은 성능을 얻을 수 있다고 주장함. 이는 MLP가 정형 데이터에 대한 충분한 표현력이 있지만, 적절한 정규화가 부족할 수 있다는 설명을 제공함. 

# A benchmark for tabular learning

## 45 reference tabular datasets

다음 기준에 맞는 45개의 정형 데이터셋을 사용함
- Heterogeneous columns
- Not high dimensional
- Undocumented datasets
- I.I.D. data
- Real-world data
- Not too small
- Not too easy
- Not deterministic 

## Removing side issues

학습 과제를 최대한 균일하게 할 수 있도록, 다음과 같이 조절하였음
- Medium-sized training set: 데이터셋의 샘플이 최대 10,000개가 되도록 조절. 더 큰 사이즈의 데이터셋은 부록에서 다룸. 
- No missing data
- Balanced classes
- Low cardinality categorical features: 고유값이 20개 이상인 범주형 특성은 제거
- High cardinality numerical features: 고유값이 10개 미만인 숫자형 특성은 제거. 2개의 고유값을 가진다면 범주형 특성으로 변환.  

## A procedure to benchmark models with hyperparameter selection

하이퍼파라미터 튜닝을 위해 데이터셋당 약 400회의 무작위 검색을 실시. 트리 기반 모델의 경우 CPU를, NN 기반 모델의 경우 GPU를 사용함.

무작위 검색 횟수 $n$에 따라 달라지는 성능을 조사하기 위해, 검증 세트에서 최적의 하이퍼파라미터 조합을 찾고, 테스트 세트에서 이를 평가함. 무작위 검색 순서를 바꾸며 이 작업을 15회 반복하였음. 또한 각 모델의 기본 하이퍼파라미터로 무작위 검색을 시작함. 

## Aggregating results across datasets
모델 성능 평가를 위해 분류의 경우 정확도를, 회귀의 경우 R2 결정계수를 사용함. 다양한 난이도의 데이터셋에 설친 결과를 집계할 수 있도록, affine renormalization를 사용하여 테스트 정확도를 0~1 사이로 정규화함. 

## Data preparation

다음과 같은 전처리를 진행함.

1. NN의 훈련 시, 사이킷런의 QuantileTransformer를 적용
2. 회귀 문제에서, 타깃 분포의 왜도가 높다면 로그 변환 수행
3. 범주형 변수를 처리할 수 없는 모델의 경우, 사이킷런의 OneHotEncoder 적용

# Tree-based models still outperform deep learning on tabular data

## Models benchmarked

트리 기반의 모델의 경우 RandomForest, GradientBoostingTrees(or HistGradientBoostingTrees when using categorical features), XGBoost를 사용.
딥러닝 기반 모델의 경우 MLP(with learning rate scheduler), Resnet(similar to MLP with dropout, batch/layer normalization, and skip connections),  FT_Transformer, SAINT를 사용.

## Result

### Tuning hyperparameters does not make NNs state-of-the-art

하이퍼파라미터 튜닝 속도를 고려하지 않고도, 모든 무작위 검색 횟수에서 트리 기반 모델이 더 우수한 성능을 보여주었음. 

### Categorical variables are not the main weakness of NNs

범주형 변수는 종종 정형 데이터에서 NN을 사용할 때의 주요 문제로 간주됨. 범주형 변수를 포함하는 데이터셋일 경우, 트리 기반 모델과 NN 간의 격차가 더 벌어졌음. 하지만 숫자형 변수만을 사용할 때도 모델 간 격차가 대부분 존재함. 

# Empirical investigation: why do tree-based models still outperform deep learning on tabular data?

## Methodology: uncovering inductive biases 

트리 기반 모델이 NN보다 좋은 성능을 보여주는 것을 확인하였음. 이번 섹션에서는 정형 데이터에 잘 맞는 결정 트리의 귀납적 편향과 NN의 귀납적 편향이 어떻게 다른지 알아보고자 함. 이는 다른 말로, 정형 데이터의 어떤 특징이 트리 기반 모델의 학습을 쉽게 하고, NN으로는 학습하기 어렵게 만드는지 알고자 하는 것임.

이를 알기 위해 정형 데이터에 다양한 변환을 적용함. 실험의 복잡함을 줄이기 위해 숫자형 변수만을 가진 분류 문제의 중간 규모의 데이터셋만을 사용함.

## Finding 1: NNs are biased to overly smooth solutions 

가우시안 커널을 사용하여 훈련 데이터셋을 smoothing 하였음. 이런 작업은 모델이 목표 함수의 불규칙한 패턴을 학습하는 것을 효과적으로 방지함. lengthscale이 작은 경우, smoothing을 적용하였을 때 트리 기반 모델의 정확도가 현저하게 감소하지만 NN의 정확도에는 큰 영향을 미치지 않았음.

이 실험 결과는 주어진 데이터셋의 목표 함수가 smooth하지 않으며, NN이 트리 기반 모델에 비해 불규칙한 함수를 훈련하는 데 어려움을 겪고 있음을 뜻함. 이 결과는 NN이 low-frequency 함수에 편향되어 있다는 다른 논문 결과와 일치함. 결정 트리 기반의 모델은 piece-wise constant function(계단 함수)을 가져 이러한 편향성을 보이지 않음. 

이러한 관찰은 ExU 활성화 함수와 periodic embedding의 이점을 설명할 수 있음. 이러한 요소는 모델이 목표 함수의 high-frequency 부분을 학습하는 데 도움이 될 수 있음.

## Finding 2: Uninformative features affect more MLP-like NNs

### Tabular datasets contain many uninformative features

RandomForest의 특성 중요도를 사용하여, 데이터셋의 특성을 제거하며 실험. GradientBoostingTree의 경우, 절반의 특성을 제거하여도 분류 정확도에 큰 영향을 미치지 않았음. 또한, 제거된 특성(특성 중요도가 낮은 특성)으로 학습된 GradientBoostingTree의 정규화된 테스트 정확도는 제거된 특성의 20%까지는 매우 낮고 50%까지는 0.5 이하인데, 이러한 결과는 제거된 특성의 대부분이 정보가 없다는 것을 의미함. 

### MLP-like architectures are not robust to uninformative features

uninformative 특성을 제거할수록 Resnet(MLP-like)과 다른 모델 간의 성능 격차가 줄어드는 반면, uninformative 특성이 추가될수록 모델 간 격차가 더 커지는 것을 확인할 수 있음. 이러한 결과는 MLP가 uninformative 특성에 강건하지 못하다는 것을 의미함.

## Finding 3: Data are non invariant by rotation, so should be learning procedures

다른 모델에 비해 MLP가 uninformative 특성에 영향을 많이 받는 이유는 무엇인가? 한 가지 대답은 훈련 세트에서 MLP를 학습하고 검증하며 테스트하는 이 과정이 회전 불변성을 가진다는 것임(훈련 데이터와 테스트 데이터에 rotation 연산을 적용하여도 learning procedure가 변경되지 않음). 다른 논문에서는, 회전 불변의 learning procedure는 관련 없는 특성의 수에 따라 최소 선형적으로 증가하는 샘플 복잡도를 가진다고 말함.

데이터셋을 무작위로 회전하였을 때, Resnet의 경우만 테스트 점수가 변하지 않아 Resnet이 회전 불변임을 확인할 수 있음. 또한, 무작위 회전 시 모델 간 성능 순서가 뒤바뀜. 

This suggests that rotation invariance is not desirable: similarly to vision [Krizhevsky et al., 2012], there is a natural basis (here, the original basis) which encodes best data-biases, and which can not be recovered by models invariant to rotations which potentially mixes features with very different statistical properties.
데이터에는 데이터를 가장 적절히 인코딩할 수 있는 natural basis가 있으며, 이는 회전 불변 모델로는 복구할 수 없어 통계적 특성이 매우 다른 특성으로 변환될 수 있음.

이러한 결과는 MLP, 트랜스포머 모델 이전에 임베딩 층을 추가하는 다른 논문들의 결과를 뒷받침 함. 임베딩 층은 회전 불변성을 깨뜨림. 임베딩 층을 도입함으로 생기는 성능의 개선은 임베딩 층이 불변성을 깨뜨리기 때문임. 임베딩보다 계산 비용이 적게 드는 회전 불변을 깨는 다른 방법을 찾는 것이 향후 연구의 유망한 방향이 될 수 있음. 

#  Discussion and conclusion

## Limitation

정형 데이터에 잘 맞는 트리 기반 모델의 다른 귀납적 편향은 무엇인지, 작거나 큰 데이터셋에서의 평가 결과는 어떠한지, 결측치가 많거나 카디널리티가 높은 범주형 특성이 포함된다면 어떨지 등의 추가적인 실험이 필요함. 

## Conclusion
정형 데이터에서 트리 기반 모델이 훨씬 적은 계산 비용으로 좋은 예측을 얻을 수 있었음. 

This superiority is explained by specific features of tabular data: irregular patterns in the target function, uninformative features, and non rotationally-invariant data where linear combinations of features misrepresent the information.
이러한 우수성은 정형 데이터의 특징들: 목표 함수의 불규칙 패턴, uninformative 특성들, 회전 불변이 아닌 데이터에 의함. 

이번 논문을 통해 공유된 벤치마크가 새로운 아키텍처 비교에 기여할 수 있기를 기대함. 