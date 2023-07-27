본 포스팅은 제가 읽었던 논문을 간단하게 정리하는 글입니다. 논문의 모든 내용을 작성하는 것이 아닌, 일부분만 담겨 있으므로 자세한 내용은 [원본 논문](https://paperswithcode.com/paper/tabnet-attentive-interpretable-tabular)을 확인해 주시기를 바랍니다. 또한, 논문을 잘못 이해한 부분이 있을 수 있으므로, 양해 바랍니다.

--- 

![](https://velog.velcdn.com/images/kyyle/post/38df09e6-4e68-4471-b426-63a3141014e0/image.png)


# Abstract

deep tabular data learning architecture, TabNet
정형 데이터를 위한 딥러닝 모델, TabNet

TabNet uses sequential attention to choose which features to reason from at each decision step
매 의사결정마다 sequential attention을 사용하여 사용할 특성을 선택함. 이는 interpretability의 역할을 함.

self-supervised learning for tabular data, significantly improving performance when unlabeled data is abundant.
자기 지도 학습을 정형 데이터에 적용하여 성능 향상을 보임 

# Introduction

딥러닝 모델은 이미지, 텍스트, 오디오 등 비정형 데이터에서 뛰어난 성과를 얻었지만, 정형 데이터에서는 그렇지 않음. 정형 데이터에서는 여전히  variants of ensemble decision trees 모델들이 강세를 보임.

DT-based model(ensemble)이 강세인 이유는? 

1.  정형 데이터에서 흔히 나타나는 초평면 결정경계(approximately hyperplane boundaries which are common in tabular data)를 학습함에 효과적임
2. 해석력이 좋으며, 사후 분석이 가능함
3.  훈련 시간이 짧음 

또한, 딥러닝 기반 모델은 파라미터가 너무 많거나, 적절한 귀납 편향(inductive bias) 설계가 어려워 최적해를 찾기가 어려울 수 있음. 

그럼에도 정형 데이터에 딥러닝을 적용해야 하는 이유는?

1. 큰 데이터셋에 대한 성능 기대
2. gradient descent based end-to-end learning

특히, gradient descent based end-to-end learning은 다음과 같은 장점을 기대할 수 있음.

1. 정형 데이터 + 이미지 등 여러 데이터를 한 번에 사용할 수 있음
2. 특성 엔지니어링 등 전처리 작업을 줄일 수 있음
3. streaming data 처리 가능 
4. end-to-end 모델의 표현 학습 기능(입력 데이터의 특성을 task에 맞게 특성 변환)

TabNet의 주요 장점

1. TabNet inputs raw tabular data without any preprocessing and is trained using gradient descent-based optimization, enabling flexible integration into end-to-end learning.
별다른 전처리 없이 데이터를 입력할 수 있으며, 경사하강법 기반 훈련이 가능하여 end-to-end 학습이 가능함.

2. TabNet uses sequential attention to choose which features to reason from at each decision step, enabling interpretability and better learning as the learning capacity is used for the most salient features. This
feature selection is instance-wise.
매 의사결정마다 sequential attention을 사용하여 특성을 선택. 이러한 과정은 instance-wise하게 적용됨.

3. Finally, for the first time for tabular data, we show significant performance improvements by using unsupervised pre-training to predict masked features.
정형 데이터에서는 처음으로, Unsupervised pre-training을 사용하여 성능 향상을 보였음.

# Related Work

## Feature selection

흔히 사용되는 특성 선택(forward selection, Lasso regularization)은 전체 학습 데이터를 기반으로 특성 중요도를 부여하고, 이를 global method라고 함.

Instance-wise feature selection은 각 입력에 대해 개별적으로 특성을 선택함. 기존의 연구와 달리, TabNet은 제어 가능한 희소성을 갖춘 soft feature selection을 수행함. 단일 모델이 특성 선택과 출력 매핑을 공동으로 수행하여 간결한 표현으로 뛰어난 성능을 발휘함. 

TabNet은 추론을 위해 여러 개의 decision block을 사용함. 

## Tree-based learning

트리 기반 모델은 정형 데이터에서 주로 사용됨. 트리 기반 모델의 가장 큰 장점은 information gain을 가장 많이 얻을 수 있는 global 특성을 효율적으로 선택하는 것임. 기본적인 결정 트리의 성능을 향상시키기 위해, RandomForest, XGBoost 등의 앙상블 방법이 자주 사용됨.

실험 결과에 따르면, 딥러닝 모델이 트리 기반 모델의 특성 선택 속성을 유지하며 표현 능력을 개선하면 트리 기반 모델보다 성능이 뛰어날 수 있음. 

## Integration of DNNs into DTs

DNN 블록으로 결정 트리를 표현하면 표현의 중복과 비효율적인 학습이 발생함. 미분 가능한 함수를 사용한 Soft DTs, soft binning function 등 다양한 연구가 있었음. TabNet은 sequential attention을 사용하여 soft feature selection을 수행한다는 점에 있어 다른 연구와 차별화됨. 

## Self-supervised learning

비지도 표현 학습은 소규모 데이터셋에서 지도 학습의 성능을 개선함. 최근, 텍스트와 이미지 데이터에서 비지도 학습(masked input prediction)과 어텐션 기반 딥러닝의 상당한 발전이 있었음.

# TabNet for Tabular Learning

특정 설계를 사용하여 기존의 DNN을 통해 결정 트리와 유사한 출력 매니폴드를 구현할 수 있음. 이러한 설계에서 개별 특징 선택은 결정 경계를 얻기 위한 핵심임.

TabNet은 이러한 기능을 기반으로 하여, 다음과 같은 설계를 기반으로 결정 트리의 성능을 뛰어넘음

1. sparse instance-wise feature selection learned from data
2. sequential multi-step architecture, where each step contributes to a portion of the decision based on the selected features
3. nonlinear processing of the selected features
4. mimics ensembling via higher dimensions and more steps

TabNet의 인코딩은 $N$단계의 sequential multi-step을 기반으로 함. $i$번째 스텝에서는 $i-1$번째 스텝에서 처리된 정보를 사용하여 어떤 특성을 선택할지 결정하고, 처리된 특성 표현을 출력하여 전체 결정에 합산함. 

## Feature selection
학습 가능한 마스크 $\mathbf M[\mathbf i]$를 사용하여 soft selection 수행. 가장 두드러진 특성을 희소하게 선택하기 때문에 의사 결정 단계에서 관계 없는 특성에 learning capacity를 낭비하지 않으므로 효율적인 학습 가능. 

마스킹은 $\mathbf M[\mathbf i] \cdot \mathbf f$로 곱셈으로 이뤄지며, 마스크를 얻기 위해 attentive transformer를 사용함. 이때, 이전 단계에서 처리된 특성 $\mathbf a[\mathbf i-1]$을 사용함. 즉, 식으로 표현하면 $\mathbf M[\mathbf i] = \text{sparsemax}(\mathbf P[i-1] \cdot h_i(\mathbf a[\mathbf i-1))$.

Sparsemax normalization encourages sparsity, which is observed to be superior in performance and aligned with the goal of sparse feature selection for explainability.

$\mathbf M[\mathbf i]$를 모두 더하면 1이 되고, $h_i$는 훈련 가능한 함수(FC-BN)임. $\mathbf P[\mathbf i]$는 prior scale term으로, 특정 특성이 이전에 얼마나 사용되었는지를 나타냄. $\mathbf P[\mathbf i]$는 $(\gamma - \mathbf M[\mathbf j])$의 곱들로 이뤄지며, $\gamma$는 영향력이 지나치게 큰 특성이 지나치게 중복 선택되는 것을 막기 위한 relaxation 파라미터임. $\gamma=1$이면 특성이 오직 1번의 결정 단계에서만 선택되도록 강제함. $\gamma$가 증가할수록 특성이 여러 결정 단계에서 선택될 수 있는 유연성이 제공됨. 

선택한 특성의 희소성을 더 제어하기 위해, 엔트로피 형태의 희소성 정규화를 사용함. 희소성은 대부분의 특징이 중복되는 데이터셋에 유리한 귀납적 편향을 제공함.

## Feature processing
Feature Transformer를 사용하여 필터된 특성들을 처리하고, 결정 단계의 출력($\mathbf d[\mathbf i]$)과 다음 단계의 입력으로 사용할 수 있게 처리된 특성($\mathbf a[\mathbf i]$)을 나눔(split). 파라미터의 효율성과 강건한 학습을 위해, feature transformer는 모든 결정 단계에서 공유되는 레이어(shared across decision steps)와 각 결정 단계에 종속적인 레이어(decision step dependent)로 구성되어 있음. 각 FC 레이어는 BN, GLU로 이어지며, residual connection과 $\sqrt{0.5}$를 통한 정규화에 연결됨. $\sqrt{0.5}$를 사용한 정규화는 네트워크 전체의 분산이 급격하게 변하지 않도록 하여 학습을 안정화시키는 데 도움이 됨.

빠른 학습을 위해 BN과 함께 큰 배치 사이즈를 사용함. BN의 경우, 처음 입력 특징을 제외하고는 ghost BN을 사용함. 출력 매핑을 위해, 모든 $\mathbf d[\mathbf i]$를 더한 후($\mathbf d_{\text{out}} = \sum \text{ReLU}(\mathbf d[\mathbf i])$), 선형 매핑($\mathbf W_{\text{final}}\mathbf d_{\text{out}}$)을 사용하여 출력 매핑을 얻음.

## Interpretability
TabNet의 특성 선택은 각 단계에서 선택된 특성을 조명할 수 있음. 만약 $\mathbf M_{b, j}[\mathbf i] = 0$이라면, b번째 샘플의 j번째 특성은 결정에 아무런 기여를 하지 않은 것임. $f_i$가 선형 함수라면(feature transformer), 마스크 계수 $\mathbf M[\mathbf i]$는 특성 중요도에 해당함. 각 의사 결정 단계에는 비선형 처리를 사용하지만, 그 출력은 나중에 선형적으로 결합됨. 

$b$번째 샘플의 $i$번째 결정 단계에서의 결정 기여도를 나타내기 위해,$\eta_\mathbf b[\mathbf i]=\sum_{c=1}^{N_d}\text{ReLU}(\mathbf d_{b, c}[\mathbf i])$를 제안함. 이 값은 각 단계의 상대적 중요도를 평가할 수 있는 계수임(여러 단계의 마스크를 결합하기 위한 계수). 직관적으로, $\mathbf d_{b,c}[\mathbf i]$ < 0이면 $i$번째 결정 단계의 모든 특징은 전체 결정에 대한 기여도가 0이어야 함. 이 값이 커질수록 전체 선형 결합에서 큰 역할을 함. 

이 계수 $\eta_\mathbf b[\mathbf i]$를 마스크 $\mathbf M[\mathbf i]$와 곱한 후 스케일링하여, 총 특징 중요도 마스크를 계산할 수 있음. 

## Tabular self-supervised learning
TabNet으로 인코딩된 표현을 재구성하기 위해 디코더 아키텍처를 제안함. 디코더 아키텍처는 각 단계별로 feature transformer와 FC 레이어로 구성됨. 

디코더 아키텍처를 사용하여, 누락된 값을 예측하는 작업을 수행함. binary mask $\mathbf S$를 사용하여, 인코더는 $(1-\mathbf S) \cdot \hat f$를 입력하고 디코더는 $\mathbf S \cdot \hat f$를 재구성함. prior scale $\mathbf P[\mathbf 0] = (1 − \mathbf S)$로 하여, 알고있는 특성만 사용할 수 있도록 조절함. 

# Experiments

다양한 데이터셋에 실험을 적용함. 범주형 특성은 훈련 가능한 임베딩을 통해 스칼라 값으로 매핑되고, 숫자형 특성의 경우 전처리 없이 바로 입력함. Ablation study에서 확인할 수 있듯, TabNet의 성능은 대부분의 하이퍼파라미터에 크게 민감하지 않았음. 

# Conclusions

정형 데이터를 위한 새로운 딥러닝 아키텍처 TabNet을 제안함. TabNet은 sequential attention을 사용하여 매 의사 결정 단계에서 특성을 선택함. instance-wise한 특성 선택은 효율적인 학습을 가능하게 하며, 특성 선택 마스크의 시각화를 통해 해석력을 높일 수 있음. 다양한 데이터셋에서 TabNet의 우수한 성능을 확인하였고, self-supervised learning을 사용하여 빠른 학습 및 성능 개선을 얻음.