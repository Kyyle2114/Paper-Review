본 포스팅은 제가 읽었던 논문을 간단하게 정리하는 글입니다. 논문의 모든 내용을 작성하는 것이 아닌, 일부분만 담겨 있으므로 자세한 내용은 [원본 논문](https://paperswithcode.com/paper/network-on-network-for-tabular-data)을 확인해 주시기를 바랍니다. 또한, 논문을 잘못 이해한 부분이 있을 수 있으므로, 양해 바랍니다.

---

![](https://velog.velcdn.com/images/kyyle/post/f93f1276-dfcd-4525-93b1-9a84f6c786cd/image.png)

# Abstract

정형 데이터는 실생활에서 가장 흔한 데이터임. 이번 논문에서는 정형 데이터 분류를 위한 심층 신경망 기반의 Network On Network(NON) 모델을 제안함.

However, most of them use operations like neural network and factorization machines to fuse the embeddings of different features directly, and linearly combine the outputs of those operations to get the final prediction. As a result, the intra-field information and the non-linear interactions between those operations (e.g. neural network and factorization machines) are ignored. 
지금까지의 딥러닝 기반 모델(정형 데이터를 위한)들은 다양한 특징의 임베딩을 직접 융합하고, 선형적으로 결합하여 최종 출력을 얻게 됨. 이러한 방법은 필드 내 정보(intra-field information)와 연산 간의 비선형 상호 작용이 무시됨. 

* field = feature

NON은 intra-field 정보와 비선형 상호 작용을 최대한 활용하기 위한 모델이며, 다음과 같은 세 가지의 구성 요소를 가짐

1. 필드 내 정보를 포착하는 field-wise 하단 네트워크
2. 데이터 기반으로 적절한 연산을 선택하는 중간 네트워크
3. 선택한 연산의 출력을 deeply 하게 융합(fuse)하는 상단 네트워크 

6개 데이터셋을 사용한 실험을 통해, NON이 다른 sota 모델에 비해 더 나은 성능을 보여주는 것을 확인함. 또한, 특성 임베딩 공간에 대한 정성/정량적 연구는 NON이 intra-field 정보를 효과적으로 캡쳐할 수 있음을 보여줌. 

# Introduction

정형 데이터는 널리 사용되는 데이터이며, 분류 성능을 개선할 경우 비즈니스에 상당한 추가 수익을 가져올 수 있음. 정형 데이터는 숫자 필드와 범주 필드가 혼합되어 있으며, 범주 필드는 대게 고차원적임(id의 경우 unique한 id의 개수가 매우 많음). 이러한 이유로 분류 문제가 까다로워짐.

Wide&Deep, DeepFM, xDeepFM, AutoInt 등 기존에 제안되었던 딥러닝 기반의 모델들은 다음과 같은 특징을 가졌음.

1. 입력되는 범주형 특성을 저차원으로 임베딩
2. DNN 또는 FM 등의 여러 연산을 사용하여 다른 필드의 임베딩을 융합
3. 각 연산의 출력을 선형적으로 결합하여 최종 예측을 얻음 

이러한 모델에는 장점이 있지만, 해결해야 할 세 가지 문제가 있음.

1. First, the information that features inside each field belong to the same field are not fully captured, since we fuse all the embeddings of different fields directly without considering this information explicitly. 
필드 내 정보를 고려하지 않고 임베딩을 융합하기 때문에 각 필드 내부의 특징이 동일한 필드에 속한다는 정보를 완전히 포착하지 못함 / 예를 들어, 특정 ID가 광고주인지 사용자인지에 대한 정보는 분류를 더 정확하게 할 수 있게함. 
2. 데이터와는 상관 없이 미리 정의된 연산 조합을 사용
3. 연산 간의 비선형 상호 작용은 2.에서 언급된 연산 조합에서 무시됨 

이러한 문제를 완화할 수 있도록 NON를 제안함. NON의 Field-wise 네트워크는 각 필드마다 고유한 DNN을 사용하여 필드 내 정보를 포착함. 사전 정의된 연산을 사용하지 않고, 입력 데이터에 가장 적합한 연산을 선택하고, 보조 분류기(Inception과 같은)를 추가하여 훈련을 더 쉽게 함. 

# Related Work

## Tabular data classification

위에서 언급했듯, 정형 데이터의 범주형 필드는 일반적으로 고차원적임. RF, GBM 등의 트리 방법은 숫자 필드의 정형 데이터에서는 잘 동작하지만, 고차원적인 정형 데이터에는 적합하지 않음. 고차원 범주형 필드의 경우, 트리의 각 노드에 대해 필드에 대한 모든 특징을 열거해야 하므로 비효율적임. 또한, 범주형 특성의 희소성으로 인해 범주형 특성을 기반으로 노드를 분할하는 것은 이득이 적음. 
 
## Shallow methods

산업 현장에서, 로지스틱 회귀는 대규모 희소 정형 데이터 분류에 많이 사용되는 방법임. 하지만 LR은 많은 피처 엔지니어링이 필요하고, 선형성 때문에 특성 간 상호작용을 학습하는 능력이 부족함. FM, FFMs의 얕은 구조로, 표현 능력이 제한적임. 

## Deep methods

임베딩 벡터와 비선형 활성 함수의 이점을 통해, DNN은 암묵적으로 고차원 특성의 상호작용을 학습할 수 있음. 임베딩 함수는 고차원의 범주형 특성을 저차원의 고밀도 벡터로 변환함. 원핫 인코딩된 벡터는 매우 희소하고 고차원적이므로, 임베딩 함수를 사용하여 저차원의 벡터로 변환할 수 있음($e = Wx$).

Wide&Deep, AutoInt 등의 모델은 필드 내의 특징이 동일한 필드에 속한다는 정보와 연산 간의 비선형적 상호 작용을 사용하지 않음.
 
# NETWORK ON NETWORK

NON은 하단의 필드별 네트워크, 가운데의 필드 간 네트워크, 상단의 연산 융합 네트워크 등 세 부분으로 구성됨.

필드별 네트워크에서는 한 필드에 속한 피처들이 신경망을 공유하여, 각 필드마다 필드 내 정보를 포착함.  필드 간 네트워크에서는 필드 간 선형적인 상호작용을 모델링하는 로지스틱 회귀, 고차원의 상호작용을 모델링하는 Multi-layer NN 등이 존재함. 연산 융합 네트워크에서는 다양한 연산으로부터 얻어진 특성(표현)들을 DNN을 사용하여 융합, 최종 예측을 얻음. 

깊은 구조의 문제를 완화하기 위해, 보조 분류기Auxiliary classifier를 매 DNN 계층에 추가함. 
 
* Auxiliary classifier란, gradient 전달이 잘되지 않는 하위 layer를 훈련하기 위해 사용하는 보조 분류기로써, 중간중간에 softmax 등을 두어 중간에서도 역전파를 하게 함. 이를 통해 gradient가 잘 전달되지 않는 문제를 해결할 수 있음. 


## The structure of NON

###  Field-wise network

필드 내 정보를 포착하기 위한 필드 고유의 DNN. 특성의 임베딩 또는 특성값이 DNN의 입력이 됨. DNN의 학습 파라미터가 필드 내 정보를 저장함. 

$e_i' = \text{DNN}_i(e_i)$

동일한 구조의 DNN이 많다면, 모든 파라미터를 하나의 행렬로 표현하여 한 번에 계산할 수 있음.

$X' = \text{ReLU} (\text{matmul} (X, W) + b)$

field-wise 네트워크의 출력을 바로 다음 단계의 입력으로 사용할 수 있지만, 다음과 같이 조금 더 refine 할 수 있음.

$\hat{e}_i = F(e_i', e_i)$ 	

입력 전, 후의 데이터를 함수 $F$를 함께 통과시킴. $F$는 concat, element-wise product, gating 메커니즘 등 다양한 함수가 될 수 있음.

### Across field network

필드 간 상호작용을 모델링. 서로 다른 유형의 필드 간 상호작용을 학습하기 위해 여러 연산을 채택함. LR(로지스틱 회귀), vanila DNN, self-attention, FM, Bi-Interaction 등의 연산 사용

NON의 구조는 기존의 딥러닝 분류 모델과 호환 가능하며, DNN을 제외한 LR, FM 등의 연산은 하이퍼파라미터로 간주되어 데이터에 맞게 사용할 수 있음. 

### Operation fusion network

기존 방법의 경우(Wide&Deep 등), 서로 다른 연산의 출력을 연결한 후 가중 합산하여 시그모이드 함수의 입력을 얻음. ($W^T[h_1, h_2]$). 이러한 방법은 내재적인 선형성 때문에 연산 간 비선형적 상호작용을 무시하게 됨.

연산 간 상호작용을 더 잘 포착하기 위해, 이전 단계의 출력을 연결한 뒤 DNN에 입력하여 서로 다른 출력을 융합시킴. 

## DNN with auxiliary losses

DNN의 훈련을 더 쉽게 할 수 있도록 보조 분류기를 추가. 훈련 과정에서만 보조 손실을 사용하며, 추론 단계에서는 vanila DNN으로 사용.

## Time complexity analysis

NON의 시간 복잡성은 필드 간 네트워크에 의해 지배되며, 필드 간 네트워크에서 경량 연산을 선택하면 NON의 시간 복잡성을 줄일 수 있음.

# NUMERICAL EXPERIMENT

다음과 같은 4가지 질문의 답을 얻기 위해 실험 진행 
- Q1: How does the design paradigm of NON perform?
- Q2: How does NON perform as compared to state-of-the-art methods? 
- Q3: What are the most suitable operations for different datasets? 
- Q4: Can the field-wise network in NON capture the intra-field information effective

## Experiment setup

Criteo, Avazu, Movielens, Talkshow, Social, Sports 총 6개의 데이터셋을 사용하여 실험 진행. FFM, DNN, Wide&Deep, NFM, xDeepFM, AutoInt와 성능을 비교. Among them, NFM, xDeepFM and AutoInt are the state-of-the-art methods.

## Study of design paradigm of NON (Q1)

보조 손실을 추가하는 디자인으로, 일반화 성능을 개선하였음. 보조 손실이 있는 경우 없는 경우에 비해 1.67배의 훈련 속도를 높일 수 있었음. 

또한, 총 3단계로 구성된 NON의 구조를 한 단계씩 제거함으로 성능을 비교하였을 때, 3단계가 모두 포함된 NON의 구조가 다른 모든 방법보다 성능이 뛰어남을 확인함. 

##  Performance comparison (Q2)

NON은 모든 데이터 세트에서 항상 최상의 결과를 얻을 수 있으며 DNN에 비해 테스트 AUC에서 0.64%∼0.99% 개선되었음. 

## Study of operations (Q3)

Across field network의 연산이 변경될 때마다 성능이 변화하며, 데이터 세트에 따라 최대와 최소 테스트 AUC의 격차가 0.1%에서 0.9%까지 다양함. 모든 데이터셋에서 항상 최상의 성능을 달성하는 연산 조합이 없으므로, 데이터에 따라 연산을 선택해야 함. 

데이터셋이 복잡할수록 연산의 수가 많고 복잡한 Attention 등의 연산을 선호하며, 작은 데이터셋일수록 연산의 수가 적고 가벼운 LR 등이 연산을 선호함. 

##  Study of field-wise network (Q4)

field-wise 네트워크의 출력을 t-SNE를 사용해 시각화하였을 때, 각 필드 내의 특성들은 서로 가깝게 위치되고, 서로 다른 필드에 속한 특성들은 쉽게 구분됨을 확인할 수 있음. 

# CONCLUSION
필드 내 정보와 비선형 상호작용을 고려하는 NON 모델을 소개하였음. 다른 모델 대비 우수한 성능을 얻었으며, 필드 내 정보를 효과적으로 포착함.
