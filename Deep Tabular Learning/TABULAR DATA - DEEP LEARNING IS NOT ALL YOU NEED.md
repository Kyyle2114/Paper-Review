본 포스팅은 제가 읽었던 논문을 간단하게 정리하는 글입니다. 논문의 모든 내용을 작성하는 것이 아닌, 일부분만 담겨 있으므로 자세한 내용은 [원본 논문](https://paperswithcode.com/paper/tabular-data-deep-learning-is-not-all-you)을 확인해 주시기를 바랍니다. 또한, 논문을 잘못 이해한 부분이 있을 수 있으므로, 양해 바랍니다.

---

![](https://velog.velcdn.com/images/kyyle/post/afad5c8b-4636-43de-8ada-282f94bc7a65/image.png)

# Abstract 

정형 데이터의 분류 및 회귀 문제에서는 트리 앙상블 모델(XGBoost 등)이 권장됨. 최근에는 TabNet과 같은 딥러닝 기반 모델이 제안되며, 이러한 모델들은 XGBoost보다 성능이 뛰어나다고 주장함.

본 논문에서는 정형 데이터를 위한 딥러닝 모델과 XGBoost와의 성능을 비교함.

성능 비교 결과, XGBoost가 다른 딥러닝 모델보다 우수한 성능을 보였으며, 하이퍼파라미터 튜닝에 있어서도 XGBoost가 훨씬 적은 튜닝이 필요하였음.

긍정적인 측면에서는, 딥러닝 모델과 XGBoost의 앙상블이 XGBoost 단독보다 더 나은 성능을 보여주었다는 것임.

# Introduction

During the last decade, traditional machine learning methods, such as gradient-boosted decision trees (GBDT) [Chen and Guestrin, 2016], still dominated tabular data modeling and showed superior performance over deep learning.
지난 10년간, 그래디언트 부스트 결정 트리와 같은 전통적인 머신러닝 방법이 정형 데이터 모델링을 지배하였으며, 딥러닝 기반 모델보다 더 좋은 성능을 보여주었음.

Although the “no free lunch” principle [Wolpert and Macready, 1997] always applies, tree-ensemble algorithms, such as XGBoost, are considered the recommended option for real-life tabular data problems [Chen and Guestrin, 2016, Friedman, 2001, Prokhorenkova et al., 2018a].
"No Free Lunch" 원칙은 항상 적용되지만, 정형 데이터 문제에서 XGBoost와 같은 트리-앙상블 알고리즘은 항상 권장되는 옵션임.

최근에는 정형 데이터를 위한 여러 딥러닝 모델이 제안되고 있으며, 그중 일부는 GBDT보다 성능이 뛰어나다고 주장함. 하지만, ImageNet과 같은 표준 벤치마크가 없어 각 연구마다 서로 다른 데이터셋을 사용하였고, 모델을 정확하게 비교하기 어려움. 또한, 각 모델을 적절하고 동일하게 최적화하지 않은 경우 또한 존재하였음.

정형 데이터를 위한 딥러닝 모델이 증가하는 만큼, 이 분야의 발전 상황을 엄격하게 검토하고, 실증적인 결론이 필요함.

이 연구에서의 질문은 다음 두 가지임.

1. 모델이 좋은 성능을 보이는지. 특히, 각 논문에서 제안되지 않은 데이터셋에서도 또한 좋은 성능을 보이는지
2. 하이퍼파라미터 튜닝에 얼마나 시간이 걸리는지 

최근 4개의 논문에서 제안된 모델을 사용하였으며, 11개의 데이터셋을 사용함. 11개 중 9개는 논문에서 제안된 데이터셋임. 

We show that in most cases, each deep model performs best on the datasets used in its respective paper but significantly worse on other datasets. Additionally, our study shows that XGBoost usually outperforms deep models on these datasets. Furthermore, we demonstrate that the hyperparameter search process was much shorter for XGBoost.

On the positive side, we examine the performance of an ensemble of deep models combined with XGBoost and show that this ensemble gives the best results. 

딥러닝 모델은 논문에서 사용된 데이터셋에서는 좋은 성능을 보이지만, 다른 데이터셋에서는 성능이 현저히 떨어짐. 일반적으로 XGBoost가 성능이 뛰어났고, 하이퍼파라미터 튜닝 시간이 가장 짧았음.

# Background

사용한 4가지의 딥러닝 모델은 크게 두 범주로 나눌 수 있음.

1. Differentiable trees
기존의 결정 트리는 미분할 수 없기 때문에 경사하강법을 사용할 수 없고, 딥러닝의 end-to-end 파이프라인에 구성요소로 사용할 수 없음. 인터널 노드의 decision function을 smoothing 하여 tree function과 tree routing를 미분가능하게 하였음.

2. Attention-based models
어텐션 기반 모델은 정형 데이터를 포함한 다양한 분야에서 널리 사용됨.

사용한 4가지의 딥러닝 모델은 다음과 같음.  

1. TabNet
2. Neural Oblivious Decision Ensembles (NODE)
3. DNF-Net
4. 1D-CNN