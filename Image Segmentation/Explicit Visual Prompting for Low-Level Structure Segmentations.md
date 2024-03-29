
본 포스팅은 제가 읽었던 논문을 간단하게 정리하는 글입니다. 논문의 모든 내용을 작성하는 것이 아닌, 일부분만 담겨 있으므로 자세한 내용은 [원본 논문](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_Explicit_Visual_Prompting_for_Low-Level_Structure_Segmentations_CVPR_2023_paper.pdf)을 확인해 주시기를 바랍니다. 또한, 논문을 잘못 이해한 부분이 있을 수 있으므로, 양해 바랍니다.

# Overview

실제 사진 같은 가짜 이미지를 찾아내거나, 이미지 속 편집된 부분(manipulated regions)을 찾아내는 등의 문제에서, **low-level structure**는 크게 도움이 되는 것으로 알려져 있음. Manipulated regions 외에도 그림자 영역, 가려진 객체 등을 찾을 때에도 **low-level clues**는 중요한 역할을 함. 

본 논문에서는 manipulated parts, identifying out-of-focus pixels, separating shadow regions, and detecting concealed objects 등 다양한 segmentation task에서 활용할 수 있는 **unfied approach**를 제안함. 

최근 NLP에서의 **prompting**은 frozen(가중치가 동결된) large foundation 모델을 다양한 downstream task에 적용할 수 있도록 **minimum extra trainable parameter(adapter 등)** 를 추가하도록 함. 이러한 prompting을 통해 downstream task에서 보다 나은 일반화 성능을 얻을 수 있음. 

NLP에서의 prompt tuning에서 영감을 받아, 새로운 visual prompting model인 **Explicit Visual Prompting (EVP)** 을 제안함. 이전의 다른 visual prompting과 달리, EVP는 각 individual image의 explicit visual content에 집중함. 

아래의 Figure 1.은 본 논문에서 제안하는 EVP를 간단하게 소개함.

![](https://velog.velcdn.com/images/kyyle/post/d6b14584-5c59-4ca0-b2a6-13661426b0bf/image.png)

위의 Figure 1.에서와 같이, EVP는 두 가지의 feature를 추가적으로 사용함. **Frozen patch embedding**과 입력 이미지의 **high-frequency component**를 사용하며, patch embedding은 large-scale 데이터셋에서 사전 훈련되고 가중치가 동결된 pre-trained model로 얻을 수 있음. 

제안한 방법을 4가지 task: forgery detection, shadow detection, defocus blur detection, camouflaged object detection에 적용하여 실험을 수행하였음. 

# Method

본 논문에서는 제안하는 EVP를 ImageNet에서 사전 훈련된 **SegFormer**에 적용(추가, adapting)하였음. EVP는 다른 prompting 방법과 유사하게 backbone을 frozen하고, tunable parameter를 조금 더 추가하여 task-specific 정보를 학습하도록 함. 

## Preliminaries

### SegFormer

**SegFormer**는 semantic segmentation을 위한 hierarchical transformer based structure임. 자세한 것은 논문을 참고. EVP는 SegFormer에만 적용할 수 있는 것이 아니라, ViT, Swin 등 다양한 network structure에도 쉽게 적용될 수 있음. 

### High-frequency Components(HFC)

![](https://velog.velcdn.com/images/kyyle/post/226f3e87-af1f-4f96-8f97-84e402315b2d/image.png)

이미지의 high-frequency 및 low-frequency component는 푸리에 변환과 푸리에 역변환 과정을 통해 얻을 수 있음. 

입력 이미지의 high-frequency component를 $I_h$, low-frequency component를 $I_l$이라고 한다면, 다음의 과정을 통해 각 component를 얻을 수 있음. 

$\text{fft}$와 $\text{ifft}$를 각각 Fast Fourier 변환과 역변환이라 하면, 푸리에 변환을 통해 이미지의 frequency component $z$를 얻을 수 있음.

$$
z = \text{fft}(I), \;\; I = \text{ifft}(z)
$$

Low-frequency 영역은 $z$에서 중앙에 위치한 영역이므로, 중앙에 가까운 영역을 0으로 처리하는 binary mak $\mathbf M_h \in \{0, 1\}^{H \times W}$를 생성함. 

$$
\mathbf M_h^{i, j}(\tau) = \begin{cases} 0, \;\; \frac{4|(i-\frac H2)(j-\frac W2)|}{HW} \le \tau \\ 1, \;\; \text{otherwise} \end{cases}

$$

$\tau$는 마스크 영역의 surface ratio를 의미하며, 값이 커질수록 더 많은 영역을 0으로 처리함. 

High-frequency component HFC는 다음과 같이 계산할 수 있음.

$$
I_{hfc} = \text {ifft}(z \mathbf M_h(\tau))
$$

Binary mask를 조금 바꾸어 LFC 또한 계산할 수 있음. 

$$
\mathbf M_l^{i, j}(\tau) = \begin{cases} 0, \;\; \frac{HW-4|(i-\frac H2)(j-\frac W2)|}{HW} \le \tau \\ 1, \;\; \text{otherwise} \end{cases}, \;\; I_{lfc} = \text{ifft}(z \mathbf M_l(\tau))

$$

RGB 이미지는 3개의 채널을 가지므로, 위의 process를 각 채널에 독립적으로 수행함.

## Explicit Visual Prompting

EVP의 목표는 image embedding과 HFC에서 explict prompt(→ task specific knowledge)를 학습하는 것임. 

아래의 Figure 3.은 EVC를 보다 자세히 설명함.

![](https://velog.velcdn.com/images/kyyle/post/02cc7f01-7eef-4eec-9d97-47cbd5dd4db5/image.png)

### Patch embedding tune

입력 이미지의 patch를 SegFormer에 입력하면 해당 patch는 $C_{seg}$차원의 벡터 $I^p$로 변환됨. 이 변환 과정의 가중치는 변경하지 않고, 추가적인 linear layer $L_{\text{pe}}$를 추가하여 $I^p$를 $c$차원으로 투영함.

$$
F_{pe} = L_{\text{pe}}(I^p), \;\;\text{with} \; c = \frac{C_{seg}}{r} 
$$

여기서 $r$은 조절할 수 있는 scale factor임. 

### High-frequency components tune

$I_{hfc}$는 SegFormer의 입력 patch size와 동일하게 분할된 다음, linear layer $L_{\text{hfc}}$에 통과되어 $c$차원으로 투영됨. 

$$
F_{hfc} = L_{\text{hfc}}(I^p_{hfc}), \;\; I_{hfc}^p \in \mathbb R^C, \;\; C=h \times w \times 3
$$

### Adaptor

Adaptor를 통해 patch embedding 정보와 high-frequency 정보를 network에 전달함. 

$i$번째 Adapter는 $F_{pe}, F_{hfc}$를 입력으로 받아 prompting $P^i$를 출력함.

$$
P^i = \text{MLP}_{\text{up}}(\text{GELU}(\text{MLP}^i_{\text{tune}}(F_{pe} + F_{hfc})))
$$

$\text{MLP}_{\text{up}}$은 모든 adaptor에서 공유되며, adaptor의 정보가 transformer layer에 더해질 수 있도록 up-projection을 수행함. 

# Experiment

앞서 소개한 4가지의 task: forgery detection, shadow detection, defocus blur detection, camouflaged object detection에서 실험을 수행하였음.

![](https://velog.velcdn.com/images/kyyle/post/50af112a-a477-4534-885d-e16adc0aceb3/image.png)

다양한 task에서 적은 수의 파라미터로 효과적인 성능 향상을 확인하였음. 자세한 실험 결과는 논문 참고. 

![](https://velog.velcdn.com/images/kyyle/post/3cf66f4e-5996-455e-8b0e-0981ca47c87a/image.png)

적은 수의 파라미터로 Full-tuning 보다 더 나은 성능을 얻을 수 있음을 확인하였음.

![](https://velog.velcdn.com/images/kyyle/post/11e8db8a-7eb1-4ab9-9390-335217ed7aa2/image.png)

$\text{MLP}^i_{\text{tune}}$를 다른 adaptor와 공유할 경우 훈련해야 하는 파라미터는 줄지만, 성능이 감소함을 확인하였음. 

$\text{MLP}_{\text{up}}$을 공유하지 않고 adaptor마다 훈련할 경우 훈련 파라미터가 증가하고 일관적인 성능 향상을 보이지 않으므로 적절하지 않음.

![](https://velog.velcdn.com/images/kyyle/post/20eaed65-8d58-44db-bc2e-2e9a1fa362e0/image.png)

Tuning Stage를 추가할수록 성능이 향상됨을 확인하였음. $\text{Stage}_x$는 $x$번째 transformer block에 adaptor를 추가한 것을 의미함. 

아래 이미지는 SegFormer의 구조.

![](https://velog.velcdn.com/images/kyyle/post/cbd7fdde-0016-4ea4-aea7-60aa5218a660/image.png)

[Ref : Xie, Enze, et al. "SegFormer: Simple and efficient design for semantic segmentation with transformers." Advances in neural information processing systems 34 (2021): 12077-12090.]

![](https://velog.velcdn.com/images/kyyle/post/4f0d8cb4-67d4-4368-a073-f3fdd2ebd211/image.png)

Scale factor $r$에 따른 실험 결과.

![](https://velog.velcdn.com/images/kyyle/post/579cfb27-c1f6-4fbd-8798-ecc474234282/image.png)

SegFormer가 아닌 plain ViT를 사용하는 SETR에 EVP를 적용한 후 성능 비교를 수행함. 다른 tuning method에 비해 보다 나은 성능을 보여줌을 확인하였음. 
