본 포스팅은 제가 읽었던 논문을 간단하게 정리하는 글입니다. 논문의 모든 내용을 작성하는 것이 아닌, 일부분만 담겨 있으므로 자세한 내용은 [원본 논문](https://paperswithcode.com/paper/deep-learning-for-anomaly-detection-a-survey)을 확인해 주시기를 바랍니다. 또한, 논문을 잘못 이해한 부분이 있을 수 있으므로, 양해 바랍니다.

---

![](https://velog.velcdn.com/images/kyyle/post/c4b47bb4-fba6-474a-b32b-b29d1f1d3bed/image.png)


# Abstract

본 논문의 목적은 두 가지로, 딥러닝 기반 이상 탐지 연구 방법의 포괄적인 개요를 제시하며, 다양한 도메인에서 이상 징후에 대한 이러한 연구 방법의 도입을 검토하고 그 효과를 평가하고자 함. 

기본 가정과 접근 방식에 따라 이상 탐지 방법을 여러 카테고리로 분류함. 각 카테고리에는 기본적인 이상 탐지 방법과 그 변형을 소개하고 주요 가정을 설명함. 또한, 각 카테고리에 맞는 장점과 한계점을 제시하고 실제 도메인에서의 계산 복잡도 또한 논의함.

마지막으로, 현실 문제에 딥러닝 기반 모델을 사용할 때 존재하는 문제와 도전 과제를 간략하게 설명함.

# Introduction

Hawkins는 outlier를 다른 관측치와 크게 벗어나 다른 메커니즘에 의해 생성되었다는 의심을 불러일으킬 정도로 다른 관측치와 크게 벗어난 관측치로 정의함. 

최근 몇 년 동안 딥러닝 기반 이상 탐지 알고리즘이 대중화되어 다양한 도메인에 적용되고 있으며, 기존 방법을 완전히 능가한다는 연구 결과도 존재함. 본 논문의 목적은 두 가지로, deep anomaly detection (DAD)의 체계적이고 종합적인 리뷰를 제시하며, 다양한 도메인에서 DAD 방법의 채택과 그 효과를 평가하고자 함. 

# What are anomalies?

Anomalies는  abnormalities, deviants, outliers 등으로 불림. 악의적인 행동, 시스템 장애, 의도적인 사기 등으로 생성되며, 이러한 이상치는 데이터에 대한 중요한 정보를 전달하는 경우가 많음. 따라서, 이상 탐지는 의사 결정 시스템에서 필수적인 단계로 간주됨.

# What are novelties?

Novelty detection은 데이터에서 새롭거나 관찰되지 않은 패턴을 식별하는 것임. unseen 데이터에 대해, novelty score를 계산하고 이 점수가 임계값과 크게 차이 난다면 anomalies 또는 outliers로 간주할 수 있음. anomaly detection에 사용되는 기법은 novelty detection에 사용될 수 있고, 그 반대도 마찬가지임. 

# Motivation and Challenges: Deep anomaly detection (DAD) techniques

- 기존의 고전적인 방법(이상 탐지를 위한)들은 데이터의 복잡한 구조를 포착하지 못하므로, 이미지 또는 시퀀스 데이터에 있어서 최선의 방법이 아님.
- 데이터의 양이 크게 증가함에 따라, 고전적인 방법들로는 대규모 데이터셋의 이상치를 찾는 것이 거의 불가능해졌음.
- DAD는 데이터로부터 계층적 변별 특성(hierarchical discriminative features)을 학습함. 이러한 딥러닝의 특징은 end-to-end 학습을 가능하게 하며, 전문가의 특성 공학 등 전처리의 필요성을 줄여주고, raw data를 그대로 입력할 수 있도록 함.
- 정상 동작과 비정상 동작 사이의 경계가 정확하게 정의되지 않은 경우가 많음. 이런 문제는 고전적인 방법과 딥러닝 기반 방법 모두 어려움을 가지게 함.

# Related Work

많은 머신러닝 문제에서 딥러닝 방법이 상당한 발전을 이루었지만, 이상 탐지 분야에서 딥러닝은 여전히 부족한 부분이 있음. 다양한 도메인에서의 이상 탐지를 위한 딥러닝 연구가 발표되었지만, 각 딥러닝 아키텍처에 대한 비교 분석은 부족한 실정임. 

# Our Contributions

1. DAD 기법에 대한 최신 연구와 실제 응용 사례에 대한 포괄적인 개요를 제공하고자 함. 
2. 최근 몇 년간 계산 비용을 크게 줄인 DAD 기법들이 개발되었음. 이러한 기법들을 조사하고, 이해를 돕기 위해 체계적인 카테고리로 분류하고자 함. 각 카테고리마다 필요한 가정과 기법, 한계점, 장점 및 단점, 계산 복잡성 등에 대한 정보를 제공하고자 함.

# Different aspects of deep learning-based anomaly detection

## Nature of Input Data

딥러닝 아키텍처의 선택은 주로 입력 데이터의 특징에 따라 달라짐. 입력 데이터는 크게 순차적 데이터(음성, 텍스트, 음악, 시계열, 단백질 구조 등)와 비순차적 데이터(이미지, 정형 데이터 등)로 나눌 수 있음. 또한, 데이터의 특성 수에 따라 저차원 데이터 혹은 고차원 데이터로 분류할 수 있음. DAD 기법은 고차원 데이터 내에서 복잡한 계층적 특성을 학습하도록 사용되어 왔음. DAD에 사용되는 layer의 개수는 입력 데이터 차원에 따라 결정되며, 네트워크가 깊을수록 고차원 데이터에서 더 나은 성능을 발휘함. 

## Based on Availability of labels

레이블은 해당 데이터가 정상인지 이상치인지 나타냄. 이상치는 드물게 발생하므로, 데이터를 충분히 얻기 어려움. 또한, 이상 행동의 정의가 시간에 따라 바뀌기 때문에 이상 탐지에서의 어려움이 존재함. DAD 기법은 레이블의 가용성에 따라 1) 지도 학습 기반, 2) 준지도 학습 기반, 3) 비지도 학습 기반으로 구분할 수 있음.

### Supervised deep anomaly detection

지도 학습 기반 DAD에서는 정상 데이터와 비정상 데이터의 레이블을 사용하여 이진 혹은 다중 분류를 수행함. 지도 학습 기반 모델의 좋은 성능에도 불구하고, 레이블이 지정된 데이터를 구하기 쉽지 않아 준지도 학습 또는 비지도 학습만큼 널리 사용되지 않음. 또한, 데이터에 존재하는 클래스 불균형 문제는 지도 학습 기반 방법의 적용을 어렵게 함. 따라서, 본 논문에서는 지도 학습 기반 DAD를 고려하지 않음. 

###  Semi-supervised deep anomaly detection

정상 데이터는 이상 데이터보다 훨씬 쉽게 구할 수 있으므로, 준지도 학습 기반 방법은 더 널리 채택됨. 이 방법은 단일 레이블을 활용하여 이상치를 구분함. 

이상 탐지에서 오토인코더를 사용하는 일반적인 방법 중 하나는 이상 데이터가 없는 정상 데이터로만 오토인코더를 훈련시키는 것임. 모델이 잘 훈련되었다면, 오토인코더는 비정상 데이터에 비해 정상 데이터에서 낮은 재구성 오류를 생성함. 

### Unsupervised deep anomaly detection

비지도 학습 기반 DAD는 데이터의 내재적 특징을 기반으로 이상치를 탐지함. 레이블이 적용된 데이터를 구하기 어렵기 때문에, 데이터 자동 레이블링에 비지도 학습 기반 DAD가 사용됨. 

## Based on the training objective

또한, 본 논문에서는 training objective에 따른 새로운 두 가지의 카테고리(DHM, OC-NN)를 제안함. 본 논문에서 DAD 기법을 분류하는 4가지의 카테고리는 다음과 같음.

**Type of Models**

- Semi-supervised
- Unsupervised
- Deep hybrid models (DHM)
- One-Class Neural Networks  (OC-NN)

###  Deep Hybrid Models (DHM)

DHM은 딥러닝 모델을 (주로 오토인코더) 사용하여 특성을 추출하고, 추출된 특성을 One-Class SVM, SVDD 등 기존 이상 탐지 알고리즘에 입력함. 다양한 도메인에서 pre-trainied 모델이 좋은 성능을 보였으며, DHM에서 또한 pre-trainied 모델을 특징 추출기로 사용하여 큰 성공을 거두었음.

이러한 하이브리드 접근법에 주요한 단점은 이상 탐지에 맞춤화된 목표 함수가 없다는 것이며, 이러한 이유로 이상치를 탐지하기 위한 풍부한 특징을 추출하지 못함. 이러한 한계점을 해결할 수 있도록, Deep one-class classification 또는 One class neural networks 등이 도입되었음.

### One-Class Neural Networks (OC-NN)

One class neural network (OC-NN) Chalapathy et al. [2018a] methods are inspired by kernel-based one-class classification which combines the ability of deep networks to extract a progressively rich representation of data with the one-class objective of creating a tight envelope around normal data.
OC-NN은 커널 기반의 one-class classification에서 영감을 얻은 것으로, 풍부한 데이터 representation을 추출하는 딥러닝 네트워크의 성능과 정상 데이터에 대한 경계를 생성하는 one-class의 objective를 결합한 것임.

hidden layer의 data representation이 OC-NN의 목적 함수에 의해 주도되며, 이상 탐지에 맞게 맞춤화됨.

OC-NN의 변형인 Deep SVDD는 정상 데이터 포인트를 구의 중심에 가깝게 매핑하여 common factors of variation을 추출할 수 있도록 DNN을 훈련함. 

## Type of Anomaly

이상 징후는 크게 세 가지로 분류할 수 있음.

- point anomalies
- contextual anomalies
- collective anomalies

### Point Anomalies

대부분의 문헌 연구는 point anomalies에 초점을 맞추고 있음. point anomalies는 종종 무작위로 발생하는 불규칙성 또는 편차를 나타내며 특별한 해석이 없을 수도 있음. 

예를 들어, 신용 카드 지출 내역 중 다른 거래 내역과 달리 매우 큰 지출은 point anomaly로 해석할 수 있음. 

### Contextual Anomaly Detection

contextual anomaly는 conditional anomaly라고도 하며, 특정 문맥에서 이상 징후로 간주될 수 있는 데이터를 뜻함. contextual anomaly는 contextual, behavioural features 모두 고려해야 함. 주로 사용되는 문맥적인 특징은 시간과 공간임. 예를 들어, 현재 기온이 32도라는 데이터 자체는 이상치가 아니지만, 해당 시간이 겨울일 경우 이상 징후로 간주할 수 있음. 

###  Collective or Group Anomaly Detection

개별 데이터 포인트의 비정상적인 모음을 collective 또는 group anomalies라고 하며, 개별 포인트 각각을 분리하면 정상 데이터로 나타나지만 그룹으로 관찰할 경우 비정상적인 특성을 나타내는 것을 뜻함. 

예를 들어 특정 마트에서 75달러의 거래가 짧은 간격 내 연속적으로 나타났다고 하면, 하나의 75달러 거래는 이상 징후가 아니지만, 연속적으로 나타나는 75달러의 거래들은 collective anomaly로 판단할 수 있음.

##  Output of DAD Techniques

일반적으로 이상 징후 탐지 방법에서 생성되는 출력은 anomaly score 또는 binary labels임. 

### Anomaly Score

anomaly score는 각 데이터 포인트의 이상치 수준을 나타냄. 이상 징후를 식별하기 위해 도메인에 특화된 임계값을 설정함. 일반적으로 이러한 점수는 이진 레이블보다 더 많은 정보를 제공함. 

### Labels 

점수를 할당하는 대신, 일부 방법은 데이터에 정상 또는 비정상이라는 레이블을 할당함. 예를 들어 오토인코더를 사용하는 경우, 재구성 오류를 계산한 뒤 설정된 임계값에 따라 데이터에 레이블을 지정함. 

#  Applications of Deep Anomaly Detection

다양한 도메인에서의 DAD를 논의함. 각 도메인별로 1) 이상 징후의 개념, 2) 데이터의 특징, 3) 이상 탐지와 관련된 과제, 4) 존재하는 기존의 DAD 방법에 대해 알아보고자 함. 

##  Intrusion Detection

intrusion detection system(침입 탐지 시스템, IDS)은 컴퓨터 관련 시스템에서 악의적인 활동을 식별하는 것임. 단일 컴퓨터에 대한 Host Intrusion Detection(HIDS), 대규모 네트워크에 대한 Network Intrusion Detection(NIDS) 등이 존재함. 

탐지 방법에 따라 IDS는 signature-based 또는 anomaly based로 분류됨. signature-based IDS는 신종 공격을 탐지하는데 효율적이지 않으므로, anomaly-based 방법이 더 많이 사용되고 있음. 

### Host-Based Intrusion Detection Systems (HIDS)

HIDS는 단일 호스트 또는 컴퓨터에서 발생하는 시스템 호출 또는 이벤트를 수신하여 악의적인 활동이나 정책 위반을 모니터링하는 소프트웨어 프로그램임. 시스템 호출 로그는 프로그램 또는 user interaction에 의해 생성되며, 악의적인 interaction은 이러한 시스템 호출을 서로 다른 순서로 실행하게 함. 

또한 HIDS는 시스템 상태와 저장된 정보, RAM, 파일 시스템, 로그 파일 등에서 유효한 시퀀스가 있는지 모니터링 할 수 있음. 

이러한 특징에 맞게, HIDS에 적용되는 DAD는 가변적인 길이를 가지고 순차적인 특징을 가진 데이터를 처리할 수 있어야 함. DAD는 시퀀스 데이터를 모델링하거나, 시퀀스 간 유사성을 계산함. 

### Network Intrusion Detection Systems (NIDS)

NIDS는 모든 네트워크 패킷을 검사하여 전체 네트워크에서 의심스러운 트래픽을 모니터링함. 실시간 스트리밍 동작으로 인해, 데이터는 매우 크고, 다양하며, 빠른 속도가 생성됨. 또한, 네트워크 데이터는 시간적인 측면도 존재함. 

IDS에서 DAD가 직면한 과제는 침입자가 기존 침입 탐지 솔루션을 회피하기 위해 이상 행위의 특성이 시간이 지남에 따라 계속 변한다는 것임. 

## Fraud Detection

Fraud(사기) Detection은 다양한 산업 분야에서 불법 활동을 탐지하는 것임. 사기 탐지에서의 주요 과제는 실시간 탐지와 예방이 필요하다는 점임. 

### Banking fraud

신용카드 사기는 카드 정보를 도용하여 거래에 사용하는 것을 말함. 신용카드 사기 탐지의 어려움은 사기에 일관적인 패턴이 없다는 것임. 신용카드 사기 탐지의 일반적인 접근 방식은 각 사용자에 대한 유저 프로필을 유지하고, 유저 프로필을 모니터링하여 편차를 탐지하는 것임. 

신용카드의 사용자가 수십억 명에 달하기 때문에, 이러한 방법은 적절하지 않을 수 있음. DAD 특유의 scalable nature로, DAD는 신용카드 사기 탐지에 자주 채택됨.

### Mobile cellular network fraud

모바일 셀룰러 네트워크는 매우 빠르게 발전되었으며, 모바일 셀룰러 네트워크는 이제 고객의 개인 정보를 훔치기 위한 음성 사기, 고객의 돈을 갈취하기 위한 메시징 관련 사기 등에 직면해 있음. 모바일 셀룰러 네트워크의 규모와 속도 때문에, 이러한 사기를 탐지하는 것은 쉬운 일이 아님. 기존의 머신러닝 방법은 진화하는 사기의 특성에 적응하지 못함. 

### Insurance fraud

몇 가지 고전적인 머신러닝 알고리즘은 보험금 청구에서 사기를 탐지하는 것에 성공적으로 적용되었음. 기존의 방법은 사기와 관련된 몇 가지의 특성을 기반으로 사기를 예측하였는데, 이러한 접근 방식의 문제점은 특성을 추출 또는 생성하기 위해 전문 지식과 수작업이 필요하다는 것임. 또한, 사기의 발생률이 전체 보험금 청구 건수에 비해 훨씬 적으며, 각 사기마다 고유의 방식이 있다는 문제가 있음. 

### Healthcare fraud

의료 보험 청구 사기는 의료 비용 증가의 주요 원인이며, 사기 탐지를 통해 그 영향을 완화할 수 있음. 여러 머신러닝 모델이 의료 보험 사기에 효과적으로 사용되어 왔음. 

##  Malware Detection

악성 코드로부터 정상 유저를 보호하기 위해, 머신러닝 기반의 멀웨어 탐지 방법이 제안되고 있음. 기존의 머신러닝 방법에서 악성 코드 탐지 과정은 일반적으로 특징 추출과 분류/클러스터링으로 이루어짐. 이러한 기존의 탐지 방법의 성능은 추출된 특징과 분류/클러스터링 방법에 따라 크게 달라짐.

악성 코드 탐지에서의 주요 과제는 데이터의 엄청난 규모임. 또한, 악성 코드는 매우 적응력이 뛰어나 공격자가 악의적인 동작을 숨기기 위해 다양한 기술을 사용할 수 있음. 

## Medical Anomaly Detection

의료 및 bio-informatics에서 딥러닝의 이론적, 실제적 응용을 위한 여러 연구가 수행되었음. 의료 영상 분석, 임상 뇌파 기록과 같은 영역에서 이상 징후를 찾아내어 다양한 질병을 진단하고 예방 치료를 제공할 수 있음. 의료 분야는 방대한 양의 불균형 데이터로 인해 이상 징후를 탐지하는 것에 어려움이 있음. 

추가로, 딥러닝 기법은 오랫동안 블랙박스 기법으로 간주되어 왔음. 이러한 딥러닝 모델들은 성능이 좋을지는 몰라도, 해석 능력이 부족함. 의료 분야의 딥러닝은 성능과 함께 해석 능력이 중요함. 

## Deep learning for Anomaly detection in Social Networks

소셜 네트워크의 이상 징후는 소셜 네트워크 내 개인의 불규칙하고 불법적인 행동으로, 스팸 발송자 / 성범죄자 / 온라인 사기꾼 / 가짜 사용자 / 루머 유포자 등으로 식별될 수 있음. 소셜 네트워크 특유의 이질적이고 동적(dynamic)인 특성은 DAD 기법에 상당한 어려움이 됨. 

## Log Anomaly Detection

로그 파일에서 이상 징후 탐지는 시스템 장애의 원인과 특징을 나타낼 수 있는 텍스트를 찾는 것을 목표로 함. 일반적으로는 도메인별 정규식을 만들어 패턴 매칭을 통해 새로운 결함을 찾아냄. 이러한 접근 방식의 한계는 새로운 장애 메시지를 쉽게 감지하지 못한다는 것임. 

로그 데이터의 형태와 의미가 모두 비정형적이고 다양하기 때문에 이상 탐지에 상당한 어려움이 있음. 이상 탐지 기술은 동시에 생성되는 데이터 집합에 적응하고, 실시간으로 이상 징후를 탐지해야 함. 몇 가지 DAD 기법은 로그 데이터를 자연어 시퀀스로 모델링하여 이상치를 탐지하는 데 매우 효과적인 것으로 알려짐. 


## Internet of things (IoT) Big Data Anomaly Detection

IoT 네트워크에서 이상 징후 탐지는 방대한 규모의 상호 연결된 디바이스에서 발생하는 이상 징후를 식별함. 이상 탐지에서의 과제는 서로 다른 디바이스가 서로 연결되어 시스템을 더욱 복잡하게 만든다는 것임. IoT 영역에서 분석 및 학습을 용이하게 하기 위해 딥러닝을 사용하는 방법이 연구되었음. 

## Industrial Anomalies Detection

풍력 터빈, 발전소, 고온 에너지 시스템 등의 산업 시스템은 일상적으로 엄청난 stress에 노출됨. 이러한 유형의 시스템에 손상이 발생하면 많은 손실이 발생하므로, 이를 조기에 감지하고 수리하는 것이 중요함. 

장비에 대한 손상은 드물게 발생하므로 이러한 이벤트를 감지하는 것은 이상 탐지 문제로 공식화할 수 있음. 산업 시스템 영역에서의 과제는 다양한 요인으로 인해 고장이 발생한다는 것과, 데이터의 양이 많다는 점임. 

## Anomaly Detection in Time Series

일정 기간 동안 지속적으로 기록되는 데이터를 시계열이라 하며, 크게 단변량 시계열과 다변량 시계열로 분류할 수 있음. 시계열에서의 이상치 유형 또한 Point, Contextual, Collective Anomalies로 분류할 수 있음. 시계열에서 이상 징후를 탐지할 때의 어려움은 다음과 같음. 

- 이상 징후가 발생하는 패턴이 정의되어 있지 않을 수 있음.
- 입력 데이터의 노이즈는 알고리즘의 성능에 영향을 미침.
- 시계열 데이터의 길이가 길어질수록 계산 복잡도가 증가함.
- 시계열 데이터는 일반적으로 non-stationary, non-linear, dynamically evolving 특성을 가짐. DAD 모델은 실시간으로 이상 징후를 감지할 수 있어야 함. 

## Video Surveillance

비디오 감시는 보안을 위해 관심 영역을 모니터링하는 것임. 비디어 감시 애플리케이션에는 레이블이 지정되지 않은 대량의 데이터가 있으며, 이것은 머신러닝 기반 방법에서의 어려움이 될 수 있음. 실제 비디오 감시에서 이상 징후에 대한 명시적인 정의가 없다는 것 또한 DAD 적용에 있어 중요한 문제임.  


# Deep Anomaly Detection (DAD) Models

이 섹션에서는 다양한 DAD 모델과, 각 모델의 가정, 구조, 계산 복잡도, 장단점 등을 알아보고자 함. 

## Supervised deep anomaly detection

지도 학습 기반 이상 탐지 기법은 레이블이 있는 샘플을 사용하기 때문에, 비지도 학습 기반 이상 탐지 기법보다 성능이 우수함. 지도 학습 기반 이상 탐지는 데이터에서 결정 경계를 학습한 다음, 학습한 모델을 사용하여 새로운 데이터를 정상 또는 비정상 클래스로 분류함. 

**Assumptions**
지도 학습 기반 방법은 데이터를 분리하는 데 중점을 두고, 비지도 학습 기반 방법은 데이터의 특성을 이해하고 설명하는 데 중점을 둠. 일반적으로, 이상 탐지를 위한 딥러닝 모델은 두 개의 하위 네트워크(특성 추출과 분류기)로 구성됨. 

딥러닝 모델은 다양한 데이터 포인트를 효과적으로 판별하기 위해 상당한 수의 훈련 데이터를 필요로 함. 레이블이 달린 clean 데이터의 부족으로 인해, 지도 학습 기반 방법은 준지도 학습 기반 방법이나 비지도 학습 기반 방법만큼 사용되지 않음. 

**Computational Complexity**
지도 학습 기반 딥러닝 모델의 계산 복잡도는 입력 데이터의 차원과 역전파 알고리즘을 사용하여 학습되는 히든 레이어의 수에 따라 달라짐. 고차원의 데이터는 의미 있는 정보의 학습을 위해 더 많은 히든 레이어를 갖는 경향이 있음. 히든 레이어의 수가 증가할수록 계산 복잡도는 선형적으로 증가하며, 모델 훈련과 업데이트에 더 많은 시간이 필요함.  

**Advantages and Disadvantages**
supervised DAD의 장점은 다음과 같음.

- supervised DAD는 준지도 / 비지도 학습 기반 DAD보다 더 정확할 수 있음.
- test instance를 미리 계산된 모델과 비교하면 되므로, test phase에 필요한 시간이 적음(빠르게 판별할 수 있음). 

supervised DAD의 단점은 다음과 같음.

- 다중 클래스 분류의 경우 다양한 클래스에 대한 정상, 비정상 데이터가 필요하며, 이러한 데이터를 구하기 어려움.  
- 데이터의 특성 공간이 매우 복잡하고 비선형적인 경우 정상 데이터와 비정상 데이터를 구분하는 것이 어려움. 

## Semi-supervised deep anomaly detection

준지도 학습 기반, 또는 one-class classification DAD는 모든 훈련 데이터가 하나의 클래스 레이블을 갖는다고 가정함. DA 기법은 정상 데이터를 중심으로 결정 경계를 학습함. 결정 경계에 속하지 않는 테스트 데이터는 이상 징후로 분류됨. 

**Assumptions**
준지도 학습 기반 DAD는 데이터 포인트를 이상 징후로 점수화하기 위해 다음과 같은 가정을 가짐. 

- Proximity and Continuity(근접성 및 연속성): 입력 공간과 학습된 특징 공간 모두에서 서로 가까운 점은 동일한 레이블을 가질 가능성이 높음.
- DNN의 hidden layer에서 강건한 특성이 학습되며, 이러한 특성은 정상 데이터와 비정상 데이터를 구분하는 discriminative attributes를 가짐. 

**Computational Complexity**
준지도 학습 기반의 계산 복잡도는 지도 학습 기반의 계산 복잡도와 유사함. 입력 데이터의 차원과 hidden layer에서의 수에 따라 계산 복잡도가 달라짐.

**Advantages and Disadvantages**
준지도 학습 기반 방법의 장점은 다음과 같음.

- 준지도 학습으로 훈련된 Generative Adversarial Networks (GANs)은 라벨링된 데이터가 매우 적은 경우에도 좋은 성능을 보여주었음.
- 레이블이 지정된 데이터(일반적으로 하나의 클래스)를 사용하면 비지도 학습 기반의 방법보다 성능 향상을 가져올 수 있음. 

다만, 준지도 학습의 근본적인 단점(Tyler Tian Lu. Fundamental limitations of semi-supervised learning. Master’s thesis, University of Waterloo, 2009.)은 딥러닝 맥락에서도 적용 가능하며, 과대적합 문제가 발생하기 쉽다는 단점이 존재함.

## Hybrid deep anomaly detection

딥러닝 모델은 강건한 특성을 학습하기 위해 주로 사용됨. Deep hybrid 모델에서는 딥러닝 모델로 학습한 representative 특성을 RBF, SVM 등의 기존의 머신러닝 알고리즘에 입력함. 

**Assumptions**
DHM은 다음과 같은 가정을 가짐.

- DNN을 통해 학습된 강건한 특성이 추출되어 이상 징후를 숨길 수 있는 irrelevant 특성을 분리함에 도움을 줌. 
- 복잡하고 고차원의 공간에서 강건한 이상 탐지 모델을 구축하기 위해서는 특성 추출기와 이상 징후 탐지기가 필요함. 

**Computational Complexity**
DHM의 계산 복잡도에는 딥러닝 모델의 계산 복잡도와 기존 알고리즘(SVM 등)의 계산 복잡도가 모두 포함됨. 입력 차원 수가 $d$일 때, RBF 커널을 사용하는 SVM의 경우 approximation $O(d^2)$의 계산 복잡도가 고려됨.

**Advantages and Disadvantages**
DHM의 장점은 다음과 같음.

- 특히 고차원 데이터에서, 특성 추출기는 "차원의 저주"의 영향을 크게 줄여줌. 
- 선형 또는 비선형 커널 모델에 축소된 차원을 가진 데이터를 입력하기 때문에, 확장성과 계산 효율성이 높음. 

DHM의 단점은 다음과 같음.

- DHM은 이상 탐지를 위한 맞춤형 손실 함수가 아닌, 일반적인 손실 함수를 사용하므로 hidden layer에서의 표현 학습에 영향을 미칠 수 없음. 
- layer가 깊어질수록 더 나은 성능을 보이지만, 더 많은 계산 비용이 요구됨. 

## One-class neural networks (OC-NN) for anomaly detection

OC-NN은 모든 정상 데이터 포인트를 비정상 데이터에서 분리하는 hyperplane  또는 hypersphere와 같은 one-class의 목표 함수와 함께, 풍부한 데이터 representation을 추출하는 딥러닝의 기능을 결합한 것임. 

OC-NN은 이상 탐지를 위한 맞춤 목적 함수를 사용하여 최적화되고, 복잡한 데이터셋에서 기존의 SOTA 모델과 비슷하거나 더 나은 성능을 달성하면서도 기존 방법에 비해 합리적인 훈련 및 테스트 시간을 보여주었음.

**Assumptions**
OC-NN은 다음과 같은 가정을 바탕으로 이상 징후를 탐지함. 

- OC-NN은 DNN의 hidden layer를 통해 데이터 분포 내의 common factors of variation을 추출함. 
- combined representation learning을 수행하고 테스트 데이터에 대해 이상값 점수를 생성함. 
- 비정상 샘플은 common factors of variation을 가지지 않으므로, hidden layer는 이상치의 representation을 포착하지 못함. 

**Computational Complexity**
하이브리드 모델과 달리, OC-NN은 deep network의 계산 복잡도만 고려함. 훈련 시간은 입력 차원에 비례한다고 알려져 있음.

**Advantages and Disadvantages**
OC-NN의 장점은 다음과 같음.

- OC-NN 모델은 데이터를 둘러싸는 hyperplane 또는 hypersphere를 최적화하며 DNN을 함께 훈련함. 
- OC-NN은 파라미터를 학습하기 위한 alternating minimization algorithm을 제안하였음. OC-NN의 목적 함수의 하위 문제는 well defined 된 사분위수 선택 문제를 푸는 것과 동일하다는 것이 관찰되었음. 

OC-NN의 단점은 다음과 같음.

- 고차원의 데이터일 경우 학습 시간과 모델 업데이트 시간이 더 길어질 수 있음.
- 입력 공간이 변화할 경우, 모델 업데이트에 더 오랜 시간이 소요됨. 

## Unsupervised Deep Anomaly Detection

비지도 학습 기반 DAD는 머신러닝 연구와 산업 애플리케이션 모두에서 필수적인 연구 분야임. 오토인코더는 이상 징후 탐지에 사용되는 기본적인 비지도 deep architectures임.

**Assumptions**
비지도 학습 기반 DAD는 다음과 같은 가정을 가짐.

- 원본 또는 잠재 공간의 정상 영역은 원본 또는 잠재 공간의 비정상 영역과 구별될 수 있음.
- 데이터셋 내의 대부분의 데이터는 정상 데이터임. 
- 비지도 학습 기반 이상 탐지 알고리즘은 거리나 밀도 같은 데이터셋의 내재적인 속성을 사용하여 이상값 점수를 생성함. DNN의 hidden layer의 목표는 데이터셋의 이러한 내재적 속성을 포착하는 것임.

**Computational Complexity**
오토인코더는 이상 탐지에서 가장 자주 사용되는 구조이며, quadratic cost를 가지고 다른 신경망 구조와 같이 non-convex optimization problem를 가짐. 

모델의 계산 복잡도는 연산의 수, 파라미터의 수, hidden layer의 수에 따라 달라짐. PCA와 같은 기존 방법보다는 훨씬 높은 계산 복잡도를 가짐. 

**Advantages and Disadvantages**
비지도 학습 기반 DAD의 장점은 다음과 같음.

- 데이터 고유의 특성을 학습하여 정상과 비정상을 구분함. 이러한 기법은 데이터 내의 공통점(commonalities)을 식별하고 이상치 탐지를 용이하게 함. 
- 레이블이 달린 데이터가 필요하지 않으므로, 비용면에서 효율적임. 
 
비지도 학습 기반 DAD의 단점은 다음과 같음.

- 복잡하고 고차원의 데이터에서는 데이터 내 공통점을 학습하기 어려움. 
- 오토인코더를 사용할 경우, 축소되는 차원 등 하이퍼파라미터를 직접 조정해야 함. 
- 비지도 학습 기반 방법의 경우, 노이즈와 데이터 손상(corruptions)에 매우 민감하여 지도 학습 또는 준지도 학습 기반의 방법보다 정확도가 떨어지는 경우가 많음. 

## Miscellaneous Techniques

이번 절에서는 위에서 소개하지 못했던 다양한 DAD 기법들을 추가로 살펴봄.

### Transfer Learning based anomaly detection

딥러닝은 좋은 결과를 얻기 위해 충분히 많은 데이터가 필요함. 전이 학습은 훈련 데이터의 부족이라는 근본적인 문제를 해결할 수 있는 방법임. 

The open research questions using transfer learning for anomaly detection is, the degree of transfer-ability, that is to define how well features transfer the knowledge and improve the classification performance from one task to another. 
이상 탐지를 위한 전이 학습에서의 open research questions은 특성이 knowledge를 얼마나 잘 전이하고 분류 성능을 향상시키는지를 정의하는 것임.

### Zero Shot learning based anomaly detection

Zero shot learning (ZSL)은 이전에 본 적이 없는 물체를 인식하는 것을 목표로 함. ZSL은 두 단계로 이루어지는데, 먼저 자연어, 메타 데이터 등으로 이루어진 데이터의 knowledge를 포착하고, 이 knowledge를 사용하여 새로운 인스턴스를 분류하는 것임. 현실 세계에서 모든 클래스의 이미지를 얻기 어려우므로, 이러한 방법은 중요한 접근임. 

이러한 접근 방식에서 관련된 주요 과제는 데이터에 대한 메타 데이터를 확보하는 것임. 최근, 이상 탐지에서 ZSL을 사용하여 최첨단의 결과를 얻을 수 있었음. 

### Ensemble based anomaly detection

DNN의 문제 중 하나는 입력 데이터의 노이즈에 민감하다는 것과, 좋은 성능을 위해 매우 많은 데이터가 필요하다는 것임. 노이즈가 많은 데이터에서 강건성을 달성하기 위해 오토인코더의 연결 아키텍처를 무작위로 변경하는 방법은 훨씬 더 나은 성능을 얻을 수 있음. 이렇게 무작위로 연결되어 다양한 오토인코더로 구성된 오토인코더 앙상블은 여러 데이터셋에서 좋은 결과를 얻었음. 

###  Clustering based anomaly detection

클러스터링은 추출된 특성을 기반으로 유사한 패턴을 그룹화하여 새로운 이상치를 탐지하는 것임. 클러스터링할 클래스의 수에 따라 시간 및 공간 복잡성이 증가하므로, 클러스터링 기반 이상 탐지는 실시간 애플리케이션에 적용하기에는 한계가 있음. 

word2vec 모델을 사용하여 정상 데이터와 비정상 데이터를 의미론적으로 표현하여 클러스터를 형성하고 이상 징후를 탐지하는 방법이 존재함. 오토인코더, 하이브리드 모델 등을 사용하여 클러스터링을 위한 representative 특성을 얻음. 

### Deep Reinforcement Learning (DRL) based anomaly detection

DRL 방법은 고차원 데이터 공간에서 복잡한 특징을 학습할 수 있는 능력으로 인해 큰 관심을 모으고 있음. DRL 기반 이상 탐지 모델은 이상 징후에 대한 가정을 고려하지 않으며, 축적된 reward signals를 통해 지속적으로 knowledge를 강화하여 새로운 이상 징후를 식별함. DRL 기반 이상 탐지 기술은 매우 새로운 개념이며, 추가적인 조사가 필요함. 

### Statistical techniques deep anomaly detection

Hilbert 변환은 실수값 신호의 analytic representation을 도출하는 통계적 신호 처리 기법임. 웨이블릿 분석, 신경망과 Hilbert 변환 기능을 결합하여 실시간 이상 징후를 탐지하는 알고리즘이 존재함. 통계 기법을 응용한 DAD에 대해서는 추가적인 조사가 필요함. 

# Deep neural network architectures for locating anomalies

## Deep Neural Networks (DNN)

딥 아키텍처는 확장성, 데이터 내 새로운 변형에 대한 일반화, 수작업 피처 엔지니어링의 필요성이라는 기존 머신러닝 접근 방식의 한계를 극복함. Deep Belief Networks (DBNs)는 Restricted Boltzmann Machine (RBMs)으로 알려진 DNN의 한 종류임. 

오토인코더를 사용한 이상 탐지와 유사하게, 비정상 샘플의 재구성 오류를 기반으로 이상치를 탐지할 수 있음. DBN은 빅데이터에 효율적으로 확장하고 해석 가능성을 개선하는 것으로 나타났음.

## Spatio Temporal Networks (STN)

Spatio Temporal Networks (STNs)는 시공간적 특징을 추출하기 위해 CNN과 LSTM을 모두 결합한 구조를 가짐. STN의 시간적 특징(LSTM을 통해 가까운 시점 간의 상관관계를 모델링)과 공간적 특징(CNN을 통해 local spatial 상관관계를 모델링)은 이상값을 탐지하는 데 효과적인 것으로 나타났음.

## Sum-Product Networks (SPN)

Sum-Product Networks (SPNs) leaves를 변수로 하는 directed acyclic graphs이며, 내부 노드와 가중치가 있는 에지가 sum과 product를 구성함. SPN은 빠르고 정확한 확률적 추론이 가능한 mixture models의 조합으로 간주됨. SPN의 가장 큰 장점은 다른 graphical models과 달리, 근사 추론이 없어도 traceable 하다는 것임. 

SPN은 입력의 불확실성을 포착하여 강건하게 이상 징후를 탐지함. 다양한 데이터셋에서 인상적인 결과를 보여주었지만, 이상 탐지 관련해서 연구해야 할 부분이 많이 남아 있음. 

## Word2vec Models 

Word2vec은 단어 임베딩을 생성하는 데 사용되는 DNN 모델임. 모델은 문장, 시간 순서 데이터와 같은 데이터 인스턴스 내의 순차적 관계를 캡처할 수 있음. 단어 임베딩 특징을 입력하는 것은 여러 딥러닝 구조에서 성능을 향상시키는 것으로 나타났음. 이상 탐지 또한 word2vec 임베딩을 활용하는 경우 성능이 크게 향상되었음.

## Generative Models

Generative Models은 새로운 데이터 포인트를 생성하기 위해 정확한 데이터 분포를 학습하는 것을 목표로 함. 가장 일반적이고 효율적인 접근 방식은 VAE와 GAN임. 

GAN의 변형 중 하나인 Adversarial autoencoders (AAE)은 adversarial training을 사용하여 잠재 변수에 arbitrary prior를 부과하고, 이를 통해 입력 분포를 효과적으로 학습함. 이러한 입력 분포 학습 능력을 활용하여 여러 가지 GAN 기반 이상 탐지가 고차원의 복잡한 데이터에서 효과적인 것으로 나타났음. 

그러나 이상 징후가 적은 경우, KNN과 같은 기존의 고전적인 머신러닝 방법이 생성 모델에 비해 더 나은 성능을 발휘하는 것으로 보고되었음. 

## Convolutional Neural Networks

복잡한 구조를 가진 고차원 데이터에서 복잡하고 숨겨진 특징을 추출하는 CNN의 능력을 기반으로, CNN을 시퀀스 및 이미지 데이터셋 모두에 대한 이상 탐지에서 특징 추출기로 사용할 수 있음. 이상 탐지를 위한 CNN 기반 프레임워크에 대한 평가는 현재도 활발히 연구되고 잇음. 

## Sequence Models

기본 RNN은 시퀀스의 길이가 길어질수록 데이터의 context를 제대로 포착하지 못하였고, 이러한 문제를 해결하기 위해 LSTM, GRU 등이 도입되었음. 시퀀스 데이터의 이상 탐지에서, LSTM 기반 신경망은 기존의 방법보다 상당한 성능 향상을 가져오는 것으로 보고되었음. 

## Autoencoders

정상 데이터로만 훈련된 오토인코더는 비정상 데이터 샘플을 재구성하지 못하므로, 재구성 오류가 크게 발생함. 높은 재구성 오류를 생성하는 데이터는 이상치로 간주됨. CAE, LSTM-AE, DAE 등 오토인코더의 다양한 변종들이 존재하며 데이터의 특징에 맞게 모델을 선택해야 함.

인코더 네트워크는 CNN으로, 디코더 네트워크는 LSTM으로 구성하는 등의 다양한 오토인코더 연구가 진행되고 있음. 오토인코더는 이상 탐지를 위한 간단하고 효과적인 구조지만, 훈련 데이터의 노이즈로 인해 성능이 저하될 수 있음. 

# Relative Strengths and Weakness : Deep Anomaly Detection Methods

이상 탐지 기법에는 각각 고유한 장단점이 존재함. 어떤 이상 탐지 기법이 주어진 문제 상황에 가장 적합한지 이해해야 함. 

지도 학습 기반 DAD는 레이블된 데이터가 많을 때 좋은 선택임. 지도 학습 또는 준지도 학습 기반 모델의 경우 훈련에는 오랜 시간이 걸리지만, 일반적으로 테스트에 걸리는 시간은 짧음. 레이블된 데이터를 구하는 것이 어렵기 때문에, 비지도 학습 기반 이상 탐지는 널리 사용되고 있음. 

하이브리드 모델은 딥러닝 모델을 사용하여 특징을 추출하고, 추출한 특징을 고전적인 머신러닝 알고리즘에 입력함. 하이브리드 기반 접근 방법은 hidden layer의 표현 학습에 영향을 줄 수 없기 때문에 최선의 선택은 아님. 

OC-NN은 딥러닝의 풍부한 특징 추출 기능과 정상 데이터와 비정상 데이터를 구분하는 경계(one-class의 목표)를 결합한 것임. 이 새로운 구조에 대해서는 추가적인 연구가 필요함. 

# Conclusion

본 논문을 통해, 딥러닝 기반 이상 징후 탐지에 대한 다양한 연구 방법과 다양한 영역에서의 적용 사례에 대해 알아보았음. 딥러닝 기반 이상 징후 탐지는 여전히 활발하게 연구되고 있으며, 향후 더 정교한 기법이 제안됨에 따라 이 조사를 확장하고 업데이트 할 수 있을 것임.  