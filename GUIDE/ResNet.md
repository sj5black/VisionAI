# Attention 메커니즘
- 시퀀스 데이터에서 중요한 부분에 더 많은 가중치를 할당하여 정보를 효율적으로 처리하는 기법
- 주로 자연어 처리(NLP)와 시계열 데이터에서 사용되며, 기계 번역, 요약, 질의응답 시스템 등 다양한 분야에서 뛰어난 성능을 발휘

1. **개요**
    - Attention 메커니즘은 입력 시퀀스의 각 요소에 대해 중요도를 계산하여 가중치를 부여
    - 이를 통해 중요한 정보에 집중하고, 불필요한 정보 무시
    - Attention 메커니즘의 구성 요소 : Query, Key, Value.
2. **Attention 스코어 계산**
    - Attention 스코어는 Query와 Key 간의 유사도를 측정하여 중요도 계산
    - 이 유사도는 내적(dot product) 등을 사용하여 계산 가능
    - $\text{score}(Q, K) = Q \cdot K^T$
3. **Softmax를 통한 가중치 계산**
    - 계산된 Attention 스코어는 Softmax 함수를 통해 확률 분포로 변환
    - 이를 통해 가중치의 합이 1이 되도록 설정
    - $\alpha_i = \frac{\exp(\text{score}(Q, K_i))}{\sum_{j} \exp(\text{score}(Q, K_j))}$
4. **Softmax를 통한 가중치 계산**
    - Softmax를 통해 얻어진 가중치를 Value에 곱하여 최종 Attention 출력을 계산

### **Self-Attention**

- 시퀀스 내의 각 요소가 서로를 참조하는 메커니즘. 입력 시퀀스의 모든 요소가 Query, Key, Value로 사용
- 이를 통해 각 요소가 시퀀스 내 다른 요소들과의 관계를 학습
- 문장 내에서 단어 간의 관계를 학습하여 **번역이나 요약에 활용** 가능

### **Multi-Head Attention**

- 여러 개의 Self-Attention을 병렬로 수행하는 메커니즘
- 각 헤드는 서로 다른 부분의 정보를 학습하며, 이를 통해 모델이 다양한 관점에서 데이터 처리


### Attention 구현
 - **Scaled Dot-Product Attention**
```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V):
    d_k = Q.size(-1)  # Key의 차원 수
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))  # 유사도 계산 및 스케일링
    attn_weights = F.softmax(scores, dim=-1)  # Softmax를 통한 가중치 계산
    output = torch.matmul(attn_weights, V)  # 가중합을 통한 최종 출력 계산
    return output, attn_weights
```

 - **Multi-Head Attention**

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask=None):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Linear transformations
        values = self.values(values).view(N, value_len, self.heads, self.head_dim)
        keys = self.keys(keys).view(N, key_len, self.heads, self.head_dim)
        queries = self.queries(query).view(N, query_len, self.heads, self.head_dim)

        # Scaled dot-product attention
        out, _ = scaled_dot_product_attention(queries, keys, values)

        out = out.view(N, query_len, self.heads * self.head_dim)
        out = self.fc_out(out)
        return out
```

# 자연어 처리 모델 (NLP)

### 워드 임베딩(Word Embedding)
 - 단어를 고정된 크기의 벡터로 변환하는 기법으로, 단어 간의 의미적 유사성을 반영

1. **Word2Vec**
- 단어를 벡터로 변환하는 두 가지 모델(CBOW와 Skip-gram)을 제공
    - **CBOW (Continuous Bag of Words)**: 주변 단어(context)로 중심 단어(target)를 예측
    - **Skip-gram**: 중심 단어(target)로 주변 단어(context)를 예측

2. **GloVe (Global Vectors for Word Representation)**
- 단어-단어 공기행렬(word-word co-occurrence matrix)을 사용, 단어 벡터를 학습
- 전역적인 통계 정보를 활용하여 단어 간의 의미적 유사성 반영

<img src="./images/word_embedding.png" style="width:25%; height:auto;display: block; margin: 0 auto;">

### 시퀀스 모델링
 - 순차적인 데이터를 처리하고 예측하는 모델링 기법
 - 주로 RNN, LSTM, GRU와 같은 순환 신경망을 사용

<img src="./images/Sequence_Modeling.png" style="width:20%; height:auto;display: block; margin: 0 auto;">

&nbsp;

### Transformer

<img src="./images/Trans_BERT.png" style="width:50%; height:auto;display: block; margin: 0 auto;">

**인코더 (Encoder)**
- 입력 시퀀스를 처리하여 인코딩된 표현을 생성
- 각 인코더 층은 셀프 어텐션(Self-Attention)과 피드포워드 신경망(Feed-Forward Neural Network)으로 구성

**디코더 (Decoder)**

- 인코딩된 표현을 바탕으로 출력 시퀀스를 생성
- 각 디코더 층은 셀프 어텐션, 인코더-디코더 어텐션, 피드포워드 신경망으로 구성

**어텐션 메커니즘 (Attention Mechanism)**

- 어텐션 메커니즘은 입력 시퀀스의 각 위치에 가중치를 부여하여, 중요한 정보를 강조
- 셀프 어텐션은 입력 시퀀스 내의 단어 간의 관계를 학습

### BERT
 - Transformer 인코더를 기반으로 한 사전 학습된 언어 모델
 - 양방향으로 문맥을 이해할 수 있어, 다양한 자연어 처리 작업에서 뛰어난 성능

 1. **사전 학습(Pre-training)**

- 대규모 텍스트 코퍼스를 사용하여 사전 학습
- 마스킹 언어 모델(Masked Language Model)과 다음 문장 예측(Next Sentence Prediction) 작업을 통해 학습

 2. **파인튜닝 (Fine-tuning)**

- 사전 학습된 BERT 모델을 특정 작업에 맞게 파인튜닝
- 텍스트 분류, 질의 응답, 텍스트 생성 등 다양한 자연어 처리 작업에 적용

---
# ResNet (Residual Network)
- 잔차 학습(Residual Learning)을 기반으로 한 딥러닝 신경망 구조로, 딥 뉴럴 네트워크에서 발생하는 기울기 소실(vanishing gradient) 문제를 해결하기 위해 제안된 아키텍처
- 잔차 학습 : 각 층에서 새로운 출력을 계산하면서, 이전 층의 입력을 그대로 다음 층에 더해 학습하는 방식. (기울기 소실 문제 해결)
- 이미지 처리, 자연어 처리 등에 효과적

$$ output=F(x)+x $$

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(Block, self).__init__()
        # 첫 번째 컨볼루션 레이어
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)  # 배치 정규화
        # 두 번째 컨볼루션 레이어
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)  # 배치 정규화

        # 입력과 출력의 차원이 다를 경우 shortcut 경로 정의
        self.skip_connection = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),  # 차원 맞추기 위한 1x1 컨볼루션
                nn.BatchNorm2d(out_ch)  # 배치 정규화
            )
        
    def forward(self, x):
        # 첫 번째 컨볼루션 + ReLU 활성화 함수
        output = F.relu(self.bn1(self.conv1(x)))
        # 두 번째 컨볼루션 후 배치 정규화
        output = self.bn2(self.conv2(output))
        # shortcut 경로 출력과 현재 블록의 출력 더하기
        output += self.skip_connection(x)
        # 최종 ReLU 활성화 함수 적용
        output = F.relu(output)
        return output

# ResNet 모델 정의
class CustomResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(CustomResNet, self).__init__()
        self.initial_channels = 64  # 첫 번째 레이어의 입력 채널 수 정의
        # 첫 번째 컨볼루션 레이어
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)  # 배치 정규화
        # ResNet의 각 레이어 생성
        self.layer1 = self._create_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._create_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._create_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._create_layer(block, 512, layers[3], stride=2)
        # 평균 풀링 레이어
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 최종 완전 연결 레이어
        self.fc = nn.Linear(512, num_classes)
        
    # ResNet의 각 레이어를 생성하는 함수
    def _create_layer(self, block, out_ch, num_layers, stride):
        layer_list = []
        # 첫 번째 블록은 stride를 받을 수 있음
        layer_list.append(block(self.initial_channels, out_ch, stride))
        self.initial_channels = out_ch  # 다음 블록을 위해 채널 수 업데이트
        # 나머지 블록들은 기본 stride를 사용
        for _ in range(1, num_layers):
            layer_list.append(block(out_ch, out_ch))
        return nn.Sequential(*layer_list)
    
    def forward(self, x):
        # 첫 번째 컨볼루션 + ReLU 활성화 함수
        x = F.relu(self.bn1(self.conv1(x)))
        # 각 레이어를 순차적으로 통과
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # 평균 풀링 및 텐서의 차원 축소
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # 최종 완전 연결 레이어를 통해 클래스별 예측값 출력
        x = self.fc(x)
        return x

# Custom ResNet-18 모델 생성 (각 레이어의 블록 수는 2개씩)
model = CustomResNet(Block, [2, 2, 2, 2], num_classes=10)
```

### 이미지 처리 모델 (주요 CNN 아키텍쳐)
    
☑️ **ResNet (Residual Network)**
- 매우 깊은 신경망을 학습할 수 있도록 설계된 아키텍처
- 잔차 연결(Residual Connection)을 도입하여, 기울기 소실 문제를 해결
- ResNet-50, ResNet-101, ResNet-152 등의 변형
    
☑️ **VGG**
- 작은 3x3 필터를 사용하여 깊이를 증가시킨 아키텍처
- 단순하고 규칙적인 구조로 인해, 다양한 변형이 가능
- VGG16, VGG19 등
    
☑️ **Inception**
- 다양한 크기의 필터를 병렬로 적용하여, 여러 수준의 특징을 추출
- Inception 모듈을 사용하여, 네트워크의 깊이와 너비를 동시에 확장
- GoogLeNet(Inception v1), Inception v2, Inception v3 등
    
☑️ **객체 탐지YOLO(You Only Look Once)**
- 이미지에서 객체의 위치와 클래스를 동시에 예측
- 이미지 전체를 한 번에 처리하여, 빠르고 정확한 객체 탐지를 수행

### **YOLO의 개념**
- 이미지를 SxS 그리드로 나누고, 각 그리드 셀에서 객체의 존재 여부를 예측
- 각 그리드 셀은 B개의 바운딩 박스와 C개의 클래스 확률을 출력

### **YOLO의 동작 원리**
- 입력 이미지를 CNN을 통해 특징 맵으로 변환
- 특징 맵을 SxS 그리드로 나누고, 각 그리드 셀에서 바운딩 박스와 클래스 확률 예측
- 예측된 바운딩 박스와 클래스 확률을 바탕으로, 객체의 위치와 클래스를 결정

### **이미지 세그멘테이션 기법과 응용**
- 이미지의 각 픽셀을 클래스 레이블로 분류하는 작업
- 주로 시맨틱 세그멘테이션과 인스턴스 세그멘테이션으로 분류
시맨틱 세그멘테이션 (Semantic Segmentation)
$\scriptsize\textsf{이미지의 각 픽셀을 클래스 레이블로 분류}$
인스턴스 세그멘테이션 (Instance Segmentation)
$\scriptsize\textsf{시맨틱 세그멘테이션과 달리, 같은 클래스 내에서도 개별 객체를 구분}$

<img src="./images/Image_seg.png" style="width:60%; height:auto;display: block; margin: 0 auto;">

### 주요 세그멘테이션 모델

- **FCN (Fully Convolutional Network)**: 모든 레이어를 합성곱 레이어로 구성하여, 픽셀 단위의 예측을 수행
- **U-Net**: U자형 구조를 가지며, 인코더-디코더 아키텍처를 사용하여 세그멘테이션을 수행
- **Mask R-CNN**: 객체 탐지와 인스턴스 세그멘테이션을 동시에 수행하는 모델