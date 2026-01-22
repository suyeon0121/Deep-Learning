## Fashion MNIST
### Overview
Fashion-MNIST 데이터셋을 사용하여 의류 의미지를 MLP 기반 신경망으로 분류한다. 
Flatten -> Dense 구조를 통해 이미지 분류의 기본 흐름을 구현한다. 

### Key Idea
- Fashion-MNIST는 28*28 흑백 이미지로 구성된 분류 문제이다.
- 이미지를 벡터로 펼친 뒤(Dense 입력), 비선형 은닉층을 통해 클래스별 확률을 에측한다.
- Early Stopping을 사용해 과적합을 방지한다.

### Model Architecture
- Input: 28x28 grayscale image
- Flatten: 784-dimensional vector
- Hidden Layer: Dense(128), ReLU
- Output Layer: Dense(10), Softmax

### Training Setup
- Input Scaling: pixel value / 255
- Loss Function: Sparse Categorical Crossentropy
- Optimizer: Adam
- Batch Size: 64
- Epochs: 최대 1000(Early Stopping 적용)
- Validation: train 데이터의 20%

### Result
Early Stopping으로 검증 손실 기준 최적 시점에서 학습을 종료한다. 
테스트 데이터에 대해 분류 정확도를 평가하고, 단일 샘플에 대한 클래스 확률 분포와 예측 결과를 시각화한다. 

## 한계
### Limitation of MLP on Image Data
Flatten 레이어를 통해 2차원 이미지가 1차원 벡터로 변환되면서, 픽셀 간의 공간적 관계와 국소 패턴 정보가 모두 제거된다. 이로 인해 모델은 이미지를 구조적인 형태가 아닌 단순한 숫자 집합으로 인식하게 된다. 
Dense 기반 MLP는 전체 입력을 전역적으로 처리하기 때문에, 의류 의미지에서 중요한 경계, 질감, 부분 형태와 같은 국소적 특징을 효과적으로 학습하기 어렵다. 이러한 구조적 한계는 클래스 간 형태 차이가 미묘한 Fahsion-MNIST 데이터셋에서 성능 한계로 드러난다.

### Why CNN is Needed
CNN는 이미지의 공간 구조를 유지한 채 국소 영역에 대한 패턴을 학습하도록 설계된 모델이다. Convolution 연산을 통해 edge, texture와 같은 저수준 특징을 추출하고, 계층적으로 더 복잡한 형태를 학습할 수 있다.
