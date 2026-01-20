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
