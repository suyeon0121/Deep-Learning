## Fashion MNIST(CNN)
### Overview
Fashion MNIST 데이터셋을 사용하여 CNN 기반 이미지 분류 모델을 구현한다. 
Convolution과 Pooling을 통해 이미지의 공간적 특징을 학습한다. 

### Key Idea
이미지는 픽셀 간 공간적 구조가 중요한 데이터이다. 
CNN은 Convolution 계층을 통해 국소적인 패턴(edge, texture)을 효과적으로 추출할 수 있다. 
MLP 대비 이미지 분류에 더 적합한 구조이다. 

### Model Architecture
- input: 1x28x28 grayscale image

convolution Block
- Conv2d(1->32, kernel=3, padding=1) + ReLU
- MaxPool2d(2)
- Conv2d(32->64, kernel=3, padding=1) + ReLU
- MaxPool2d(2)

Fully Connected
- Flatten(64x7x7)
- Linear(64x7x7 -> 10)

### Training Setup
- Input Normalization: mean=0.5, std=0.5
- Loss Function: CrossEntropyLoss
- Optimizer: Adam(lr=1e-3)
- Batch Size: 64
- Epochs: 5
