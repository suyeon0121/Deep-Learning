## MNIST Digit Classification
### Overview
MLP(Multi-Layer Perceptron)를 사용하여 MNIST 데이터셋의 손글씨 숫자를 분류한다. 
이미지를 벡터로 변환하여 fully connected 신경망 기반 분류 파이프라인을 구성한다.

### Key Idea
MNIST는 28x28 흑백 이미지로 구성된 데이터셋이다. 
이미지를 1차원 벡터로 펼친 뒤, 비선형 활성화 함수를 가진 MLP로 분류를 수행한다. 

### Model Architecture
- Input: 28 x 28 grayscale image
- Flatten -> Linear(784 -> 256) -> ReLU -> Linear(256 -> 128) -> ReLU -> Linear(128 -> 10)
- Output: 10-class logits

### Regularization
- L2 regularization
- Dropout (0.5)

### Training
- Loss Function: CrossEntropyLoss
- Optimizaers: Adam(lr = 1e-3)
- Batch size: 64
- Epochs: 5
