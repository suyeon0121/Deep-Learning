## MNIST Digit Classification
### Overview
MNIST 숫자 데이터셋을 대상으로 Flatten 기반 MLP 분류 모델을 구현.
이미지 데이터를 벡터로 변환할 때 발생하는 parameter 폭증 문제와 overfitting을 다룬다.

### Key Idea
Flatten된 이미지 입력은 고차원 벡터를 생성하여 parameter 수를 급격히 증가시킨다. 
과도한 parameter는 overfitting을 유발할 수 있다. 
L2 regularization과 Dropout은 overfitting 완화를 위해 사용된다.
SGD와 Adam optimizer의 학습 특성을 비교한다. 

### Model Architecture
- Input: 28 x 28 image
- Flatten --> 784
- Hidden layers: 512, 256 neurons
- Output: 10
- Fully connected MLP
- ReLU activation
- Softmax output

### Regularization
- L2 regularization
- Dropout (0.5)

### Optimizer Comparison
- SGD는 고정 learning rate를 사용하며 수렴 속도가 느린 편이다.
- Adam은 adaptive learning rate를 사용해 초기 수렴 속도가 빠르다.
- 따라서 동일한 epoch 조건에서 Adam이 더 빠르게 accuracy를 향상시키는 경향이 보인다.

### Training
- Loss: Categorical Cross Entropy
- Optimizaers: SGD, Adam
- Batch size: 128
- Epochs: 10
- Evaluation metric: Accuracy
