## Iris Classification
### Overview
iris 데이터셋을 대상으로 Multi-Layer Perception 기반 딥러닝 분류 모델을 구현한다.
입력 feature 정규화, train/test 분리, softmax 기반 다중 분류 학습을 통해 딥러닝 분류의 전체 학습 흐름을 다룬다

### Key Idea
아이리스 분류는 다중 클래스 분류 문제
입력 feature 간 스케일 차이는 학습 불안정을 유발하므로 정규화가 필요하다.
MLP는 은닉층과 비선형 activation을 통해 복잡한 decision boundary를 학습할 수 있다. 

### Model Architecture
- input: 4
- Hidden layer: 2 layers, 16 neurons each
- Output: 3
- Fully connected MLP
- ReLU activation (hidden layers)
- Softmax activation (output layer)

### Data Processing
- Dataset: iris (150 samples, 3 classes)
- Train/Test split: 80% / 20%
- Feature scaling: StandardScaler
- Label encoding: One-hot encoding

### Training
- Loss: Categorical Cross Entropy
- Optimizer: Adam
- Batch size: 8
- Epochs: 100
- Evaluation metric: Accuracy
  
