## Wine Classification
### Overview
Wine 데이터셋을 대상으로 Multi-Layer Perceptron 기반 다중 분류 모델을 구현한다. 
입력 feature 정규화와 train/test 분리를 포함한 딥러닝 분류의 표준 학습 파이프라인을 구성한다. 

### Key Idea
Wine 데이터셋은 13차원 feature를 가지는 다중 클래스 분류 문제이다. 
입력 feature 간 스케일 차이가 크므로 정규화가 필요하다
MLP는 비선형 activation을 통해 클래스 간 복잡한 decision boundary를 학습할 수 있다. 

### Model Architecture
- input: 13
- Hidden layer: 2 layers (32, 16 neurons)
- Output: 3
- Fully connected MLP
- ReLU activation (hidden layers)
- Softmax activation (output layer)

### Data Processing
- Dataset: Wine (178 samples, 3 classes)
- Train/Test split: 80% / 20%
- Feature scaling: StandardScaler
- Label encoding: One-hot encoding

### Training
- Loss: Categorical Cross Entropy
- Optimizer: Adam
- Batch size: 8
- Epochs: 100
- Evaluation metric: Accuracy
