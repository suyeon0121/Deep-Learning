## MLP Autoencoder
### Overview
Wine 데이터셋의 수치형 feature를 입력으로 하는 MLP 기반 AutoEncoder를 학습하여 입력 데이터를 재구성한다. 
재구성 오차를 계산해 이상치 점수(anomaly score)로 활용된다. 

### Key Idea
Wine 데이터는 연속형 feature로 구성된 tabular 데이터이다. 
AutoEncoder는 정상 데이터의 공통 구조를 압축된 표현으로 학습하며, 분포에서 벗어난 데이터는 재구성 오차가 커진다. 
라벨 없이 데이터 구조를 학습하는 비지도 학습 방식이다. 

### Model Architecture
- input: 13-dimensional feature vector
- Encoder: 13 -> 8 -> 3 (ReLU)
- Decoder: 3 -> 8 -> 13 (ReLU)
- Bottleneck: 3-dimensional latent representation

### Training
- Feature Scaling: StandardScaler
- Loss Function: Mean Squared Error(MSE)
- Optimizer: Adam (lr = 0.001)
- Batch Size: 32
- Epochs: 100

### Result
학습이 진행될수록 재구성 손실이 감소한다. 
학습 후 각 샘플의 평균 재구성 오차를 계산하여 이를 anomaly score로 사용할 수 있다. 
