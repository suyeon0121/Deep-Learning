## MLP Autoencoder
### Overview
MLP Autoencoder를 사용하며 Wine 데이터셋의 feature를 저차원 공간으로 압축하고, 입력과 복원 간의 차이를 기반으로 이상치 점수를 계산한다. 

### Key Idea
Autoencoder는 입력 데이터를 그대로 재현하도록 학습한다. 
정상 데이터에 대해 낮은 reconstruction error를 보이며, 패턴에서 벗어난 샘플을 높은 reconstruction error를 가진다. 

### Model Architecture
- input: 13 numerical feature
- Encoder: Linear -> ReLU -> Linear
- Bottleneck: 3-dimensional latent vector
- Decoder: Linear -> ReLU -> Linear
- Output: reconstructed input

### Training
- Feature Scaling: StandardScaler
- Loss Function: Mean Squared Error
- Optimizer: Adam
- Epochs: 100

### Result
입력 데이터의 구조적 패턴을 저차원 latent space로 압축할 수 있으며, reconstruction error를 통해 상대적인 이상치 여부를 정량적으로 평가할 수 있다. 
