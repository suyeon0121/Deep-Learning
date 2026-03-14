## MLP Autoencoder
### Overview
Wine 데이터셋의 수치형 feature를 입력으로 하는 MLP 기반 AutoEncoder를 학습하여 입력 데이터를 재구성한다. 
AutoEncoder는 입력 데이터를 저차원 latent representation으로 압축한 뒤 다시 복원하는 방식으로 학습하며, 모델의 재구성 오차(reconstruction error)를 활용하여 이상치를 탐지할 수 있다. 
- AutoEncoder 기반 reconstruction 학습
- reconstruction error 기반 anomaly score 계산
- error distribution 분석
- latent space visualization

### Dataset
- 샘플 수: 178
- feature 수: 13
- feature 타입: 연속형 수치 데이터
AutoEncoder는 라벨 없이 데이터 분포 구조를 학습하는 비지도 학습 방식으로 학습된다. 

### Key Idea
- AutoEncoder는 입력 데이터를 압축된 latent representation으로 인코딩한 후 다시 복원하는 구조를 가진다. 
- 모델은 정상 데이터의 공통 구조를 학습하게 되며, 분포에서 벗어난 데이터는 재구성 오차가 크게 나타나는 특성을 가진다.

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
```
Epoch [20/100], Loss: 5.7486
Epoch [40/100], Loss: 3.8560
Epoch [60/100], Loss: 3.0736
Epoch [80/100], Loss: 2.8733
Epoch [100/100], Loss: 2.6155
Threshold: 1.2249794602394104
Number of anomalies: 8
```

### Reconstruction Error
<img width="195" height="51" alt="Image" src="https://github.com/user-attachments/assets/97924b9c-2366-40f4-b191-3c5f1ce280fa" />

모델이 해당 데이터를 얼마나 잘 재구성하는지를 나타낸다. 
- 낮은 error -> 정상 데이터
- 높은 error -> 이상 데이터

### Reconstruction Error Distribution
<img width="640" height="480" alt="Image" src="https://github.com/user-attachments/assets/016afcf6-a843-47be-be3c-93008bab1370" />

- 대부분의 데이터는 낮은 reconstruction error
- 일부 샘플은 높은 error
이는 AutoEncoder가 데이터의 공통 구조를 학습했음을 의미한다.

### Anomaly Threshold
``threshold = mean + 2 × std``
threshold를 초과하는 데이터는 데이터 분포에서 벗어난 anomaly 후보로 판단할 수 있다. 

### Latent Space Visualization(PCA)
AutoEncoder encoder는 입력 데이터를 3차원 latent representation으로 압축한다
<img width="640" height="480" alt="Image" src="https://github.com/user-attachments/assets/33bde2e2-4457-410e-8f07-262a87dc52f0" />

latent representation을 PCA로 2차원 공간에 투영하여 시각화를 했다. 
- 데이터가 몇 개의 cluster 형태로 분리됨
- encoder가 데이터 구조를 학습했음을 확인했다.

### Latent Space Visualization(t-SNE)
<img width="640" height="480" alt="Image" src="https://github.com/user-attachments/assets/8ec2ffe5-0331-4ca1-b7eb-5d036170df31" />

- PCA보다 더 명확한 cluster 구조를 확인할 수 있다. 
- AutoEncoder가 데이터의 구조적 패턴을 학습했음을 보여준다

### Reconstruction Quality
<img width="640" height="480" alt="Image" src="https://github.com/user-attachments/assets/27ab2fac-a331-4245-bff5-bf771ad18160" />

대부분의 feature 값이 유사하게 복원되었으며 이는 AutoEncoder가 데이터의 주요 패턴을 학습했음을 의미한다. 
<br />
<br />

## PCA vs AutoEncoder
<img width="640" height="480" alt="Image" src="https://github.com/user-attachments/assets/e83cef9c-4472-4d6e-bc98-a380065a9d81" />

<img width="640" height="480" alt="Image" src="https://github.com/user-attachments/assets/96238c4a-32ee-4321-977c-9235f79ac2b3" />

- AutoEncoder -> 더 낮은 reconstruction error
- PCA -> 더 높은 reconstruction error
AutoEncoder가 비선형 feature 관계를 더 잘 학습할 수 있기 때문이다.
<br/>
PCA는 **선형** 차원 축소이고 AutoEncoder는 **비선형** representation learning이다. <br/>
따라서 복잡한 데이터 구조에서는 AutoEncoder가 더 강력한 표현력을 가진다. 





