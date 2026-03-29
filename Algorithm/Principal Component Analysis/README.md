## PCA
- 변수가 많은 고차원 데이터의 경우 중요하지 않은 변수로 처리해야 할 데이터양이 많아지고 성능 또한 나빠지는 경향이 있다.
- 이러한 문제를 해결하기 위해 고차원 데이터를 저차원으로 축소시켜 데이터가 가진 대표 특성만 추출한다면 성능은 좋아지고 작업도 좀 더 간편해진다.
- 이때 대표적인 알고리즘이 PCA(Principal Component Analysis)이다.
- 즉, 고차원 데이터를 저차원(차원 축소) 데이터로 축소시키는 알고리즘이다. 

<br/>
<br/>

차원 축소 방법은 다음과 같다. 

**데이터들의 분포 특성을 잘 설명하는 벡터를 두 개 선택**
- 다음 그림에서 $e_1$과 $e_2$ 두 벡터는 데이터 분포를 잘 설명한다.
- $e_1$의 방향과 크기, $e_2$의 방향과 크기를 알면 데이터 분포가 어떤 형태인지 알 수 있기 때문이다.

<br/>

**벡터 두 개를 위한 적정한 가중치를 찾을 때까지 학습을 진행**

<img width="328" height="223" alt="image" src="https://github.com/user-attachments/assets/b8d1ee3d-b947-4315-a2d1-62868803374f" />

<br/>

즉, PCA는 데이터 하나하나에 대한 성분을 분석하는 것이 아니라, 여러 데이터가 모여 하나의 분포를 이룰 때 이 분포의 주성분을 분석하는 방법이다. 

### 결과 

- min_samples=50

<br/>

<img width="444" height="726" alt="image" src="https://github.com/user-attachments/assets/c833e713-cc05-4065-b076-cc0b4465be0b" />

<br/>
<br/>

- min_samples=100

<br/>

<img width="417" height="727" alt="image" src="https://github.com/user-attachments/assets/faff8d0c-f28a-437b-b1ce-09163fe396e0" />

<br/>

- 50보다 100에서 많은 클러스터 부분이 무시된 것을 확인할 수 있다. 
- 이와 같이 모델에서 하이퍼파라미터 영향에 따라 클러스터 결과(성능)가 달라지므로, 최적의 성능을 내려면 하이퍼파라미터를 이용한 튜닝이 중요하다.
