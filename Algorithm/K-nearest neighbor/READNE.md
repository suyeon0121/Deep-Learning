## K-최근접 이웃(K-nearest neighbor)
- K-최근접 이웃은 새로운 입력(학습에 사용하지 않은 새로운 데이터)을 받았을 때 기존 클러스터에서 모든 데이터와 인스턴스 기반 거리를 측정한 후 가장 많은 속성을 가진 클러스터에 할당하는 분류 알고리즘이다.
- 즉, 과거 데이터를 사용하여 미리 분류 모형을 만드는 것이 아니라, 과거 데이터를 저장해 두고 필요할 때마다 비교를 수행하는 방식이다.

<br/>

### 오류:
- This StandardScaler instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.
  - StandardScaler의 평균과 표준편차를 계산하는 과정이 fit이다.
  - 데이터 변환(Transofrm)하기 전에, 기준을 잡는 학습(fit) 단계를 건너뛰어 오류 발생

<br/>

### 해결
훈련 데이터에 fit_transform()를 사용

- X_train = s.fit_transform(X_train):
    - 훈련 데이터는 학습(fit)과 변환(transform)을 동시에 수행
    - 훈련 데이터의 평균과 표준편차를 계산한 뒤, 그 기준으로 데이터를 변형함
- X_test = s.transform(X_test):
    - 테스트 데이터는 훈련 데이터에서 이미 계산된 기준을 그대로 적용(transform)만 수행
    - 모델이 미리 테스트 데이터를 학습해버리는(Data Leakage) 현상을 방지하기 위함

<br/>

마지막에 수치가 아닌 시각화를 확인할 수 있도록 코드 추가
- 정확도가 올라가거나 떨어지는 지점을 확인할 수 있다.

<br/>

<img width="787" height="579" alt="image" src="https://github.com/user-attachments/assets/0cf7e777-d78f-499d-828b-3bb2d09c075f" />

<br/>
- 선의 길이는 데이터 간의 유클리드 거리를 의미
- KNN은 이 선의 길이가 가장 짧은 순서대로 이웃을 맺음

