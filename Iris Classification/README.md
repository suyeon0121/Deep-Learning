## Iris Classification
### Overview
Logistic Regression을 사용하여 iris 데이터셋의 세 가지 품종(setosa, versicolor, virginica)을 분류하는 모델
- feature scaling 기반 분류 모델 학습
- decision boundary 시각화
- PCA 기반 feature 공간 분석
- confusion matrix를 통한 오류 분석을 추가로 실험을 해봤다.

### Dataset
iris 데이터셋은 꽃의 특성 4개로 구성된 tabular classification dataset이다.
- sepal length: 꽃받침 길이
- sepal width: 꽃받침 너비
- petal length: 꽃잎 길이
- petal width: 꽃잎 너비 

각 데이터는 3개의 클래스 중 하나에 속한다. 
- setosa
- versicolor
- virginica

데이터 구성
- 총 샘플: 150
- feature 수: 4
- 클래스 수: 3

### Training Setup
- Train/Test Split: 80/20
- Feature Scaling: StandardScaler
- Solver: lbfgs
- Max Iterations: 200
- Evaluation Metric: Accuracy
Feature Scaling을 적용하는 이유는 Logisitic Regression이 feature scale에 민감한 모델이기 때문

### Result
Accuracy: 0.9333333333333333

              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       0.90      0.90      0.90        10
   virginica       0.90      0.90      0.90        10

    accuracy                           0.93        30
   macro avg       0.93      0.93      0.93        30
weighted avg       0.93      0.93      0.93        30

### Decision Boundary Visualization
<img width="640" height="480" alt="Image" src="https://github.com/user-attachments/assets/b4ccf64c-cd89-4400-98fa-8514dadee0f0" />
- setosa는 다른 클래스와 명확히 분리되었다.
- versicolor와 virginica는 일부 영역에서 overlap 발생
- Logistic Regression이 선형 경계로 feature 공간을 분리함을 확인

### PCA feature
<img width="640" height="480" alt="Image" src="https://github.com/user-attachments/assets/43092fa2-7936-4685-9783-bb0c8e3bd203" />
- setosa는 독립적인 cluster 형성
- versicolor와 virginica는 일부 겹침
- 데이터 구조가 완전히 선형적으로 분리되지 않음

### Confusion Matrix
<img width="640" height="480" alt="Image" src="https://github.com/user-attachments/assets/356e7b8b-52de-492f-a1d2-ca8af13bd99a" />
- setosa는 거의 완벽하게 분류됨
- versicolor와 virginica 사이에서 오분류 발생
- 두 클래스가 feature 공간에서 겹치는 구조와 일치
