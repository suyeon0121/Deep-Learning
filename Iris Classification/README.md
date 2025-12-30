## Iris Classification
### Overview
- Logistic Regression을 사용하여 iris 데이터셋의 세 가지 품종(setosa, versicolor, virginica)을 분류한다.
- 입력 feature scaling과 다중 클래스 분류 설정을 포함한 전체 분류 파이프라인을 구현한다.

### Key Idea
- iris 데이터는 수치형 feature로 구성된 tabular 데이터이다.
- 선형 결정 경계를 기반으로 하는 Logistic Regression으로도 높은 분류 성능을 얻을 수 있다.
- 다중 클래스 문제이므로 multinomial Logistic Regression을 사용한다 

### Model Architecture
- input: 4 feature (sepal length, sepal width, petal length, petal width)
- Model: Multinomial Logistic Regression
- Output: 3-class probability distribution

### Training Setup
- Feature Scaling: StandardScaler
- Solver: lbfgs
- Max Iterations: 200
- Evaluation Metric: Accuracy

### Result
간단한 선형 모델만으로도 iris 데이터의 클래스 구조를 효과적으로 분리할 수 있음을 확인
