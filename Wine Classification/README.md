## Wine Classification
### Overview
Logistic Regression을 사용하여 wine 데이터셋의 세 가지 클래스를 분류한다.
전처리와 모델을 Pipeline으로 구성하여 재사용 가능한 분류 파이프라인을 구현한다.

### Key Idea
Wine 데이터는 연속형 수치 feature로 구성된 tabular 데이터이다. 
선형 결정 경계를 기반으로 하는 Logistic Regression으로도 충분히 분류 성능을 얻을 수 있다.
다중 클래스 문제이므로 multinomial Logistic Regression을 사용한다. 

### Model Architecture
- input: 13 feature (wine chemical properties)
- Model: Multinomial Logistic Regression
- Output: 3-class probability distribution

### Training
- Preprocessing: StandardScaler
- Solver: lbfgs
- Max Iterations: 200
- Evaluation metric: Accuracy
