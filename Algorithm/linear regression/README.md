## 선형 회귀 
- 선형 회귀는 독립 변수 x를 사용하여 종속 변수 y의 움직임을 예측하고 설명하는 데 사용된다.
- 독립 변수 x는 하나일 수도 있고, x1, x2, x3처럼 여러 개일 수도 있다.
- 하나의 x 값으로 y 값을 설명할 수 있다면 단순 선형 회귀(Simple linear regression)라고 하며, x 값이 여러 개라면 다중 선형 회귀(multiple linear regression)라고 한다.

<br/>

- 선형 회귀는 종속 변수와 독립 변수 사이의 관계를 설정하는 데 사용된다.
- 즉, 독립 변수가 변경되었을 때 종속 변수를 추정하는 데 유용하다.
- 반면, 로지스틱 회귀는 사건의 확률(0 또는 1)을 확인하는 데 사용된다.

<br/>

<img width="696" height="284" alt="image" src="https://github.com/user-attachments/assets/797354b7-31d8-40cb-af32-79be377ad0ed" />

<br/>

## 결과 

<img width="504" height="383" alt="image" src="https://github.com/user-attachments/assets/2bf2998d-2244-4fc1-98f6-b66bd76930fc" />

<br/>

- 빨간색 선은 모델이 학습을 통해 찾아낸 최적의 예측값이다.
- 회색 점들이 빨간 선에 완벽하게 붙어있지 않고 주변에 흩어져 있다.
  - 선과 점 사이의 수직 거리가 바로 오차이다.
  - 점들이 선에 가까울수록 모델의 예측 정확도가 높다는 뜻이며, 멀리 떨어진 점들은 모델이 설명하기 어려운 특이값일 가능성이 있다.
