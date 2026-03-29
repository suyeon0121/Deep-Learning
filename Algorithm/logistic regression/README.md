## 로지스틱 회귀
- 로지스틱 회귀는 분석하고자 하는 대상들이 두 집단 혹은 그 이상의 집단으로 나누어진 경우, 개별 관측치들이 어느 집단으로 분류될 수 있는지 분석하고 이를 예측하는 모형을 개발하는 데 사용되는 통계 기법이다.

<br/>

### 오류
- 1. <function gray at 0x7be4bd4b7c40> is not a valid value for cmap; supported values are 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', ...
  - plt.gray로 작성
  - gray는 pyplot 바로 아래에 있는 것이 아니라 cm(Colormap 전용 보관함) 안에 들어있음
- 2. ConvergenceWarning: lbfgs failed to converge (status=1): STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT.
  - lbfgs 알고리즘이 (정답을 찾는 데) 실패했다. 설정된 최대 반복 횟수에 도달하여 중단되었다라는 뜻


<br/>

### 해결
- 1번 오류:
  - plt.imshow() 함수는 기본적으로 숫자가 크면 노란색, 작으면 보라색으로 보여주는 ‘Viridis’라는 색상 지도를 사용
  - 하지만 다루는 숫자 데이터는 보통 흑백으로 명령 하기 위해서는 plt.cm.gray를 사용해야 한다.
- 2번 오류:
  - 반복 횟수(max_iter=2000)을 늘려 해결

<br/>

### 결과 
<img width="666" height="669" alt="image" src="https://github.com/user-attachments/assets/7a5da436-9c4a-4b82-a3ff-054051fc4c61" />

<br/>

- 대각선의 의미:
    - 왼쪽 위에서 오른쪽 아래로 이어지는 대각선 셀들은 모델이 정답을 맞힌 경우이다.
    - 이 부분에 숫자가 몰려있고 색이 진할수록 좋은 모델이다.
