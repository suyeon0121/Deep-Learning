## LeNet
- LeNet-5는 합성곱 신경망이라는 개념을 최초로 얀 르쿤(Yann LeCun)이 개발한 구조이다.
- LeNet-5는 합성곱(convolutional)과 다운 샘플링(sub-samling)(혹은 풀링)을 반복적으로 거치면서 마지막에 완전연결층에서 분류를 수행한다. 

<br/>

- C1에서 5x5 합성곱 연산 후 28x28 크기의 특성 맵 여섯 개를 생성한다.
- S2에서 다운 샘플링하여 특성 맵 크기를 14x14로 줄인다.
- 다시 C3에서 5x5 합성곱 연산하여 10x10 크기의 특성 맵 16개를 생성하고, S4에서 다운 샘플링하여 특성 맵 크기를 5x5로 줄인다.
- C5에서 5x5 합성곱 연산하여 1x1 크기의 특성 맵 120개를 생성하고, 마지막으로 F6에서 완전연결층으로 C5의 결과를 유닛(또는 노드) 84개에 연결시킨다.

<br/>

- 이때 C로 시작하는 것은 합성곱층을 의미하고, S로 시작하는 것은 풀링층을 의미한다. 또한, F로 시작하는 것은 완전연결층을 의미한다.

<br/>

<img width="567" height="241" alt="image" src="https://github.com/user-attachments/assets/6fc1e4c3-10ce-49b0-91f6-f272ac8563dc" />

<br/>
<br/>

- 구현해볼 신경망 구조 

<br/>

<img width="570" height="119" alt="image" src="https://github.com/user-attachments/assets/a669f632-836e-439e-bb6e-158b4962f5f9" />

<br/>
<br/>

- 32x32 크기의 이미지에 합성곱층과 최대 풀링층이 쌍으로 두 번 적용된 후 완전연결층을 거쳐 이미지가 분류되는 신경망이다.

<br/>

<img width="569" height="449" alt="image" src="https://github.com/user-attachments/assets/e7f5f128-4f2b-4507-ba68-cda5cf8f0238" />








