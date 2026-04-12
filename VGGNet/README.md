## VGGNet
- VGGNet은 카렌 시모니안(Karen Simonyan)과 앤드류 지서만(Andrew Zisserman)이 2015 ICLR에 게재한 “Very deep convoluational networks for large-scale image recognition” 논문에서 처음 발표했다.
- VGGNet은 합성곱층의 파라미터 수를 줄이고 훈련 시간을 개선하려고 탄생했다. 즉, 네트워크를 깊게 만드는 것이 성능에 어떤 영향을 미치는지 확인하고자 나온 것이 VGG이다.
- VGG 연구 팀은 깊이의 영향만 최대한 확인하고자 합성곱층에서 사용하는 필터/커널의 크기를 가장 작은 3 x 3으로 고정했다.

<br/>

- 네트워크 계층의 총 개수에 따라 여러 유형의 VGGNet(VGG16, VGG19 등)이 있으며, 이중 VGG16 네트워크의 구조적 세부 사항은 다음 그림과 같다.
- VGG16에는 파라미터가 총 1억 3300만개 있다. 여기에서 모든 합성곱 커널의 크기는 3 x 3, 최대 풀링 커널의 크기는 2 x 2이며, 스트라이드는 2이다.
- 결과적으로 64개의 224 x 224 특성 맵(224 x 224 x 64)들이 생성된다. 또한 마지막 16번째 계층을 제외하고는 모두 ReLU 활성화 함수가 적용된다.

<br/>

<img width="569" height="377" alt="image" src="https://github.com/user-attachments/assets/6e3f86f9-1e9f-440a-9344-662b492f3315" />

<br/>
