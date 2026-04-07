## AlexNet
- AlexNet은 ImageNet 영상 데이터베이스를 기반으로 한 화상 인식 대회인 ‘ILSVRC 2012’에서 우승한 CNN 구조이다.
- AlexNet을 설명하기 앞서 AlexNet의 세부 블록을 이해하고자 CNN 구조를 다시 살펴보아야 한다.
- CNN은 다음 그림과 같이 3차원 구조를 갖는다는 것을 이해해야 한다.(이미지를 다루기 때문에 기본적으로 3차원 데이터를 다룬다)
- 이미지 크기를 나타내는 너비(width)와 높이(height)뿐만 아니라 깊이(depth)를 갖는다.
- 보통 색상이 많은 이미지는 R/G/B 성분 세 개를 갖기 때문에 시작이 3이지만, 합성곱을 거치면서 특성 맵이 만들어지고 이것에 따라 중간 영상의 깊이가 달라진다.

<br/>

<img width="384" height="147" alt="image" src="https://github.com/user-attachments/assets/4e7463fc-46a9-4461-9f59-1be44b72a5b8" />

<br/>

- AlexNet은 합성곱층 총 다섯 개와 완전연결층 세 개로 구성되어 있으며, 맨 마지막 완전연결층은 카테고리 1000개를 분류하기 위해 소프트맥스 활성화 함수를 사용하고 있다.
- 전체적으로 보면 GPU 두 개를 기반으로 한 병렬 구조인 점을 제외하면 LeNet-5와 크게 다르지 않다.

<br/>

<img width="510" height="263" alt="image" src="https://github.com/user-attachments/assets/3ddc7172-88a9-4281-b865-f90ee11668e0" />

