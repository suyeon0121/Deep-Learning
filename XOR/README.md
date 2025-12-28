## XOR Classification
### Overview
Multi-Layer Perceptron을 직접 구현하여 XOR 분류 문제를 해결한다. 
선형 모델로는 해결할 수 없는 XOR 문제를 은닉층과 비선형 activation을 통해 학습하는 과정을 다룬다

### Key Idea
- XOR는 linearly separable 하지 않는다.
- 은닉층과 비선형 activation이 필요하다
- forward/backpropagation을 NumPy로 구현

  ### Model Architecture
  - Input: 2
  - Hidden layer: 1 layer, 4 neurons
  - Output: 1
  - Fully connected MLP
  - Sigmoid activation

  ### Training
  - Loss: Binary Cross Entropy
  - Optimizer: Gradient Descent
  - Epochs: 10,000
  - Learning rate: 0.1
