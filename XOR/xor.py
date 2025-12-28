import numpy as np

x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

np.random.seed(0)
w1 = np.random.randn(2, 4)
b1 = np.zeros((1, 4))
w2 = np.random.randn(4, 1)
b2 = np.zeros((1, 1))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    return x * (1 - x)

lr = 0.1

for epoch in range(10000):
    z1 = x @ w1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ w2 + b2
    y_hat = sigmoid(z2)

    loss = -np.mean(
        y * np.log(y_hat + 1e-8) +
        (1 - y) * np.log(1 - y_hat + 1e-8)
    )

    dz2 = y_hat - y
    dw2 = a1.T @ dz2 / len(x)
    db2 = np.mean(dz2, axis=0, keepdims=True)

    da1 = dz2 @ w2.T
    dz1 = da1 * sigmoid_grad(a1)
    dw1 = x.T @ dz1 / len(x)
    db1 = np.mean(dz1, axis=0, keepdims=True)

    w2 -= lr * dw2
    b2 -= lr * db2
    w1 -= lr * dw1
    b1 -= lr * db1

    if epoch % 2000 == 0:
        print(epoch, loss)

print("predictions:")
print((y_hat > 0.5).astype(int))