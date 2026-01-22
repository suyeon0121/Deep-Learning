import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()

train_x = train_x / 255.0
test_x = test_x / 255.0

class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    train_x, train_y, epochs=1000, batch_size=64, 
    validation_split=0.2, callbacks=[early_stop]
)

test_loss, test_acc = model.evaluate(test_x, test_y)

pred = model.predict(test_x)
pred_label = np.argmax(pred[0])

plt.figure(figsize=(6, 4))
plt.bar(class_names, pred[0])
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.imshow(test_x[0])
plt.title(f"True: {class_names[test_y[0]]} / pred: {class_names[pred_label]}")
plt.axis("off")
plt.show()