import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


x_train_final = x_train.reshape(len(x_train),784)
x_train_final = x_train_final/255

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(10, activation='relu', input_shape=(784,)))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train_final, y_train, epochs=15)

x_test_final=x_test.reshape(len(x_test),784)
x_test_final_flattened = x_test_final/255


predict = model.predict(x_test_final_flattened)

predict[1356]

output = np.argmax(predict[1356])

print(output)
