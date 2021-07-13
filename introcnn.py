
#Mnist dataset by Rodrigo De la Torre

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import RMSprop

#We get the dataset from keras.datasets, then we normalize it to have values from 0 to 1
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

model = tf.keras.Sequential([
                             #Because all images are the same size, we just declare the input as 28x28 (the size of the images)
                             tf.keras.layers.Flatten(input_shape=(28,28)),
                             #I'm using 350 for the first dense just because it's near the half of the input size
                             tf.keras.layers.Dense(350, activation = 'relu'),
                             #I'm using 35 for the next dense just because 350/10, to make the Network smaller
                             tf.keras.layers.Dense(35, activation='relu'),
                             #We add dropout to make the neurons learn better with less help of all the others
                             tf.keras.layers.Dropout(0.4),
                             #Finally we get the output layer of 10 because we have 10 classes
                             #I'm using softmax because we have multi class and only one answer
                             tf.keras.layers.Dense(10, activation = 'softmax')
])

model.summary()

model.compile(optimizer=RMSprop(learning_rate=0.001), 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

