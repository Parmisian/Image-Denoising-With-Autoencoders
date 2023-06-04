# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 16:05:27 2023

@author: ulas0
"""

import matplotlib.pyplot as plt
#import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images / 255.0

test_images = test_images / 255.0


train_labels_onehot = tf.keras.utils.to_categorical(train_labels, num_classes=10)
test_labels_onehot = tf.keras.utils.to_categorical(test_labels, num_classes=10)

train_images = train_images[..., tf.newaxis]
test_images = test_images[..., tf.newaxis]

noise_factor = 0.2
train_images_noisy = train_images + noise_factor * tf.random.normal(shape=train_images.shape) 
test_images_noisy = test_images + noise_factor * tf.random.normal(shape=test_images.shape) 

# Make sure values still in (0,1)
train_images_noisy = tf.clip_by_value(train_images_noisy, clip_value_min=0., clip_value_max=1.)
test_images_noisy = tf.clip_by_value(test_images_noisy, clip_value_min=0., clip_value_max=1.)


# CNN CLASSIFIER
# declare input shape 
input = tf.keras.Input(shape=(28,28,1))
# Block 1 (convolution)
conv1 = tf.keras.layers.Conv2D(64, 3, strides=1, activation="relu")(input)
# Block 2 (convolution 2)
conv2 = tf.keras.layers.Conv2D(64, 3, strides=1, activation="relu")(conv1)
# Block 3 (full connected9)
fc = tf.keras.layers.Flatten()(conv2)
fc = tf.keras.layers.Dense(10)(fc)
# Finally, we add a classification layer.
output = tf.keras.layers.Dense(10, activation="softmax")(fc)
# bind all
cnn_model = tf.keras.Model(input, output)


loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

cnn_model.compile(loss=loss_fn, optimizer="adam", metrics=["accuracy"])
cnn_model.summary()

history = cnn_model.fit(train_images, train_labels, epochs=3)

test_loss, test_acc = cnn_model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy for clean test images:', test_acc)

test_loss_n, test_acc_n = cnn_model.evaluate(test_images_noisy,  test_labels, verbose=2)

print('\nTest accuracy for noisy tst images:', test_acc_n)


# CNN AUTOENCODER -- TRAIN WITH NOISY TRAINING IMGS

class Denoise(Model):
  def __init__(self):
    super(Denoise, self).__init__()
    self.encoder = tf.keras.Sequential([
      tf.keras.layers.Input(shape=(28, 28, 1)),
      tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
      tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)])

    self.decoder = tf.keras.Sequential([
      tf.keras.layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
      tf.keras.layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
      tf.keras.layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = Denoise()

autoencoder.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())

autoencoder.fit(train_images_noisy, train_images,
                epochs=1,
                shuffle=True,
                validation_data=(test_images_noisy, test_images),verbose=2)

#Visual Check
encoded_imgs = autoencoder.encoder(test_images_noisy).numpy() #change if you encode not- noisy test imgs
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):

    # display original + noise
    ax = plt.subplot(2, n, i + 1)
    plt.title("original + noise")
    plt.imshow(tf.squeeze(test_images_noisy[i])) #change if you are encoding not-noisy test imgs
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    bx = plt.subplot(2, n, i + n + 1)
    plt.title("reconstructed")
    plt.imshow(tf.squeeze(decoded_imgs[i]))
    plt.gray()
    bx.get_xaxis().set_visible(False)
    bx.get_yaxis().set_visible(False)
plt.show()


reconstructions = autoencoder.predict(test_images_noisy) #denoised images

test_loss_rec, test_acc_rec = cnn_model.evaluate(reconstructions,  test_labels, verbose=2)

print('\nTest accuracy for autoencoder denoised test images:', test_acc_rec)
print('\n')

history_2 = cnn_model.fit(train_images_noisy, train_labels, epochs=3)

test_loss_2, test_acc_2 = cnn_model.evaluate(test_images_noisy,  test_labels, verbose=2)

print('\nTest accuracy for training with noisy train images and test with noisy test images:', test_acc_2)






