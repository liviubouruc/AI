from typing import Sequence
import numpy as np
import matplotlib.pyplot as plt
import PIL
import copy
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.applications.resnet import ResNet50

train_data = np.loadtxt("train.txt", dtype=str)
train_images = []
train_labels = []
for data in train_data:
    train_image, train_label = data.split(',')

    pre_img = PIL.Image.open('train/%s' % train_image)
    rgb_img = PIL.Image.new("RGB", pre_img.size)
    rgb_img.paste(pre_img)
    img = copy.deepcopy(np.asarray(rgb_img))
    train_images.append(img)
    pre_img.close()

    train_labels.append(train_label)

train_images = np.array(train_images)
train_images = train_images.reshape(len(train_data), 50, 50, 3)
print(train_images.shape)
train_labels = np.array(train_labels).astype('float')
train_labels = train_labels.reshape(len(train_data), 1)
print(train_labels.shape)


validation_data = np.loadtxt("validation.txt", dtype=str)
validation_images = []
validation_labels = []
for data in validation_data:
    validation_image, validation_label = data.split(',')

    pre_img = PIL.Image.open('validation/%s' % validation_image)
    rgb_img = PIL.Image.new("RGB", pre_img.size)
    rgb_img.paste(pre_img)
    img = copy.deepcopy(np.asarray(rgb_img))
    validation_images.append(img)
    pre_img.close()

    validation_labels.append(validation_label)

validation_images = np.array(validation_images)
validation_images = validation_images.reshape(len(validation_data), 50, 50, 3)
validation_labels = np.array(validation_labels).astype('float')
validation_labels = validation_labels.reshape(len(validation_data), 1)


test_data = np.loadtxt("test.txt", dtype=str)
test_images = []
for test_image in test_data:
    pre_img = PIL.Image.open('test/%s' % test_image)
    rgb_img = PIL.Image.new("RGB", pre_img.size)
    rgb_img.paste(pre_img)
    img = copy.deepcopy(np.asarray(rgb_img))
    test_images.append(img)
    pre_img.close()

test_images = np.array(test_images)
test_images = test_images.reshape(len(test_data), 50, 50, 3)


def preprocess_data(data, labels):
    data = keras.applications.resnet50.preprocess_input(data)
    labels = keras.utils.to_categorical(labels, 3)
    return data, labels

train_images, train_labels = preprocess_data(train_images, train_labels)
print(train_images.shape, train_labels.shape)
validation_images, validation_labels = preprocess_data(validation_images, validation_labels)

train_images, train_labels = preprocess_data(train_images, train_labels)
print(train_images.shape, train_labels.shape)
validation_images, validation_labels = preprocess_data(validation_images, validation_labels)

resnet_model = keras.models.Sequential()
resnet_model.add(keras.layers.Lambda(lambda image: tf.image.resize(image, (192, 192))))
resnet_model.add(ResNet50(include_top=False, pooling='max', weights='imagenet'))
resnet_model.add(keras.layers.Flatten())
resnet_model.add(keras.layers.Dense(256, activation='relu'))
resnet_model.add(keras.layers.BatchNormalization())
resnet_model.add(keras.layers.Dropout(0.3))
resnet_model.add(keras.layers.Dense(128, activation='relu'))
resnet_model.add(keras.layers.BatchNormalization())
resnet_model.add(keras.layers.Dropout(0.5))
resnet_model.add(keras.layers.Dense(3, activation='softmax'))

# cu 10 epoci si droputuri 0.3 0.5 [0.8269070982933044, 0.7431111335754395]
#cu 15 epoci si 0.3 0.7 [0.7292381525039673, 0.7775555849075317]
# 20 epoci 0.3 0.7 [0.7863541841506958, 0.7955555319786072] si 0.99 pe train
# 20 epoci 0.7 0.9 2e-5 [0.6383113861083984, 0.8028888702392578]
opt = keras.optimizers.Adam(learning_rate=2e-5) 
resnet_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
resnet_model.fit(train_images/255, train_labels, epochs=10, batch_size=64, validation_data=(validation_images/255, validation_labels))
print(resnet_model.evaluate(validation_images/255, validation_labels))
# resnet_model.add(keras.layers.Dense(3, activation='relu'))
# resnet_model.layers[0].trainable = False

# resnet_model.summary()

# opt = keras.optimizers.Adam(learning_rate=0.01, decay = 1e-6, momentum=0.9, nesterov=True)
# resnet_model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy']) 