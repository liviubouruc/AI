import numpy as np
import matplotlib.pyplot as plt
import PIL
import copy
import tensorflow as tf
from tensorflow import keras

train_data = np.loadtxt("train.txt", dtype=str)
train_images = []
train_labels = []
for data in train_data:
    train_image, train_label = data.split(',')

    pre_img = PIL.Image.open('train/%s' % train_image)
    img = copy.deepcopy(np.asarray(pre_img))
    train_images.append(img)
    pre_img.close()

    train_labels.append(train_label)

train_images = np.array(train_images)
print(type(train_images[0]))
train_labels = np.array(train_labels).astype('float')
print(type(train_labels[0]))

validation_data = np.loadtxt("validation.txt", dtype=str)
validation_images = []
validation_labels = []
for data in validation_data:
    validation_image, validation_label = data.split(',')

    pre_img = PIL.Image.open('validation/%s' % validation_image)
    img = copy.deepcopy(np.asarray(pre_img))
    validation_images.append(img)
    pre_img.close()

    validation_labels.append(validation_label)

validation_images = np.array(validation_images)
validation_labels = np.array(validation_labels).astype('float')

test_data = np.loadtxt("test.txt", dtype=str)
test_images = []
for test_image in test_data:
    pre_img = PIL.Image.open('test/%s' % test_image)
    img = copy.deepcopy(np.asarray(pre_img))
    test_images.append(img)
    pre_img.close()

test_images = np.array(test_images)

cnn_model = keras.models.Sequential([
    keras.layers.Conv2D(6, kernel_size=5, strides=1,  activation='tanh', input_shape=(50,50,1), padding='same'), #C1
    keras.layers.AveragePooling2D(), #S2
    keras.layers.Conv2D(16, kernel_size=5, strides=1, activation='tanh', padding='valid'), #C3
    keras.layers.AveragePooling2D(), #S4
    keras.layers.Flatten(), #Flatten
    keras.layers.Dense(120, activation='tanh'), #C5
    keras.layers.Dense(84, activation='tanh'), #F6
    keras.layers.Dense(10, activation='softmax') #Output layer
])
# [5.512113571166992, 0.4915555417537689]

# print(train_images.shape)
train_images = train_images.reshape(len(train_data), 50, 50, 1)
validation_images = validation_images.reshape(len(validation_data), 50, 50, 1)
test_images = test_images.reshape(len(test_data), 50, 50, 1)

cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn_model.fit(train_images/255, train_labels, epochs=100, batch_size=64, validation_data=(validation_images/255, validation_labels))
print(cnn_model.evaluate(validation_images/255, validation_labels))

# loss: 0.9351 - accuracy: 0.7695

# predicted_test_labels = cnn_model.predict(test_images/255)
# output = zip(test_data, predicted_test_labels)
# with open("submission.csv", "w") as g:
#     g.write("id,label\n")
#     for predict in output:
#         g.write(str(predict[0]) + ',' + str(predict[1]) + '\n')

