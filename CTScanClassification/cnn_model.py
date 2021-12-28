import numpy as np
import matplotlib.pyplot as plt
import PIL
import copy
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers.pooling import MaxPool2D

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


# cnn_model = keras.models.Sequential([
#     keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(50, 50, 1)),
#     keras.layers.MaxPooling2D(),
#     keras.layers.BatchNormalization(),
#     keras.layers.Dropout(0.25),

#     keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
#     keras.layers.MaxPooling2D(),
#     keras.layers.BatchNormalization(),
#     keras.layers.Dropout(0.35),

#     keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
#     keras.layers.MaxPooling2D(),
#     keras.layers.BatchNormalization(),
#     keras.layers.Dropout(0.5),

#     keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu'),
#     keras.layers.MaxPooling2D(),
#     keras.layers.BatchNormalization(),
#     keras.layers.Dropout(0.75), fara asta si iese 0.77

#     keras.layers.Flatten(),
#     keras.layers.Dense(256, activation='relu'),
#     keras.layers.BatchNormalization(),
#     keras.layers.Dropout(0.3),
#     keras.layers.Dense(128, activation='relu'),
#     keras.layers.BatchNormalization(),
#     keras.layers.Dropout(0.5),
#     keras.layers.Dense(3, activation='softmax')
# ])

# cnn_model = keras.models.Sequential([
#     keras.layers.BatchNormalization(),
#     keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(50, 50, 1)),
#     keras.layers.MaxPooling2D(),
#     keras.layers.BatchNormalization(),
#     keras.layers.Dropout(0.25),

#     keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
#     keras.layers.MaxPooling2D(),
#     keras.layers.BatchNormalization(),
#     keras.layers.Dropout(0.35),

#     keras.layers.Flatten(),
#     keras.layers.Dense(256, activation='relu'),
#     keras.layers.BatchNormalization(),
#     keras.layers.Dropout(0.25),
#     keras.layers.Dense(3, activation='softmax')
# ])
#[1.3095098733901978, 0.7566666603088379]

# cnn_model = keras.models.Sequential([
#     keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(50, 50, 1)),
#     keras.layers.MaxPooling2D(),
#     keras.layers.BatchNormalization(),
#     keras.layers.Dropout(0.25),

#     keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
#     keras.layers.MaxPooling2D(),
#     keras.layers.BatchNormalization(),
#     keras.layers.Dropout(0.35),

#     keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
#     keras.layers.MaxPooling2D(),
#     keras.layers.BatchNormalization(),
#     keras.layers.Dropout(0.5),

#     keras.layers.Flatten(),
#     keras.layers.Dense(256, activation='relu'),
#     keras.layers.BatchNormalization(),
#     keras.layers.Dropout(0.3),
#     keras.layers.Dense(128, activation='relu'),
#     keras.layers.BatchNormalization(),
#     keras.layers.Dropout(0.7),
#     keras.layers.Dense(3, activation='softmax')
# ])
# [0.7642309665679932, 0.7848888635635376]
cnn_model = keras.models.Sequential([
    keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(50, 50, 1)),
    keras.layers.MaxPooling2D(),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.25),

    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.35),

    keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),

    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.7),
    keras.layers.Dense(3, activation='softmax')
])
# [0.6695919632911682, 0.7935555577278137]

# cnn_model = keras.models.Sequential([
#     keras.layers.BatchNormalization(),
#     keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(50, 50, 1)),
#     keras.layers.MaxPooling2D(),
#     keras.layers.BatchNormalization(),
#     keras.layers.Dropout(0.25),

#     keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
#     keras.layers.MaxPooling2D(),
#     keras.layers.BatchNormalization(),
#     keras.layers.Dropout(0.35),

#     keras.layers.Flatten(),
#     keras.layers.Dense(256, activation='relu'),
#     keras.layers.BatchNormalization(),
#     keras.layers.Dropout(0.25),
#     keras.layers.Dense(3, activation='softmax')
# ])
# print(train_images.shape)

# scad dropout pe convolutii -> creste pe train

train_images = train_images.reshape(len(train_data), 50, 50, 1)
validation_images = validation_images.reshape(len(validation_data), 50, 50, 1)
test_images = test_images.reshape(len(test_data), 50, 50, 1)

opt = keras.optimizers.Adam(learning_rate=1e-3) 
cnn_model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn_model.fit(train_images/255, train_labels, epochs=100, batch_size=64, validation_data=(validation_images/255, validation_labels))
print(cnn_model.evaluate(validation_images/255, validation_labels))

# loss: 0.9351 - accuracy: 0.7695

predicted_test_labels = cnn_model.predict(test_images/255)
output = zip(test_data, predicted_test_labels)
with open("submission.csv", "w") as g:
    g.write("id,label\n")
    for predict in output:
        g.write(str(predict[0]) + ',' + str(predict[1].argmax()) + '\n')

