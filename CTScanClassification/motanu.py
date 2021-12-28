import datetime
import os

import cv2
import numpy as np
import tensorflow as tf
import keras
import efficientnet.keras as efn
import efficientnet.tfkeras
import matplotlib as plt
from keras.callbacks import ReduceLROnPlateau
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tqdm.keras import TqdmCallback


def accuracy_score(y_true, y_pred):
    return (y_true == y_pred).mean()


# Learning rates
# start_lr = 0.9
# exp_decay = 0.9817


# Learning Rate Scheduler
def scheduler(epoch, lr):
    if epoch < 9:
        return lr
    elif epoch < 12:
        return lr * np.exp(-1)
    else:
        return lr * np.exp(-2)

# Learning Rate Scheduler
# def scheduler(epoch):
#     return start_lr * (exp_decay ** epoch)


def predictTestAndSave(classifier):
    print("Saving..")

    test_path = 'test/'
    test_name = 'submission'

    test_images_names = [f for f in os.listdir(test_path) if os.path.isfile(os.path.join(test_path, f))]
    tst_images = []

    for tst_image in test_images_names:
        img = cv2.imread(test_path + tst_image, cv2.IMREAD_GRAYSCALE)
        img = (img - np.mean(img)) / np.std(img)
        tst_images.append(img)

    # test_images = np.array(tst_images)
    # test_images = np.repeat(test_images[:, :, :, np.newaxis], 3, -1)
    test_images = np.expand_dims(tst_images, axis=3)

    # test_images = test_images / 255.0

    test_prediction = classifier.predict(test_images)

    tst_file = open("submissions/" + test_name + "-" + datetime.datetime.now().strftime("%Y%m%d-%H%M") + '.csv', 'w')
    tst_file.write("id,label\n")

    for i in range(0, len(tst_images)):
        tst_file.write(str(test_images_names[i]) + "," + str(np.argmax(test_prediction[i])) + "\n")

    tst_file.close()
    print("Saved!")


IMG_SIZE = 32
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
print("Loading images..")

train_labels = np.loadtxt('train_true_labels.txt', 'int').astype(int)
# train_labels = np.loadtxt('train_true_labels_flipped.txt', 'int').astype(int)
validation_labels = np.loadtxt('validation_true_labels.txt', 'int').astype(int)

train_path = 'train/'
# train_path = 'flipped/'
validation_path = 'validation/'

train_images_names = [f for f in os.listdir(train_path) if os.path.isfile(os.path.join(train_path, f))]
validation_images_names = [f for f in os.listdir(validation_path) if os.path.isfile(os.path.join(validation_path, f))]


t_images, v_images = [], []

for t_image in train_images_names:
    img = cv2.imread(train_path + t_image, cv2.IMREAD_GRAYSCALE)
    img = (img - np.mean(img)) / np.std(img)
    t_images.append(img)

for v_image in validation_images_names:
    img = cv2.imread(validation_path + v_image, cv2.IMREAD_GRAYSCALE)
    img = (img - np.mean(img)) / np.std(img)
    v_images.append(img)


# train_images = np.array(t_images)
# validation_images = np.array(v_images)
# train_images = np.repeat(train_images[:, :, :, np.newaxis], 3, -1)
# validation_images = np.repeat(validation_images[:, :, :, np.newaxis], 3, -1)

train_images = np.expand_dims(t_images, axis=3)
validation_images = np.expand_dims(v_images, axis=3)

print(train_images.shape)

# train_images = np.array(t_images).squeeze()
# validation_images = np.array(v_images).squeeze()

# train_images = train_images / 255.0
# validation_images = validation_images / 255.0

print('Images loaded!')


# print("Scaling!")
# scaler = StandardScaler()
# scaler.fit(train_images)
# train_images_scaled = scaler.transform(train_images)

# print("Augmenting data..")
# augmented_data_gen = ImageDataGenerator(
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )
#
# train_images = augmented_data_gen.flow(train_images, train_labels, batch_size=1)
# print("Augmented data!")

print("Creating model!")
# classifier = SVC(C=1, kernel='poly', gamma='auto')

# data_augmentation = keras.Sequential([
#     keras.layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(32, 32, 1)),
#     keras.layers.experimental.preprocessing.RandomRotation(0.1),
#     keras.layers.experimental.preprocessing.RandomZoom(0.1)
# ])

# 0.8522%
# classifier = keras.Sequential([
#     keras.layers.Conv2D(64, (3, 3), input_shape=(32, 32, 1), activation='relu'),
#     keras.layers.MaxPooling2D((2, 2)),
#     keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     keras.layers.MaxPooling2D((2, 2)),
#     keras.layers.Conv2D(128, (3, 3), activation='relu'),
#     keras.layers.Dropout(0.2),
#     keras.layers.Flatten(),
#     keras.layers.Dense(128, activation='relu'),
#     keras.layers.Dense(9, activation='softmax')
# ])

# 0.8831%
# classifier = keras.Sequential([
#     keras.layers.Conv2D(50, (7, 7), input_shape=(32, 32, 1), activation='relu'),
#     keras.layers.Conv2D(75, (3, 3), activation='relu'),
#     keras.layers.MaxPooling2D((2, 2)),
#     keras.layers.Dropout(0.25),
#     keras.layers.Conv2D(125, (3, 3), activation='relu'),
#     keras.layers.MaxPooling2D((2, 2)),
#     keras.layers.Dropout(0.25),
#     keras.layers.Flatten(),
#     keras.layers.Dense(500, activation='relu'),
#     keras.layers.Dropout(0.4),
#     keras.layers.Dense(250, activation='relu'),
#     keras.layers.Dropout(0.3),
#     keras.layers.Dense(9, activation='softmax')
# ])

# classifier = keras.Sequential([
#     keras.layers.Conv2D(64, (3, 3), input_shape=(32, 32, 1), activation='relu'),
#     keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
#     keras.layers.MaxPooling2D((2, 2)),
#     keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
#     keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
#     keras.layers.MaxPooling2D((2, 2)),
#     keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
#     keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
#     keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
#     keras.layers.MaxPooling2D((2, 2)),
#     keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
#     keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
#     keras.layers.MaxPooling2D((2, 2)),
#     keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
#     keras.layers.GlobalAveragePooling2D(),
#     keras.layers.Dense(9, activation='softmax')
# ])

# 0.8826%
# classifier = keras.Sequential([
#     keras.layers.Conv2D(20, (5, 5), input_shape=(32, 32, 1), activation='relu', padding='same'),
#     keras.layers.MaxPool2D(),
#     keras.layers.Dropout(0.25),
#     keras.layers.Conv2D(50, (5, 5), activation='relu', padding='same'),
#     keras.layers.MaxPool2D(),
#     keras.layers.Dropout(0.25),
#     keras.layers.Flatten(),
#     keras.layers.Dense(500, activation='relu'),
#     keras.layers.Dropout(0.4),
#     keras.layers.Dense(9, activation='softmax')
# ])


# EfficientNetB7
# model = efn.EfficientNetB7(weights='imagenet', include_top=False, input_shape=(32, 32, 3))  # , input_shape=(32, 32, 3)
# model.trainable = False

# classifier = keras.Sequential()
# classifier.add(model)
# classifier.add(keras.layers.GlobalMaxPooling2D(name="gap"))
# classifier.add(keras.layers.Dense(9, activation='softmax'))

# classifier.compile(optimizer=keras.optimizers.Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

# Working VGG 16
# from keras.applications.vgg16 import VGG16
#
# base_model = VGG16(input_shape=(32, 32, 3), include_top=False, weights='imagenet')
#
# for layer in base_model.layers:
#     layer.trainable = False
#
# x = keras.layers.Flatten()(base_model.output)
# x = keras.layers.Dense(512, activation='relu')(x)
# x = keras.layers.Dropout(0.3)(x)
# x = keras.layers.Dense(1, activation='softmax')(x)
#
# model = keras.models.Model(base_model.input, x)
#
# model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss='crossentropy_crossentropy', metrics=['acc'])

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    brightness_range=(0.4, 1.0)
)

# datagen.fit(train_images)

# LeNet-5
# classifier = keras.models.Sequential([
#     keras.layers.Conv2D(6, (5, 5), input_shape=(32, 32, 1), activation='tanh'),
#     keras.layers.AveragePooling2D((2, 2)),
#     keras.layers.Conv2D(16, (5, 5), activation='tanh'),
#     keras.layers.AveragePooling2D((2, 2)),
#     keras.layers.Flatten(),
#     keras.layers.Dense(120, activation='tanh'),
#     keras.layers.Dense(84, activation='tanh'),
#     keras.layers.Dense(9, activation='softmax')
# ])

# classifier = keras.models.Sequential([
#     keras.layers.Conv2D(6, (5, 5), input_shape=(32, 32, 1), activation='relu'),
#     keras.layers.BatchNormalization(),
#     keras.layers.MaxPooling2D((2, 2)),
#     keras.layers.Conv2D(12, (5, 5), activation='relu'),
#     keras.layers.BatchNormalization(),
#     keras.layers.MaxPooling2D((2, 2)),
#     keras.layers.Flatten(),
#     keras.layers.Dense(320, activation='tanh'),
#     keras.layers.Dense(9, activation='softmax')
# ])

# classifier = keras.Sequential()

# classifier.add(keras.layers.Conv2D(20, (5, 5), input_shape=(32, 32, 1), padding='same'))
# classifier.add(keras.layers.BatchNormalization())
# classifier.add(keras.layers.Activation('relu'))
# classifier.add(keras.layers.Dropout(0.2))
#
# classifier.add(keras.layers.MaxPooling2D((2, 2)))
# classifier.add(keras.layers.BatchNormalization())
#
# classifier.add(keras.layers.Conv2D(50, (5, 5), padding='same'))
# classifier.add(keras.layers.BatchNormalization())
# classifier.add(keras.layers.Activation('relu'))
# classifier.add(keras.layers.Dropout(0.2))
#
# classifier.add(keras.layers.MaxPooling2D((2, 2)))
# classifier.add(keras.layers.BatchNormalization())
#
# classifier.add(keras.layers.Flatten())
# classifier.add(keras.layers.Dense(500))
# classifier.add(keras.layers.BatchNormalization())
# classifier.add(keras.layers.Dense(9, activation='softmax'))

# classifier = keras.Sequential([
#     keras.layers.Conv2D(50, (5, 5), input_shape=(32, 32, 1), activation='relu'),
#     keras.layers.BatchNormalization(),
#     keras.layers.Conv2D(75, (3, 3), activation='relu'),
#     keras.layers.BatchNormalization(),
#     keras.layers.MaxPooling2D((2, 2)),
#     keras.layers.Dropout(0.25),
#     keras.layers.Conv2D(125, (3, 3), activation='relu'),
#     keras.layers.MaxPooling2D((2, 2)),
#     keras.layers.Dropout(0.25),
#     keras.layers.Flatten(),
#     keras.layers.Dense(500, activation='relu'),
#     keras.layers.Dropout(0.4),
#     keras.layers.Dense(250, activation='relu'),
#     keras.layers.Dropout(0.3),
#     keras.layers.Dense(9, activation='softmax')
# ])

# base_model = keras.applications.resnet50.ResNet50(weights=None, include_top=False, input_shape=(32, 32, 3))
# x = base_model.output
# x = keras.layers.GlobalAveragePooling2D()(x)
# x = keras.layers.Dropout(0.7)(x)
# predictions = keras.layers.Dense(9, activation='softmax')(x)
# classifier = keras.Model(inputs=base_model.inputs, outputs=predictions)


# DE ANTRENAT CU MAI MULTE EPOCI (~50)
# classifier = keras.Sequential([
#     keras.layers.Conv2D(32, (7, 7), input_shape=(32, 32, 1), activation='relu', padding='same'),
#     keras.layers.Conv2D(32, (7, 7), activation='relu'),
#     keras.layers.MaxPooling2D((2, 2)),
#     keras.layers.Dropout(0.4),
#
#     keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
#     keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     keras.layers.MaxPooling2D((2, 2)),
#     keras.layers.Dropout(0.4),
#
#     keras.layers.Flatten(),
#     keras.layers.Dense(512, activation='relu'),
#     keras.layers.Dropout(0.7),  # 0.5
#     keras.layers.Dense(9, activation='softmax')
# ])

# classifier = keras.Sequential([
#
#     keras.layers.Conv2D(96, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 1)),
#     keras.layers.Dropout(0.2),
#
#     keras.layers.Conv2D(96, (3, 3), activation='relu', padding='same'),
#     keras.layers.Conv2D(96, (3, 3), activation='relu', padding='same', strides=2),
#     keras.layers.Dropout(0.5),
#
#     keras.layers.Conv2D(192, (3, 3), activation='relu', padding='same'),
#     keras.layers.Conv2D(192, (3, 3), activation='relu', padding='same'),
#     keras.layers.Conv2D(192, (3, 3), activation='relu', padding='same', strides=2),
#     keras.layers.Dropout(0.5),
#
#     keras.layers.Conv2D(192, (3, 3), activation='relu', padding='same'),
#     keras.layers.Conv2D(192, (1, 1), activation='relu', padding='valid'),
#     keras.layers.Dropout(0.5),
#
#     keras.layers.Flatten(),
#     keras.layers.Dense(500, activation='relu'),
#     keras.layers.Dropout(0.5),
#
#     keras.layers.Dense(9, activation='softmax'),
# ])

# classifier = keras.Sequential([
#     keras.layers.Dense(256, activation='relu', input_shape=(256,)),
#     keras.layers.Dense(1024, activation='relu'),
#     keras.layers.Dense(9, activation='softmax')
# ])

# classifier = keras.Sequential([
#     keras.layers.Conv2D(50, (7, 7), input_shape=(32, 32, 1), activation='relu'),
#     keras.layers.Conv2D(75, (3, 3), activation='relu'),
#     keras.layers.MaxPooling2D((2, 2)),
#     keras.layers.Dropout(0.25),
#     keras.layers.Conv2D(125, (3, 3), activation='relu'),
#     keras.layers.MaxPooling2D((2, 2)),
#     keras.layers.Dropout(0.25),
#     keras.layers.Flatten(),
#     keras.layers.Dense(500, activation='relu'),
#     keras.layers.Dropout(0.4),
#     keras.layers.Dense(250, activation='relu'),
#     keras.layers.Dropout(0.3),
#     keras.layers.Dense(9, activation='softmax')
# ])


# classifier.compile(optimizer=keras.optimizers.Adam(0.0005), loss='sparse_categorical_crossentropy', metrics=['accuracy'])  # 0.0001

# reduce_lr = ReduceLROnPlateau(monitor='val_acc', patience=5, factor=0.2, min_lr=0.000001)

# print("Started training!")

# callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
# classifier.fit(train_images, train_labels, epochs=50, validation_data=(validation_images, validation_labels), verbose=2, callbacks=[reduce_lr])
# classifier.fit_generator(datagen.flow(train_images, train_labels, batch_size=32), steps_per_epoch=len(train_images) // 32, epochs=150, validation_data=(validation_images, validation_labels), verbose=2)  # callbacks=[callback], BATCH_SIZE = 32
# plot_hist(hist)

print("Predicting!")
# prediction = classifier.predict(validation_images)

# score_mean = accuracy_score(validation_labels, prediction)
# validation_loss, validation_acc = classifier.evaluate(validation_images, validation_labels)

# print("Prediction score by mean: " + str(score_mean * 100) + "%")
# print("Prediction score: " + str(validation_acc * 100) + "%")


# Save to file
# predictTestAndSave(classifier)