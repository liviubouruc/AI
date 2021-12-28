import numpy as np
import matplotlib.pyplot as plt
import PIL
import copy
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers.pooling import MaxPool2D
from keras_preprocessing.image import ImageDataGenerator


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
#     keras.layers.Dropout(0.5),
#     keras.layers.Dense(64, activation='relu'),
#     keras.layers.BatchNormalization(),
#     keras.layers.Dropout(0.7),
#     keras.layers.Dense(3, activation='softmax')
# ])
# [0.6695919632911682, 0.7935555577278137]

# train_images = train_images.reshape(len(train_data), 50, 50, 1)
# validation_images = validation_images.reshape(len(validation_data), 50, 50, 1)
# test_images = test_images.reshape(len(test_data), 50, 50, 1)


# from tensorflow.keras.applications import EfficientNetB7

# model = EfficientNetB7(weights='imagenet', include_top=False, input_shape=(50, 50, 3))  # , input_shape=(32, 32, 3)
# model.trainable = False

# classifier = keras.Sequential()
# classifier.add(model)
# classifier.add(keras.layers.GlobalMaxPooling2D(name="gap"))
# classifier.add(keras.layers.Dense(3, activation='softmax'))

# from keras.callbacks import ReduceLROnPlateau
# reduce_lr = ReduceLROnPlateau(monitor='val_acc', patience=5, factor=0.2, min_lr=0.000001)

# classifier.compile(optimizer=keras.optimizers.Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
# classifier.fit(train_images/255, train_labels, epochs=100, batch_size=64, validation_data=(validation_images/255, validation_labels), callbacks=[reduce_lr])
# print(classifier.evaluate(validation_images/255, validation_labels))



# from keras.applications.vgg16 import VGG16

# train_images = train_images.reshape(len(train_data), 50, 50, 1)
# validation_images = validation_images.reshape(len(validation_data), 50, 50, 1)
# test_images = test_images.reshape(len(test_data), 50, 50, 1)

# base_model = VGG16(input_shape=(50, 50, 3), include_top=False, weights='imagenet')

# for layer in base_model.layers:
#     layer.trainable = False

# x = keras.layers.Flatten()(base_model.output)
# x = keras.layers.Dense(512, activation='relu')(x)
# x = keras.layers.Dropout(0.3)(x)
# x = keras.layers.Dense(3, activation='softmax')(x)

# model = keras.models.Model(base_model.input, x)

# model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss='crossentropy_crossentropy', metrics=['acc'])
# model.fit(train_images/255, train_labels, epochs=100, batch_size=64, validation_data=(validation_images/255, validation_labels))


# opt = keras.optimizers.Adam(learning_rate=1e-3) 
# cnn_model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# cnn_model.fit(train_images/255, train_labels, epochs=100, batch_size=64, validation_data=(validation_images/255, validation_labels))
# print(cnn_model.evaluate(validation_images/255, validation_labels))

cnn_model = keras.models.Sequential([
    keras.layers.Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=(50, 50, 1)),
    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Dropout(0.25),

    keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Dropout(0.25),

    keras.layers.Flatten(),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(3, activation='softmax')
])

train_images = train_images.reshape(len(train_data), 50, 50, 1)
validation_images = validation_images.reshape(len(validation_data), 50, 50, 1)
test_images = test_images.reshape(len(test_data), 50, 50, 1)

opt = keras.optimizers.Adam(learning_rate=1e-3) 
cnn_model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn_model.fit(train_images/255, train_labels, epochs=100, batch_size=64, validation_data=(validation_images/255, validation_labels))
print(cnn_model.evaluate(validation_images/255, validation_labels))

# predicted_test_labels = cnn_model.predict(test_images/255)
# output = zip(test_data, predicted_test_labels)
# with open("submission.csv", "w") as g:
#     g.write("id,label\n")
#     for predict in output:
#         g.write(str(predict[0]) + ',' + str(predict[1].argmax()) + '\n')



# resnet_model = keras.models.Sequential()
# resnet_model.add(keras.layers.Lambda(lambda image: tf.image.resize(image, (224, 224))))
# resnet_model.add(ResNet50(include_top=False, pooling='max', weights='imagenet'))
# resnet_model.add(keras.layers.Flatten())
# resnet_model.add(keras.layers.Dense(1000, activation='relu'))
# resnet_model.add(keras.layers.Dropout(0.4))
# resnet_model.add(keras.layers.Dense(500, activation='relu'))
# resnet_model.add(keras.layers.Dropout(0.6))
# resnet_model.add(keras.layers.Dense(3, activation='softmax'))


# def preprocess_data(data):
#     data = keras.applications.resnet50.preprocess_input(data)
# #     labels = keras.utils.to_categorical(labels, 3)
#     return data

# train_images = preprocess_data(train_images)
# print(train_images.shape, train_labels.shape)
# validation_images = preprocess_data(validation_images)
train_labels = keras.utils.to_categorical(train_labels, 3)
validation_labels = keras.utils.to_categorical(validation_labels, 3)

# resnet_model = keras.models.Sequential()
# resnet_model.add(keras.layers.Lambda(lambda image: tf.image.resize(image, (224, 224))))
# resnet_model.add(ResNet50(include_top=False, pooling='max', weights='imagenet'))
# resnet_model.add(keras.layers.Flatten())
# resnet_model.add(keras.layers.Dense(1000, activation='relu'))
# resnet_model.add(keras.layers.Dropout(0.5))
# resnet_model.add(keras.layers.Dense(500, activation='relu'))
# resnet_model.add(keras.layers.Dropout(0.6))
# resnet_model.add(keras.layers.Dense(3, activation='softmax'))

# def scheduler(epoch, lr):
#     if epoch < 9:
#         return lr
#     elif epoch < 12:
#         return lr * np.exp(-1)
#     else:
#         return lr * np.exp(-3)

# callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
# opt = keras.optimizers.Adam(learning_rate=5e-5) 
# resnet_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
# resnet_model.fit(train_images/255, train_labels, epochs=20, batch_size=32, validation_data=(validation_images/255, validation_labels), callbacks=[callback])
# print(resnet_model.evaluate(validation_images/255, validation_labels))



# resnet_model = keras.models.Sequential()
# resnet_model.add(keras.layers.Lambda(lambda image: tf.image.resize(image, (224, 224))))
# resnet_model.add(ResNet50(include_top=False, pooling='max', weights='imagenet'))
# resnet_model.add(keras.layers.Flatten())
# resnet_model.add(keras.layers.Dense(1250, activation='relu'))
# resnet_model.add(keras.layers.Dropout(0.5))
# resnet_model.add(keras.layers.Dense(750, activation='relu'))
# resnet_model.add(keras.layers.Dropout(0.6))
# resnet_model.add(keras.layers.Dense(3, activation='softmax'))

# def scheduler(epoch, lr):
#     if epoch < 9:
#         return lr
#     elif epoch < 12:
#         return lr * np.exp(-1)
#     else:
#         return lr * np.exp(-2)

# callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
# opt = keras.optimizers.Adam(learning_rate=5e-5) 
# resnet_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
# resnet_model.fit(train_images/255, train_labels, epochs=20, batch_size=32, validation_data=(validation_images/255, validation_labels), callbacks=[callback])
# print(resnet_model.evaluate(validation_images/255, validation_labels))