import numpy as np
import matplotlib.pyplot as plt
import PIL
import copy
from sklearn.naive_bayes import MultinomialNB

train_data = np.loadtxt("train.txt", dtype=str)
train_images = []
train_labels = []
for data in train_data:
    train_image, train_label = data.split(',')

    pre_img = PIL.Image.open('train/%s' % train_image)
    img = copy.deepcopy(np.asarray(pre_img).flatten())
    train_images.append(img)
    pre_img.close()

    train_labels.append(train_label)

train_images = np.array(train_images)
train_labels = np.array(train_labels)

validation_data = np.loadtxt("validation.txt", dtype=str)
validation_images = []
validation_labels = []
for data in validation_data:
    validation_image, validation_label = data.split(',')

    pre_img = PIL.Image.open('validation/%s' % validation_image)
    img = copy.deepcopy(np.asarray(pre_img).flatten())
    validation_images.append(img)
    pre_img.close()

    validation_labels.append(validation_label)

validation_images = np.array(validation_images)
validation_labels = np.array(validation_labels)

def values_to_bins(x, bins):
    return np.digitize(x, bins)-1


num = 11
bins = np.linspace(0, 256, num)
train_images_bins = values_to_bins(train_images, bins)
test_images_bins = values_to_bins(validation_images, bins)
naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(train_images_bins, train_labels)
acc = naive_bayes_model.score(test_images_bins, validation_labels)
print("num bins = %d, accuracy = %f" % (num, acc))
