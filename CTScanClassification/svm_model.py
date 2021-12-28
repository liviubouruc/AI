import numpy as np
import matplotlib.pyplot as plt
import PIL
import copy
from sklearn import svm

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

test_data = np.loadtxt("test.txt", dtype=str)
test_images = []
for test_image in test_data:
    pre_img = PIL.Image.open('test/%s' % test_image)
    img = copy.deepcopy(np.asarray(pre_img).flatten())
    test_images.append(img)
    pre_img.close()

test_images = np.array(test_images).astype(int)

def accuracy_score(ground_truth_labels, predicted_labels):
    return np.mean(ground_truth_labels == predicted_labels)

svm_classifier = svm.SVC(C=3.5)
svm_classifier.fit(train_images, train_labels)
predicted = svm_classifier.predict(validation_images)
acc = accuracy_score(validation_labels, predicted)
print(acc)


predicted_test_labels = svm_classifier.predict(test_images)
output = zip(test_data, predicted_test_labels)
with open("submission.csv", "w") as g:
    g.write("id,label\n")
    for predict in output:
        g.write(str(predict[0]) + ',' + str(predict[1]) + '\n')