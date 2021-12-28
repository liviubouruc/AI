import numpy as np
import matplotlib.pyplot as plt
import PIL
import copy
from sklearn.neighbors import KNeighborsClassifier


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

train_images = np.array(train_images).astype(int)
train_labels = np.array(train_labels).astype(int)

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

validation_images = np.array(validation_images).astype(int)
validation_labels = np.array(validation_labels).astype(int)

test_data = np.loadtxt("test.txt", dtype=str)
test_images = []
for test_image in test_data:
    pre_img = PIL.Image.open('test/%s' % test_image)
    img = copy.deepcopy(np.asarray(pre_img).flatten())
    test_images.append(img)
    pre_img.close()

test_images = np.array(test_images).astype(int)


# def accuracy_score(ground_truth_labels, predicted_labels):
#      return np.mean(ground_truth_labels == predicted_labels)

# knn_classifier = KNeighborsClassifier(n_neighbors=13)
# knn_classifier.fit(train_images, train_labels)
# predicted_labels = knn_classifier.predict(validation_images)
# acc = accuracy_score(validation_labels, predicted_labels)
# print(acc)

class KnnClassifier:
    def __init__(self, train_images, train_labels):
        self.train_images = train_images
        self.train_labels = train_labels

    def classify_image(self, test_image, num_neighbors=3, metric='L2'):
        if metric == 'L1':
            distances = np.sum(np.abs(self.train_images - test_image), axis=1)
        elif metric == 'L2':
            distances = np.sqrt(np.sum((self.train_images - test_image)**2, axis=1))
        else:
            raise Exception("Metric not implemented!")

        sorted_indices = distances.argsort()
        nearest_indices = sorted_indices[:num_neighbors]
        nearest_labels = self.train_labels[nearest_indices]
        return np.bincount(nearest_labels).argmax()

    def classify_images(self, images, num_neighbors=3, metric='L2'):
        predicted_labels = [self.classify_image(image, num_neighbors, metric) for image in images]
        return np.array(predicted_labels)


def accuracy_score(ground_truth_labels, predicted_labels):
    return np.mean(ground_truth_labels == predicted_labels)


knn_classifier = KnnClassifier(train_images, train_labels)
# predicted_labels = knn_classifier.classify_images(validation_images, num_neighbors=11, metric='L1')
# acc = accuracy_score(validation_labels, predicted_labels)
# print(acc)


predicted_test_labels = knn_classifier.classify_images(test_images, num_neighbors=11, metric='L1')
output = zip(test_data, predicted_test_labels)
with open("submission.csv", "w") as g:
    g.write("id,label\n")
    for predict in output:
        g.write(str(predict[0]) + ',' + str(predict[1]) + '\n')
