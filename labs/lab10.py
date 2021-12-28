import numpy as np
import matplotlib.pyplot as plt


class KnnClassifier:
    def __init__(self, train_images, train_labels):
        self.train_images = train_images
        self.train_labels = train_labels

    def classify_image(self, test_image, num_neightbors=3, metric='L2'):
        if metric == 'L1':
            distances = np.sum(np.abs(self.train_images - test_image), axis=1)
        elif metric == 'L2':
            distances = np.sqrt(np.sum((self.train_images - test_image)**2, axis=1))
        else:
            raise Exception("Metric not implemented!")

        sorted_indices = distances.argsort()
        nearest_indices = sorted_indices[:num_neightbors]
        nearest_labels = self.train_labels[nearest_indices]

        return np.bincount(nearest_labels).argmax()

    def classify_images(self, test_images, num_neighbors=3, metric='L2'):
        predicted_labels = [self.classify_image(image, num_neighbors, metric) for image in test_images]

        return np.array(predicted_labels)

def accuracy_score(ground_truth_labels, predicted_labels):
    return np.mean(ground_truth_labels == predicted_labels)

train_images = np.loadtxt("datalab8/train_images.txt")
train_labels = np.int32(np.loadtxt("datalab8/train_labels.txt"))
test_images = np.loadtxt("datalab8/test_images.txt")
test_labels = np.int32(np.loadtxt("datalab8/test_labels.txt"))

knn_classifier = KnnClassifier(train_images, train_labels)

predicted_labels = knn_classifier.classify_images(test_images, num_neighbors=3, metric='L2')

acc = accuracy_score(test_labels, predicted_labels)
print(acc)


def get_accuracies(train_images, train_labels, test_images, test_labels, metric='L2'):
    knn_classifier = KnnClassifier(train_images, train_labels)

    return [accuracy_score(test_labels, knn_classifier.classify_images(test_images, num_neighbors=num, metric=metric)) for num in [1, 3, 5, 7, 9]]


acc_l2 = get_accuracies(train_images, train_labels, test_images, test_labels, metric='L2')
acc_l1 = get_accuracies(train_images, train_labels, test_images, test_labels, metric='L1')

plt.plot([1, 3, 5, 7, 9], acc_l2)
plt.plot([1, 3, 5, 7, 9], acc_l1)
plt.legend(['L2', 'L1'])
plt.xlabel("num neighbors")
plt.ylabel("accuracy")
plt.show()
