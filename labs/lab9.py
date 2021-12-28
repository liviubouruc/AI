import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB

train_images = np.loadtxt('datalab8/train_images.txt')
train_labels = np.loadtxt('datalab8/train_labels.txt', 'int')

test_images = np.loadtxt('datalab8/test_images.txt')
test_labels = np.loadtxt('datalab8/test_labels.txt', 'int')

print(train_images.shape)
# print(train_labels.shape)

image = train_images[0]
image = np.reshape(image, (28, 28))
plt.imshow(image.astype(np.uint8), cmap='gray')
plt.show()

# bins = np.linspace(0, 256, 5)
# np.digitize(5, bins)

def values_to_bins(x, bins):
    return np.digitize(x, bins)-1

bins = np.linspace(0, 256, 5)
train_images_bins = values_to_bins(train_images, bins)
test_images_bins = values_to_bins(test_images, bins)

naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(train_images_bins, train_labels)

predicted_labels = naive_bayes_model.predict(test_images_bins)

accuracy = np.mean(predicted_labels == test_labels)
print(accuracy)

for num in [3, 5, 7, 9, 11]:
    bins = np.linspace(0, 256, num)
    train_images_bins = values_to_bins(train_images, bins)
    test_images_bins = values_to_bins(test_images, bins)
    naive_bayes_model = MultinomialNB()
    naive_bayes_model.fit(train_images_bins, train_labels)
    acc = naive_bayes_model.score(test_images_bins, test_labels)
    print("num bins = %d, accuracy = %f" % (num, acc))

bins = np.linspace(0, 256, 7)
train_images_bins = values_to_bins(train_images, bins)
test_images_bins = values_to_bins(test_images, bins)
naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(train_images_bins, train_labels)
predicted_labels = naive_bayes_model.predict(test_images_bins)

misclassified_indices = np.where(test_labels != predicted_labels)[0]

for idx in misclassified_indices[:10]:
    img = test_images[idx].reshape(28, 28)
    plt.imshow(img, cmap='gray')
    plt.show()
    print("predicted label = %d, ground-truth label = %d" % (predicted_labels[idx], test_labels[idx]))

def confusion_matrix(predicted_labels, ground_truth_labels):
    num_classes = max(predicted_labels.max(), ground_truth_labels.max()) + 1
    conf_mat = np.zeros((num_classes, num_classes))
    for idx in range(len(predicted_labels)):
        conf_mat[ground_truth_labels[idx], predicted_labels[idx]] += 1
    return conf_mat

print(predicted_labels, test_labels)

conf_mat = confusion_matrix(predicted_labels, test_labels)
print(conf_mat)
plt.imshow(conf_mat, cmap='gray')
plt.show()
