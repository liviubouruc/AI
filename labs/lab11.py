# %%
import numpy as np
from sklearn import svm
# %%
def normalize_data(train_data, test_data, type_=None):
    # standard, l1, l2
    # N X F
    if type_ == 'standard':
        mean_train = np.mean(train_data, axis=0)
        std_train = np.std(train_data, axis=0)
        scaled_train_data = (train_data - mean_train) / std_train
        scaled_test_data = (test_data - mean_train) / std_train
    elif type_ == 'l1':
        norm_train = np.sum(np.abs(train_data), axis=1, keepdis=True) + 10 ** -8
        scaled_train_data = train_data / norm_train
        norm_test = np.sum(np.abs(test_data), axis=1, keepdims=True) + 10 ** -8
        scaled_test_data = test_data / norm_test
    elif type_ == 'l2':
        norm_train = np.sqrt(np.sum(train_data ** 2, axis=1, keepdims=True)) + 10 ** -8
        scaled_train_data = train_data / norm_train
        norm_test = np.sum(np.abs(test_data ** 2), axis=1, keepdims=True) + 10 ** -8
        scaled_test_data = test_data / norm_test
    else:
        raise Exception("Type not found")
# %%
class BagOfWords:
    def __init__(self):
        self.vocabulary = {} # word: id
        self.voc_len = 0
        self.words = []
        
    def build_vocabulary(self, data):
        for sentence in data:
            for word in sentence:
                if word not in self.vocabulary:
                    self.vocabulary[word] = len(self.vocabulary)
                    self.words.append(word)
            self.voc_len = len(self.vocabulary)
            
    def get_features(self, data):
        features = np.zeros((len(data), self.voc_len))
        
        for id_sen, sentence in enumerate(data):
            for word in sentence:
                if word in self.vocabulary:
                    features[id_sen, self.vocabulary[word]] += 1
        return features
# %%
train_sentences = np.load('datalab11/training_sentences.npy', allow_pickle=True)
train_labels = np.load('datalab11/training_labels.npy')

test_sentences = np.load('datalab11/test_sentences.npy', allow_pickle=True)
test_labels = np.load('datalab11/test_labels.npy')

print(train_sentences[0])
# %%
bag_of_words = BagOfWords()
bag_of_words.build_vocabulary(train_sentences)
# %%
train_features = bag_of_words.get_features(train_sentences)
# %%
test_features = bag_of_words.get_features(test_sentences)
# %%
train_features_norm, test_features_norm = normalize_data(train_features, test_features, type_='l2')
# %%
svm_model = svm.SVC(C=1, kernel='linear')
# %%
svm_model.fit(train_features_norm, train_labels)
# %%
