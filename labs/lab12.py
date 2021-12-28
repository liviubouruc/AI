# %%
import numpy as np
from sklearn.utils import shuffle
# %%
# load training data
training_data = np.load('datalab12/training_data.npy')
prices = np.load('datalab12/prices.npy')
# print the first 4 samples
print('The first 4 samples are:\n ', training_data[:4])
print('The first 4 prices are:\n ', prices[:4])
# shuffle
training_data, prices = shuffle(training_data, prices, random_state=0)
# %%
import sklearn.preprocessing as sk
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
# %%
def normalize(train_data, test_data):
    scaler = sk.StandardScaler()
    scaler.fit(train_data)
    scaled_train_data = scaler.transform(train_data)
    scaled_test_data = scaler.transform(test_data)

    return scaled_train_data, scaled_test_data
# %%
num_examples_fold = int(len(prices) // 3)
# %%
train_1, labels_1 = training_data[:num_examples_fold], prices[:num_examples_fold]
train_2, labels_2 = training_data[num_examples_fold : 2*num_examples_fold], prices[num_examples_fold : 2*num_examples_fold]
train_3, labels_3 = training_data[2*num_examples_fold:], prices[2*num_examples_fold:]
# %%
print(train_1.shape, train_2.shape, train_3.shape)
# %%
def normalize_train_and_eval(model, train_data, train_labels, test_data, test_labels):
    scaled_train_data, scaled_test_data = normalize(train_data, test_data)
    model.fit(scaled_train_data, scaled_test_data)
    predicted_prices = model.predict(scaled_test_data)
    return mean_squared_error(test_labels, predicted_prices), mean_absolute_error(test_labels, predicted_prices)
# %%
linear_model = LinearRegression()
mse_1, mae_1 = normalize_train_and_eval(linear_model, train_data=np.concatenate((train_1, train_2)), train_labels=np.concatenate((labels_1, labels_2)), test_data=train_3, test_labels=labels_3)
mse_2, mae_2 = normalize_train_and_eval(linear_model, train_data=np.concatenate((train_1, train_3)), train_labels=np.concatenate((labels_1, labels_3)), test_data=train_2, test_labels=labels_2)
mse_3, mae_3 = normalize_train_and_eval(linear_model, train_data=np.concatenate((train_2, train_3)), train_labels=np.concatenate((labels_2, labels_3)), test_data=train_1, test_labels=labels_1)

# %%
mae = (mae_1 + mae_2 + mae_3) / 3
mse = (mse_1 + mse_2 + mse_3) / 3
# asemenea pt mse