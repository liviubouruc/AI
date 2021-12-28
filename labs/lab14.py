# %%
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier # importul clase
# %%
perceptron_model = Perceptron(penalty=None, alpha=0.0001, fit_intercept=True,
max_iter=5, tol=1e-5, shuffle=True, eta0=0.1, early_stopping=False,
validation_fraction=0.1, n_iter_no_change=5) 
# %%
import matplotlib.pyplot as plt 
# %%
def plot3d_data(X, y):
    ax = plt.axes(projection='3d')
    ax.scatter3D(X[y == -1, 0], X[y == -1, 1], X[y == -1, 2],'b')
    ax.scatter3D(X[y == 1, 0], X[y == 1, 1], X[y == 1, 2],'r')
    plt.show()

def plot3d_data_and_decision_function(X, y, W, b): 
    ax = plt.axes(projection='3d')
    # create x,y
    xx, yy = np.meshgrid(range(10), range(10))
    # calculate corresponding z
    # [x, y, z] * [coef1, coef2, coef3] + b = 0
    zz = (-W[0] * xx - W[1] * yy - b) / W[2]
    ax.plot_surface(xx, yy, zz, alpha=0.5) 
    ax.scatter3D(X[y == -1, 0], X[y == -1, 1], X[y == -1, 2],'b')
    ax.scatter3D(X[y == 1, 0], X[y == 1, 1], X[y == 1, 2],'r')
    plt.show()

# %%
# incarcarea datelor de antrenare
X = np.loadtxt('./datalab14/3d-points/x_train.txt')
y = np.loadtxt('./datalab14/3d-points/y_train.txt', 'int') 

plot3d_data(X, y)
# incarcarea datelor de testare
X_test = np.loadtxt('./datalab14/3d-points/x_test.txt')
y_test = np.loadtxt('./datalab14/3d-points/y_test.txt', 'int') 

# %%
perceptron_model.fit(X, y)
print(perceptron_model.score(X, y))
print(perceptron_model.score(X_test, y_test))
W = perceptron_model.coef_
b = perceptron_model.intercept_
epochs = perceptron_model.n_iter_
print(W)
print(b)
plot3d_data_and_decision_function(X_test, y_test, W[0], b)
# %%
X = np.loadtxt('./datalab14/MNIST/train_images.txt')
y = np.loadtxt('./datalab14/MNIST/train_labels.txt', 'int') 

# incarcarea datelor de testare
X_test = np.loadtxt('./datalab14/MNIST/test_images.txt')
y_test = np.loadtxt('./datalab14/MNIST/test_labels.txt', 'int')
# %%
def normalize(train_data, test_data):
    scaler = StandardScaler()
    scaler.fit(train_data)
    scaled_train_data = scaler.transform(train_data)
    scaled_test_data = scaler.transform(test_data)

    return scaled_train_data, scaled_test_data
# %%
X, X_test = normalize(X, X_test)
# %%
mlp_classifier_model = MLPClassifier(hidden_layer_sizes=(1, ),
activation='tanh', learning_rate_init=0.01, max_iter=200, momentum=0.0) # 0.253, 0.186
# %%
mlp_classifier_model = MLPClassifier(hidden_layer_sizes=(10, ),
activation='tanh', learning_rate_init=0.01, max_iter=200, momentum=0.0) # 1.0, 0.822
# %%
mlp_classifier_model = MLPClassifier(hidden_layer_sizes=(10, ),
activation='tanh', learning_rate_init=0.00001, max_iter=200, momentum=0.0) # 0.352, 0.326
# %%
mlp_classifier_model = MLPClassifier(hidden_layer_sizes=(10, ),
activation='tanh', learning_rate_init=10, max_iter=200, momentum=0.0) # 0.443, 0.452
# %%
mlp_classifier_model = MLPClassifier(hidden_layer_sizes=(10, ),
activation='tanh', learning_rate_init=0.01, max_iter=20, momentum=0.0) # 0.984, 0.832
# %%
mlp_classifier_model = MLPClassifier(hidden_layer_sizes=(10, 10),
activation='tanh', learning_rate_init=0.01, max_iter=2000, momentum=0.0) # 0.95, 0.796
# %%
mlp_classifier_model = MLPClassifier(hidden_layer_sizes=(10, 10),
activation='relu', learning_rate_init=0.01, max_iter=2000, momentum=0.0) # 0.998, 0.816
# %%
mlp_classifier_model = MLPClassifier(hidden_layer_sizes=(100, 100),
activation='relu', learning_rate_init=0.01, max_iter=2000, momentum=0.0) # 1.0, 0.906
# %%
mlp_classifier_model = MLPClassifier(hidden_layer_sizes=(100, 100),
activation='relu', learning_rate_init=0.01, max_iter=2000, momentum=0.9) # 1.0, 0.912
# %%
mlp_classifier_model = MLPClassifier(hidden_layer_sizes=(100, 100),
activation='relu', learning_rate_init=0.01, max_iter=2000, momentum=0.9, alpha=0.005) # 1.0, 0.898
# %%
mlp_classifier_model.fit(X, y)
print(mlp_classifier_model.score(X, y))
print(mlp_classifier_model.score(X_test, y_test))
# %%
