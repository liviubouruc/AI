# %%
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
# %%
def compute_y(x, W, bias):
    # dreapta de decizie
    # [x, y] * [W[0], W[1]] + b = 0
    return (-x*W[0] - bias) / (W[1] + 1e-10)

def plot_decision_boundary(X, y , W, b, current_x, current_y):
    x1 = -0.5
    y1 = compute_y(x1, W, b)
    x2 = 0.5
    y2 = compute_y(x2, W, b)
    # sterge continutul ferestrei
    plt.clf()
    # ploteaza multimea de antrenare
    color = 'r'
    if(current_y == -1):
        color = 'b'
    plt.ylim((-1, 2))
    plt.xlim((-1, 2))
    plt.plot(X[y == -1, 0], X[y == -1, 1], 'b+')
    plt.plot(X[y == 1, 0], X[y == 1, 1], 'r+')
    # ploteaza exemplul curent
    plt.plot(current_x[0], current_x[1], color+'s')
    # afisarea dreptei de decizie
    plt.plot([x1, x2] ,[y1, y2], 'black')
    plt.show(block=False)
    plt.pause(0.3)


# %%
num_epochs = 70
lr = 0.1
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([-1, 1, 1, 1])
W = np.zeros(2)
b = 0
for e in range(num_epochs):
    X, Y = shuffle(X, Y)
    for t in range(X.shape[0]):
        y_hat = np.dot(X[t], W) + b 
        loss = ((y_hat - Y[t]) ** 2) / 2 
        acc = (np.sign(np.dot(X, W) + b) == Y).mean()
        print(f'epoch = {e}, loss = {loss}, acc = {acc}')

        dw = (y_hat - Y[t]) * X[t]
        db = y_hat - Y[t]
        W = W - lr*dw
        b = b - lr*db
        plot_decision_boundary(X, Y , W, b, X[t], Y[t])

# %%
def compute_y(x, W, bias):
    # dreapta de decizie
    # [x, y] * [W[0], W[1]] + b = 0
    return (-x*W[0] - bias) / (W[1] + 1e-10)

def plot_decision(X_, W_1, W_2, b_1, b_2):
    # sterge continutul ferestrei
    plt.clf()
    # ploteaza multimea de antrenare
    plt.ylim((-0.5, 1.5))
    plt.xlim((-0.5, 1.5))
    xx = np.random.normal(0, 1, (100000))
    yy = np.random.normal(0, 1, (100000))
    X = np.array([xx, yy]).transpose()
    X = np.concatenate((X, X_))
    _, _, _, output = forward(X, W_1, b_1, W_2, b_2)
    y = np.squeeze(np.round(output))
    plt.plot(X[y == 0, 0], X[y == 0, 1], 'b+')
    plt.plot(X[y == 1, 0], X[y == 1, 1], 'r+')
    plt.show(block=False)
    plt.pause(0.1)
# %%
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0, 1, 1, 0]]).T
print(Y.shape)
# %%
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def tanh_der(x):
    return 1 - np.tanh(x) ** 2
# %%
def forward(X, W_1, b_1, W_2, b_2):
    z_1 = np.dot(X, W_1) + b_1
    a_1 = np.tanh(z_1)

    z_2 = np.dot(a_1, W_2) + b_2
    a_2 = sigmoid(z_2)
    return z_1, a_1, z_2, a_2
# %%
def backwards(a_1, a_2, z_1, W_2, X, Y):
    num_samples = X.shape[0]
    dz_2 = a_2 - Y
    dw_2 = np.dot(a_1.T, dz_2) / num_samples
    db_2 = np.sum(dz_2, axis=0) / num_samples


    da_1 = np.dot(dz_2, W_2.T)
    dz_1 = da_1 * tanh_der(z_1)

    dw_1 = np.dot(X.T, dz_1) / num_samples
    db_1 = np.sum(dz_1, axis=0) / num_samples

    return dw_1, db_1, dw_2, db_2
# %%
epochs = 70
lr = 0.5
num_hidden_neurons = 5
num_output_neurons = 1
W_1 = np.random.normal(0, 1, (2, num_hidden_neurons))
b_1 = np.zeros((num_hidden_neurons))
W_2 = np.random.normal(0, 1, (num_hidden_neurons, 1))
b_2 = np.zeros((1))
# %%
for e in range(epochs):
    X, Y = shuffle(X, Y)
    z_1, a_1, z_2, a_2 = forward(X, W_1, b_1, W_2, b_2)
    loss = (-Y * np.log(a_2) - (1-Y) * np.log(1-a_2)).mean()
    acc = (np.round(a_2) == Y).mean()
    print(f'epoch = {e}, loss = {loss}, acc = {acc}')

    dw_1, db_1, dw_2, db_2 = backwards(a_1, a_2, z_1, W_2, X, Y)
    W_1 -= lr * dw_1
    b_1 -= lr * db_1
    W_2 -= lr * dw_2
    b_2 -= lr * db_2
    plot_decision(X, W_1, W_2, b_1, b_2)
# %%
