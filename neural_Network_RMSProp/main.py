
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('C:/Users/pc/Desktop/firstdataset/train.csv')

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape

Y_train

def init_params(nbr1):
    W1 = np.random.rand(nbr1, 784) - 0.5
    b1 = np.random.rand(nbr1, 1) - 0.5
    W2 = np.random.rand(10, nbr1) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2


def ReLU(Z):
    return np.maximum(Z, 0)


def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2


def rmsProp(W1, W2, b1, b2, dW1, dW2, db1, db2, sigma, beta , epsilon, V1, V2, V3, V4):

    W1 = W1 - sigma / np.sqrt(V1 + epsilon) * dW1
    b1 = b1 - sigma / np.sqrt(V2 + epsilon) * db1
    W2 = W2 - sigma / np.sqrt(V3 + epsilon) * dW2
    b2 = b2 - sigma / np.sqrt(V4 + epsilon) * db2

    V1 = beta * V1 + (1-beta) * dW1 ** 2
    V2 = beta * V2 + (1-beta) * db1 ** 2
    V3 = beta * V3 + (1-beta) * dW2 ** 2
    V4 = beta * V4 + (1-beta) * db2 ** 2

    return W1, W2, b1, b2, V1, V2, V3, V4

def get_predictions(A2):
    return np.argmax(A2, 0)


def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y ,nbr1 ,  alpha,gama ,  beta, iterations):
    W1, b1, W2, b2 = init_params(nbr1)
    M1 = np.zeros_like(W1)
    M2 = np.zeros_like(b1)
    M3 = np.zeros_like(W2)
    M4 = np.zeros_like(b2)

    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, W2, b1, b2, M1, M2, M3, M4 = rmsProp(W1, W2, b1, b2, dW1, dW2, db1, db2, alpha, gama ,  beta, M1, M2, M3, M4)

        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))

    return W1, b1, W2, b2

nbr1 = int(input("Entrer le nombre de neurones pour la couche cachée : "))
iteration = int(input("Entrer le nombre des iteration : "))

W1, b1, W2, b2 = gradient_descent(X_train, Y_train, nbr1, alpha=0.10, beta=0.9 ,gama=0.10 , iterations=iteration)


def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions


def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Label: ", label)
    print("Prediction: ", prediction)

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


test_prediction(25, W1, b1, W2, b2)
test_prediction(30, W1, b1, W2, b2)
test_prediction(35, W1, b1, W2, b2)
test_prediction(40, W1, b1, W2, b2)

dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
print(get_accuracy(dev_predictions, Y_dev))



