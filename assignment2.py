import numpy as np
import matplotlib.pyplot as plt
from functions import *

n = 10000
k = 10
d = 3072


def one_hot_representation(label):
    labelY = np.zeros((k, n))
    for index in range(len(label)):
        labelY[label[index]][index] = 1
    return labelY


def preprocess(dataset):
    mean = np.mean(dataset, axis=1)
    std = np.std(dataset, axis=1)
    norm = (dataset - mean[:, None]) / std[:, None]
    return norm


def EvaluateClassifier(X, W, b):
    # S=W*X+b; S (k*n)
    S = np.dot(W, X) + b
    # P = e(S)/(1 ^ T*e(S))(k*n); each colum is the pobability of the class
    P = softmax(S)
    return P


def ComputeCost(X, Y, W, b, lam):
    P = EvaluateClassifier(X, W, b)
    # J = sum(-log(Y ^ T*P))/N+lambda*r; sum of loss
    cross_ent = -np.log(np.dot(Y.T, P))
    reg = lam * np.sum(np.square(W))
    J = np.sum(cross_ent.diagonal()) / X.shape[1] + reg
    return J


def ComputeGradients(X, Y, P, W, lamda):
    # grad_W(k*d); grad_b(k, 1)

    G = -(Y - P)
    grad_W = np.dot(G, X.T) / G.shape[1] + 2 * lamda * W
    grad_b = np.dot(G, np.ones(G.shape[1]).reshape(-1, 1)) / G.shape[1]
    return grad_W, grad_b


def ComputeAccuracy(X, y, W, b):
    P = EvaluateClassifier(X, W, b)
    # k: a vector shows the row index of max element on P in each column
    # k(1*n)
    k = np.amax(P, axis=0)
    # length of the vector which y = k is the total accuacy
    acc = 1 - np.count_zero(k - y) / len(k)
    return acc


def MiniBatchGD(trainX, trainY, valX, valY, W, b, lam, n_batch, eta, n_epochs):
    trainJ = []
    valJ = []

    for i in range(n_epochs):

        # shuffle the input images
        shuffle_idx = np.random.permutation(trainX.shape[1])
        shuffledX = trainX[:, shuffle_idx]
        shuffledY = trainY[:, shuffle_idx]

        trainCost = ComputeCost(shuffledX, shuffledY, W, b, lam)
        trainJ.append(trainCost)
        valCost = ComputeCost(valX, valY, W, b, lam)
        valJ.append(valCost)
        print(trainCost)
        print(valCost)

        for j in range(n // n_batch):
            j_start = j * n_batch
            j_end = (j + 1) * n_batch
            Xbatch = shuffledX[:, j_start:j_end]
            Ybatch = shuffledY[:, j_start:j_end]

            P = EvaluateClassifier(Xbatch, W, b)
            grad_W, grad_b = ComputeGradients(Xbatch, Ybatch, P, W, lam)
            W -= eta * grad_W
            b -= eta * grad_b

    plt.plot(trainJ, 'g', label="training loss")
    plt.plot(valJ, 'r', label="validation loss")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Traning and Validation Loss')
    plt.legend(loc="best")
    plt.show()
    return W, b


def main():
    trainSet = LoadBatch('cifar-10-batches-py/data_batch_1')
    valSet = LoadBatch('cifar-10-batches-py/data_batch_2')
    testSet = LoadBatch('cifar-10-batches-py/test_batch')

    # extract input images; from  n*d to d*n
    trainX = np.transpose(trainSet[bytes("data", "utf-8")])
    valX = np.transpose(valSet[bytes("data", "utf-8")])
    testX = np.transpose(testSet[bytes("data", "utf-8")])

    # extract labels; 1*10000
    trainy = trainSet[bytes("labels", "utf-8")]
    valy = valSet[bytes("labels", "utf-8")]
    testy = testSet[bytes("labels", "utf-8")]

    trainY = one_hot_representation(trainy)
    valY = one_hot_representation(valy)

    # normalize input data
    trainX = preprocess(trainX)
    valX = preprocess(valX)

    # initialize w and b
    W = np.random.normal(0, 0.01, (k, d))
    b = np.random.normal(0, 0.01, (k, 1))

    # parameters
    n_batch = 100
    eta = 0.001
    n_epochs = 40
    lam = 1

    W, b = MiniBatchGD(
        trainX, trainY, valX, valY, W, b, lam, n_batch, eta, n_epochs)

    acc_train = ComputeAccuracy(trainX, trainy, W, b)
    acc_test = ComputeAccuracy(testX, testy, W, b)
    print(acc_train)
    print(acc_test)


if __name__ == "__main__":
    main()
