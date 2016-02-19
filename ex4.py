import scipy.io
import scipy.misc
import scipy.optimize
import scipy.special
from numpy import *

def sigmoid(z):
    return scipy.special.expit(z)

def loadmat(file_path, *names):
    mat = scipy.io.loadmat(file_path)
    return [mat[name] for name in names]

def neural_network():
    X, y = loadmat('ex3data1.mat', 'X', 'y')
    theta1, theta2 = loadmat('ex3weights.mat', 'Theta1', 'Theta2')
    m, n = X.shape
    X = c_[ones((m, 1)), X]
    A = c_[ones((m, 1)), sigmoid(theta1.dot(X.T)).T]
    out = theta2.dot(A.T).T
    correct = 0
    for i in range(0, m):
        prediction = argmax(out[i]) + 1
        correct += prediction == y[i]
    print('Accuracy: %.2f%%' % (correct * 100.0 / m))


def main():
    print('Neural Network')
    neural_network()

if __name__ == '__main__':
    main()
