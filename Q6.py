import numpy as np
#import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.special import softmax
#from sklearn.linear_model import LogisticRegression
from math import log
import re
import operator
from Q4 import generate_word_collection

# x: feature vector
# theta:

# probabilities
# probability distributions of a list of potential outcomes
# theta: parameter matrix
# x: feature vector
# I think this might be the score function of each class
# SOFTMAX: GIVES YOU THE PREDICTION FOR EACH CLASS

def inference(theta, x):
    theta_t = theta.transpose()
    theta_x = np.dot(theta_t, x)
    soft = softmax(theta_x)
    return np.log(soft)

# gradient of cross-entropy loss
# "loss function"
# the loss function creates the errors
def gradient(theta, x, y):
    y_hat = np.exp(inference(theta, x))
    return y_hat - y

def full_gradient(theta, x, y, learning_rate, iterations):
    m = len(y)
    for i in range(iterations):
        predict = np.dot(x, theta) #our predictions
        # update theta using gradient descent formula
        theta = theta - (1/m) * learning_rate * (x.T.dot(predict-y))
    return theta


def stochastic_gradient(x, y, theta, learning_rate, iterations):
    m = len(y)
    for i in range(iterations):
        for value in range(m):
            rand = np.random.randint(0, m)
            x_value = x[rand, :].reshape(1, x.shape[1])
            y_value = y[rand].reshape(1, 1)
            predict = np.dot(x_value, theta) #predictions
            # update theta using stochastic gradient descent formula
            theta = theta - (1/m) * learning_rate * (x_value.T.dot((predict - y_value)))
    return theta






