''' Utility funtions for Perceptron '''

import numpy as np

def validate(features, targets, alpha, epochs):
    ''' Function to validate the parameters '''
    assert features.ndim == 2, ('Features entered should be a numpy array of rank 2')
    assert targets.ndim == 2 and targets.shape[0] == 1, ('Targets should be a numpy array of rank 2 and dimensions 1 * m \n(m: samples)')
    assert alpha > 0 and alpha <= 1, ('Learning rate should be between 0 and 1')
    assert epochs > 0, ('Epochs should be greater than 1')

def activate_sigmoid(Z):
    ''' Function to return the activated value matrix '''
    return 1 / (1 + np.exp(-Z))

def forward_propagation(W, b, X, Y):
    ''' Function for Forward Propagation '''
    Z = np.dot(W.T, X) + b
    A = activate_sigmoid(Z)
    C = cost_fucntion(A, Y, X.shape[1])
    return A, C

def loss_function(A, Y):
    ''' Function to return the residual (loss or error) '''
    return Y * np.log(A) + (1 - Y) * (np.log(1 - A))

def cost_fucntion(A, Y, m):
    ''' Function to return the cost '''
    return (- 1 / m) * np.sum(loss_function(A, Y))

def backward_propagation(X, Y, A, m):
    ''' Function to backpropagate using gradient descent '''
    dZ = A - Y
    dW = np.dot(X, dZ.T) / m
    db = np.sum(dZ, axis=1) / m
    return dW, db
   
def optimize(W, b, alpha, dW, db):
    ''' Function to optimize (or update) the weights '''
    return W - alpha * dW, b - alpha * db