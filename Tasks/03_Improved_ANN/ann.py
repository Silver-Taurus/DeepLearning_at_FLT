''' Artificial Neural Network (Binary Classifier) '''

import numpy as np
import ann_utils
import math

class ANN:
    def __init__(self, layers_units, learning_rate, lambda_, momentum, batch_size, epochs, show_cost=False):
        self.layers_units = layers_units
        self.alpha = learning_rate
        self.l = lambda_
        self.beta = momentum
        self.bs = batch_size
        self._epochs = epochs
        self.show_cost = show_cost
        self._train_acc = 0.0
        self._costs = []

    def fit(self, features, targets):
        ann_utils.validate(features, targets, self.alpha, self._epochs)
        self.X = features
        self.Y = targets
        self.W, self.b, self.Z, self.A, self.dWs, self.dbs, self.V_dWs, self.V_dbs = ann_utils.create(self.X, self.layers_units)
     
        for e in range(self._epochs):
            self.W, self.b, self.Z, self.A, self.dWs, self.dbs, self.V_dWs, self.V_dbs, C = ann_utils.propagate(self.X, self.Y, self.W, 
                                                                                                                self.b, self.Z, self.A, 
                                                                                                                self.alpha, self.l, 
                                                                                                                self.beta, self.bs, 
                                                                                                                math.ceil(self.X.shape[1] / self.bs),
                                                                                                                self.dWs, self.dbs, 
                                                                                                                self.V_dWs, self.V_dbs,
                                                                                                                len(self.layers_units))
            if self.show_cost:
                print(f'Cost for Epoch-{e+1}: {C}')
            self._train_acc += (100 - np.mean(np.abs(self.A[-1] - self.Y)) * 100) / self._epochs 
            self._costs.append(C)
    
    def predict(self, features):
        ''' Method to predict the output for the given test set features '''
        _, A = ann_utils.forward_propagation(features, self.W, self.b, self.Z, self.A, len(self.layers_units))
        self.Y_hat = A[-1]
        for i in range(self.Y_hat.shape[1]):
            self.Y_hat[0, i] = 1 if self.Y_hat[0, i] > 0.5 else 0
        return self.Y_hat.astype(int)
    
    @property
    def costs(self):
        ''' Method to get the cost data for all all epochs '''
        return self._costs

    @property
    def train_acc(self):
        ''' Method to get the training accuracy '''
        return self._train_acc 
    