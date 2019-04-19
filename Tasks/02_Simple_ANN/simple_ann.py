''' Artificial Neural Network (Binary Classifier) '''

import numpy as np
import simple_ann_utils

class SimpleANN:
    def __init__(self, layers_units, learning_rate, epochs, show_cost=False):
        self.layers_units = layers_units
        self.alpha = learning_rate
        self._epochs = epochs
        self.show_cost = show_cost
        self._train_acc = 0.0
        self._costs = []

    def fit(self, features, targets):
        simple_ann_utils.validate(features, targets, self.alpha, self._epochs)
        self.X = features
        self.Y = targets
        self.W, self.b, self.Z, self.A, self.dWs, self.dbs = simple_ann_utils.create(self.X, self.layers_units)
     
        for e in range(self._epochs):
            self.W, self.b, self.Z, self.A, self.dWs, self.dbs, C = simple_ann_utils.propagate(self.X, self.Y, self.W, self.b,
                                                                                        self.Z, self.A, self.alpha,
                                                                                        self.dWs, self.dbs, len(self.layers_units))
            if self.show_cost:
                print(f'Cost for Epoch-{e+1}: {C}')
            self._train_acc += (100 - np.mean(np.abs(self.A[-1] - self.Y)) * 100) / self._epochs 
            self._costs.append(C)
    
    def predict(self, features):
        ''' Method to predict the output for the given test set features '''
        _, A = simple_ann_utils.forward_propagation(features, self.W, self.b, self.Z, self.A, len(self.layers_units))
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
    