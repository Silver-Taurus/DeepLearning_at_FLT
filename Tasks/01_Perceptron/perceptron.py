''' Perceptron: based on Logistic Regression (Binary Classifier) '''

import numpy as np
import perceptron_utils

class Perceptron:
    def __init__(self, learning_rate, epochs, print_cost=False):
        self.alpha = learning_rate
        self._epochs = epochs
        self.print_cost = print_cost
        self._train_acc = 0.0
        self._costs = []

    def fit(self, features, targets):
        self.X = features
        self.Y = targets
        n = self.X.shape[0]
        m = self.X.shape[1]
        
        self.W = np.zeros((n, 1))
        self.b = 0
        
        perceptron_utils.validate(self.X, self.Y, self.alpha, self.epochs)
     
        for _ in range(self._epochs):
            Y_hat, C = perceptron_utils.forward_propagation(self.W, self.b, self.X, self.Y)
            dW, db = perceptron_utils.backward_propagation(self.X, self.Y, Y_hat, m)            
            self.W, self.b = perceptron_utils.optimize(self.W, self.b, self.alpha, dW, db)

            if self.print_cost:
                print(f'Cost: {C}')
                
            self._train_acc += (100 - np.mean(np.abs(Y_hat - self.Y)) * 100) / self._epochs 
            self._costs.append(C)

    def predict(self, features):
        ''' Method to predict the output for the given test set features '''
        self.Y_hat = np.zeros((1, features.shape[1]))
        self.Y_hat = perceptron_utils.activate_sigmoid(np.dot(self.W.T, features) + self.b)
        for i in range(self.Y_hat.shape[1]):
            self.Y_hat[0, i] = 1 if self.Y_hat[0, i] > 0.5 else 0
        return self.Y_hat.astype(int)

    @property
    def epochs(self):
        ''' Method to get the epochs '''
        return self._epochs
    
    @property
    def costs(self):
        ''' Method to get the cost data for all all epochs '''
        return self._costs

    @property
    def train_acc(self):
        ''' Method to get the training accuracy '''
        return self._train_acc 
    