''' Artificial Neural Network (Binary Classifier) '''

import numpy as np
import nn_utils

class SequentialANN:
    def __init__(self, layers=None):
        self.layers = layers or []
        self.size = 0
        self.input_shape = None
        self._costs = []
        self._train_acc = 0.0

    def add(self, layer):
        layer.validate_fields(self.input_shape)
        self.input_shape = layer.units
        self.layers.append(layer)
        self.size += 1

    def compile_(self, optimizer, batch_norm=False):
        self.optimizer_fn = optimizer 
        self.layers_units = []; self.act_fns = []; self.Ws = []; self.Bs = []; self.Zs = []; self.As = []; self.dWs = []; self.dBs = []
        for l in self.layers:
            self.layers_units.append(l.units)
            self.act_fns.append(l.activation)
            self.Ws.append(l.W)
            self.Bs.append(l.B)
            self.dWs.append(l.dW)
            self.dBs.append(l.dB)        
        self.loss_fn = 'categorical_crossentropy' if self.layers_units[-1] > 1 else 'binary_crossentropy'
    
    def fit(self, features, targets, batch_size, epochs, learning_rate, lambda_=0, lr_decay=0, **kwargs):
        self.X = features
        for l in self.layers:
            l.Z = np.broadcast_to(l.Z, (l.Z.shape[0], self.X.shape[1]))
            self.Zs.append(l.Z)
            l.A = np.broadcast_to(l.A, (l.A.shape[0], self.X.shape[1]))
            self.As.append(l.A)
        self.Y = targets
        self.bs = batch_size
        self.epochs = epochs
        self.alpha = learning_rate
        self.l = lambda_
        self.lrd = lr_decay
        self._costs, self._train_acc = nn_utils.fit(self.optimizer_fn)(self.X, self.Y, self.Ws, self.Bs, self.Zs, self.act_fns, self.As, 
                                                  self.alpha, self.bs, self.dWs, self.dBs, len(self.layers_units), self.epochs, self.l,
                                                  self.lrd, **kwargs)   
    
    def predict(self, features):
        ''' Method to predict the output for the given test set features '''
        _, As = nn_utils.forward_propagation(features, self.Ws, self.Bs, self.Zs, self.act_fns, self.As, len(self.layers_units))
        self.Y_hat = As[-1]
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


class Layer:
    def __init__(self, units, activation, **kwargs):
        self.units = units
        self.activation = activation
        self.input_shape = kwargs.get('input_shape', None)
        self.W = self.B = self.Z = self.A = self.dW = self.dB = None

    def validate_fields(self, input_shape):
        self.input_shape = input_shape or self.input_shape 
        if self.input_shape == None:
            raise AttributeError('input_shape should be provided for the first layer (i.e., Input layer)')
        self.W = np.random.randn(self.input_shape, self.units) * np.sqrt(2 / self.input_shape)
        self.B = np.random.randn(self.units, 1)
        self.Z = np.zeros((self.units, 1))
        self.A = np.zeros_like(self.Z)
        self.dW = np.zeros_like(self.W)
        self.dB = np.zeros_like(self.B)
        self.W.setflags()