''' Utility funtions for Artificial Neural Network '''

import numpy as np
from functools import partial
import math

def function_dispatcher(default_fn):
    ''' Decorator for dispatching the function from the registry '''
    registry = {}
    registry['Default'] = default_fn
    
    def decorated_function(fn):
        '''  function is decorated to give the desired function back 
        but with an additional property '''
        return registry.get(fn, registry['Default'])
    
    def register(act_fn_name):
        ''' Decorator factory (or Paramterized Decorator) that will be a
        property of our decorated function, which when called return the
        decorator '''
        def register_decorator(act_fn):
            ''' decorator to register the function in the registry and
            return the function back as it is '''
            registry[act_fn_name] = act_fn
            return act_fn
        return register_decorator
    
    decorated_function.register = register
    return decorated_function

@function_dispatcher
def fit(fn):
    return AttributeError('No Such optimzier fit Exists!!!')

@function_dispatcher
def activation_function(fn):
    return AttributeError('No such function Exists!!!')

@function_dispatcher
def dactivation_function(fn):
    return AttributeError('Passed function name not found!!!')

@fit.register('adam')
def adam_fit(X, Y, Ws, Bs, Zs, act_fns, As, alpha, bs, dWs, dBs, num_layers, epochs, l, lrd, **kwargs):
    iterations = math.ceil(X.shape[1] / bs)       
    costs = []
    train_acc = 0.0
   
    beta1 = kwargs['beta1']
    beta2 = kwargs['beta2']
    V_dWs = np.zeros_like(dWs)
    V_dBs = np.zeros_like(dBs)
    S_dWs = np.zeros_like(Ws)
    S_dBs = np.zeros_like(Bs)
    
    for e in range(epochs):
        Ws, Bs, Zs, As, dWs, dBs, V_dWs, V_dBs, S_dWs, S_dBs, C = adam_propagate(X, Y, Ws, Bs, Zs, act_fns, As, alpha, bs, iterations, dWs, 
                                                                                 dBs, num_layers, l, beta1, beta2, V_dWs, V_dBs, S_dWs, S_dBs)   
        print(f'Cost for Epoch-{e+1}: {C}')
        costs.append(C)
        train_acc += (100 - np.mean(np.abs(As[-1] - Y)) * 100) / epochs 
        alpha = alpha / (1 + lrd*e)
    return costs, train_acc

@fit.register('rmsprop')
def rmsprop_fit(X, Y, Ws, Bs, Zs, act_fns, As, alpha, bs, dWs, dBs, num_layers, epochs, l, lrd, **kwargs):
    iterations = math.ceil(X.shape[1] / bs)       
    costs = []
    train_acc = 0.0
        
    beta = kwargs['beta']    
    S_dWs = np.zeros_like(Ws)
    S_dBs = np.zeros_like(Bs)
    
    for e in range(epochs):
        Ws, Bs, Zs, As, dWs, dBs, S_dWs, S_dBs, C = rms_propagate(X, Y, Ws, Bs, Zs, act_fns, As, alpha, bs, iterations, dWs, dBs, num_layers,
                                                                  l, beta, S_dWs, S_dBs)   
        print(f'Cost for Epoch-{e+1}: {C}')
        costs.append(C)
        train_acc += (100 - np.mean(np.abs(As[-1] - Y)) * 100) / epochs 
        alpha = alpha / (1 + lrd*e)
    return costs, train_acc
  
@fit.register('gdm')
def gdm_fit(X, Y, Ws, Bs, Zs, act_fns, As, alpha, bs, dWs, dBs, num_layers, epochs, l, lrd, **kwargs):
    iterations = math.ceil(X.shape[1] / bs)       
    costs = []
    train_acc = 0.0
    
    beta = kwargs['beta']
    V_dWs = np.zeros_like(Ws)
    V_dBs = np.zeros_like(Bs)
    
    for e in range(epochs):
        Ws, Bs, Zs, As, dWs, dBs, V_dWs, V_dBs, C = gdm_propagate(X, Y, Ws, Bs, Zs, act_fns, As, alpha, bs, iterations, dWs, dBs, num_layers,
                                                                  l, beta, V_dWs, V_dBs)   
        print(f'Cost for Epoch-{e+1}: {C}')
        costs.append(C)
        train_acc += (100 - np.mean(np.abs(As[-1] - Y)) * 100) / epochs 
        alpha = alpha / (1 + lrd*e)
    return costs, train_acc

@fit.register('gd')
def gd_fit(X, Y, Ws, Bs, Zs, act_fns, As, alpha, bs, dWs, dBs, num_layers, epochs, l, lrd):
    iterations = math.ceil(X.shape[1] / bs)       
    costs = []
    train_acc = 0.0
    for e in range(epochs):
        Ws, Bs, Zs, As, dWs, dBs, C = gd_propagate(X, Y, Ws, Bs, Zs, act_fns, As, alpha, bs, iterations, dWs, dBs, num_layers, l)   
        print(f'Cost for Epoch-{e+1}: {C}')
        costs.append(C)
        train_acc += (100 - np.mean(np.abs(As[-1] - Y)) * 100) / epochs
        alpha = alpha / (1 + lrd*e)
    return costs, train_acc

def adam_propagate(X, Y, Ws, Bs, Zs, act_fns, As, alpha, bs, iterations, dWs, dBs, num_layers, l, beta1, beta2, V_dWs, V_dBs, S_dWs, S_dBs):
    ''' Function to perform both forward pass and backward pass '''
    for i in range(iterations):
        Zs, As = batch_forward_propagation(X, Ws, Bs, Zs, act_fns, As, bs, i, num_layers)
        dWs, dBs, V_dWs, V_dBs, S_dWs, S_dBs = adam_batch_backward_propagation(X, Y, Ws, Zs, act_fns, As, bs, i, dWs, dBs, num_layers, l, 
                                                                              beta1, beta2, V_dWs, V_dBs, S_dWs, S_dBs)
        Ws, Bs = adam_optimize(Ws, Bs, alpha, V_dWs, V_dBs, num_layers, S_dWs, S_dBs)
    C = cost_function(As[-1], Y)
    return Ws, Bs, Zs, As, dWs, dBs, V_dWs, V_dBs, S_dWs, S_dBs, C
 
def rms_propagate(X, Y, Ws, Bs, Zs, act_fns, As, alpha, bs, iterations, dWs, dBs, num_layers, l, beta, V_dWs, V_dBs, pow_=2):
    ''' Function to perform both forward pass and backward pass '''
    for i in range(iterations):
        Zs, As = batch_forward_propagation(X, Ws, Bs, Zs, act_fns, As, bs, i, num_layers)
        dWs, dBs, V_dWs, V_dBs = rms_batch_backward_propagation(X, Y, Ws, Zs, act_fns, As, bs, i, dWs, dBs, num_layers, l, beta, V_dWs, 
                                                                V_dBs, pow_)
        if beta is not None and pow_ == 1:
            Ws, Bs = rms_optimize(Ws, Bs, alpha, V_dWs, V_dBs, num_layers)
        elif beta is not None and pow_ == 2:
            Ws, Bs = rms_optimize(Ws, Bs, alpha, dWs, dBs, num_layers, V_dWs, V_dBs)
        else:
            Ws, Bs = rms_optimize(Ws, Bs, alpha, dWs, dBs, num_layers)
    C = cost_function(As[-1], Y)
    if beta is not None:
        return Ws, Bs, Zs, As, dWs, dBs, V_dWs, V_dBs, C
    else:
        return Ws, Bs, Zs, As, dWs, dBs, C
    
def batch_forward_propagation(X, Ws, Bs, Zs, act_fns, As, bs, i, num_layers):
    ''' Function for forward propagating a batch '''
    start = i * bs
    end = (i + 1) * bs
    Z_batch = [z[:, start: end] for z in Zs]
    A_batch = [a[:, start: end] for a in As]
    Z_batch, A_batch = forward_propagation(X[:, start: end], Ws, Bs, Z_batch, act_fns, A_batch, num_layers)
    Zs = [np.concatenate((z[:, :start], zb, z[:, end:]), axis=1) for zb, z in zip(Z_batch, Zs)]
    As = [np.concatenate((a[:, :start], ab, a[:, end:]), axis=1) for ab, a in zip(A_batch, As)]
    return Zs, As

def forward_propagation(X, Ws, Bs, Zs, act_fns, As, num_layers):
    ''' Function for Forward Propagation '''
    A_cache = [X]
    for num in range(num_layers):
        Zs[num], As[num] = process_layer(Ws[num], Bs[num], A_cache.pop(), act_fns[num])
        A_cache.append(As[num])
    return Zs, As

def process_layer(W, b, A_cache, activation):
    ''' Function to process a layer of NN '''
    z = np.dot(W.T, A_cache) + b
    a =  activation_function(activation)(z)
    return z, a  

def adam_batch_backward_propagation(X, Y, Ws, Zs, act_fns, As, bs, i, dWs, dBs, num_layers, l, beta1, beta2, V_dWs, V_dBs, S_dWs, S_dBs):
    ''' Function for backward propagating a batch '''
    start = i * bs
    end = (i + 1) * bs
    Z_batch = [z[:, start: end] for z in Zs]
    A_batch = [a[:, start: end] for a in As]
    X_batch = X[:, start: end]
    Y_batch = Y[:, start: end]
    return adam_backward_propagation(X_batch, Y_batch, Ws, Z_batch, act_fns, A_batch, dWs, dBs, num_layers, l, beta1, beta1, V_dWs, V_dBs,
                                    S_dWs, S_dBs)

def rms_batch_backward_propagation(X, Y, Ws, Zs, act_fns, As, bs, i, dWs, dBs, num_layers, l, beta, V_dWs, V_dBs, pow_):
    ''' Function for backward propagating a batch '''
    start = i * bs
    end = (i + 1) * bs
    Z_batch = [z[:, start: end] for z in Zs]
    A_batch = [a[:, start: end] for a in As]
    X_batch = X[:, start: end]
    Y_batch = Y[:, start: end]
    return rms_backward_propagation(X_batch, Y_batch, Ws, Z_batch, act_fns, A_batch, dWs, dBs, num_layers, l, beta, V_dWs, V_dBs, pow_)

def adam_backward_propagation(X, Y, Ws, Zs, act_fns, As, dWs, dBs, num_layers, l, beta1, beta2, V_dWs, V_dBs, S_dWs, S_dBs):
    ''' Function to backpropagate using gradient descent '''
    dZ_cache = [As[-1] - Y]
    for num in range(num_layers - 1, -1, -1):
        dZ = dZ_cache.pop()
        a = X if num == 0 else As[num-1]
        dWs[num] = (1 / a.shape[1]) * (np.dot(dZ, a.T).T) + (l / X.shape[1]) * Ws[num]
        dBs[num] = (1 / a.shape[1]) * np.sum(dZ, axis=1, keepdims=True)
        if num - 1 >= 0:
            dZ_cache.append(np.dot(Ws[num], dZ) * dactivation_function(act_fns[num-1])(Zs[num-1]))
        V_dWs[num] = beta1 * V_dWs[num] + (1 - beta1) * dWs[num]
        V_dBs[num] = beta1 * V_dBs[num] + (1 - beta1) * dBs[num]
        S_dWs[num] = beta2 * S_dWs[num] + (1 - beta2) * dWs[num]**2
        S_dBs[num] = beta2 * S_dBs[num] + (1 - beta2) * dBs[num]**2
    return dWs, dBs, V_dWs, V_dBs, S_dWs, S_dBs

def rms_backward_propagation(X, Y, Ws, Zs, act_fns, As, dWs, dBs, num_layers, l, beta, V_dWs, V_dBs, pow_):
    ''' Function to backpropagate using gradient descent '''
    dZ_cache = [As[-1] - Y]
    for num in range(num_layers - 1, -1, -1):
        dZ = dZ_cache.pop()
        a = X if num == 0 else As[num-1]
        dWs[num] = (1 / a.shape[1]) * (np.dot(dZ, a.T).T) + (l / X.shape[1]) * Ws[num]
        dBs[num] = (1 / a.shape[1]) * np.sum(dZ, axis=1, keepdims=True)
        if num - 1 >= 0:
            dZ_cache.append(np.dot(Ws[num], dZ) * dactivation_function(act_fns[num-1])(Zs[num-1]))
        if beta is not None:
            V_dWs[num] = beta * V_dWs[num] + (1 - beta) * dWs[num]**pow_
            V_dBs[num] = beta * V_dBs[num] + (1 - beta) * dBs[num]**pow_
    return dWs, dBs, V_dWs, V_dBs
 
def adam_optimize(Ws, Bs, alpha, V_dWs, V_dBs, num_layers, S_dWs, S_dBs):
    ''' Function to optimize (or update) the weights and bias '''
    for num in range(num_layers):
        Ws[num] = Ws[num] - alpha * V_dWs[num] / (np.sqrt(S_dWs[num] + 10e-8))
        Bs[num] = Bs[num] - alpha * V_dBs[num] / (np.sqrt(S_dBs[num] + 10e-8))
    return Ws, Bs    

def rms_optimize(Ws, Bs, alpha, V_dWs, V_dBs, num_layers, S_dWs=None, S_dBs=None):
    ''' Function to optimize (or update) the weights and bias '''
    if S_dWs == None or S_dBs == None:
        for num in range(num_layers):
            Ws[num] = Ws[num] - alpha * V_dWs[num]
            Bs[num] = Bs[num] - alpha * V_dBs[num]
    else:
        for num in range(num_layers):
            Ws[num] = Ws[num] - alpha * V_dWs[num] / (np.sqrt(S_dWs[num] + 10e-8))
            Bs[num] = Bs[num] - alpha * V_dBs[num] / (np.sqrt(S_dBs[num] + 10e-8))
    return Ws, Bs

gdm_propagate = partial(rms_propagate, pow_=1)
gd_propagate = partial(rms_propagate, beta=None, V_dWs=None, V_dBs=None, pow_=None)

def loss_function(Y_hat, Y):
    ''' Function to return the residual (loss or error) '''
    return -(Y * np.log(Y_hat) + (1 - Y) * (np.log(1 - Y_hat)))

def cost_function(Y_hat, Y):
    ''' Function to return the cost '''
    return (1 / Y.shape[1]) * np.sum(loss_function(Y_hat, Y))

@activation_function.register('tanh')
def activate_tanh(Z):
    ''' Function to return the activated values '''
    return np.tanh(Z)

@activation_function.register('sigmoid')
def activate_sigmoid(Z):
    ''' Function to return the activated value matrix '''
    return 1 / (1 + np.exp(-Z))
 
@dactivation_function.register('tanh')
def dtanh(Z):
    ''' Function to return gradient descent value of the matrix '''
    return 1 - activate_tanh(Z)**2

@dactivation_function.register('sigmoid')
def dsigmoid(Z):
    return np.exp(-Z) * (activate_sigmoid(Z) ** 2)
