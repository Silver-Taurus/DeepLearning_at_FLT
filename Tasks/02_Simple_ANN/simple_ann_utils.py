''' Utility funtions for Artificial Neural Network '''

import numpy as np

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

def validate(features, targets, alpha, epochs):
    ''' Function to validate the parameters '''
    assert features.ndim == 2, ('Features entered should be a numpy array of rank 2')
    assert targets.ndim == 2 and targets.shape[0] == 1, ('Targets should be a numpy array of rank 2 and dimensions 1 * m \n(m: samples)')
    assert alpha > 0 and alpha <= 1, ('Learning rate should be between 0 and 1')
    assert epochs > 0, ('Epochs should be greater than 1')

def create(X, layers_units):
    ''' Function to create the layer inputs, initial weights and bias matrix '''
    layer_inputs = [X.shape[0]] + layers_units[:-1]
    layer_samples = X.shape[1]
    W = []; b = []; Z = []; A = []; dWs = []; dbs = []
    for inputs, units in zip(layer_inputs, layers_units):
        W.append(np.random.rand(inputs, units)*0.001)
        b.append(np.random.rand(units, 1))
        Z.append(np.random.rand(units, layer_samples))
        A.append(np.random.rand(units, layer_samples))
        dWs.append(np.random.rand(inputs, units))
        dbs.append(np.random.rand(units, 1))
    return W, b, Z, A, dWs, dbs
    
def propagate(X, Y, W, b, Z, A, alpha, dWs, dbs, layers):
    ''' Function to perform both forward pass and backward pass '''
    Z, A = forward_propagation(X, W, b, Z, A, layers)
    C = cost_function(A[-1], Y)
    dWs, dbs = backward_propagation(X, Y, W, Z, A, dWs, dbs, layers)
    W, b = optimize(W, b, alpha, dWs, dbs, layers)
    return W, b, Z, A, dWs, dbs, C

def forward_propagation(X, W, b, Z, A, layers):
    ''' Function for Forward Propagation '''
    A_cache = [X]
    for num in range(layers):
        activation = 'sigmoid' if num == layers - 1 else 'tanh'
        Z[num], A[num] = process_layer(W[num], b[num], A_cache.pop(), activation)
        A_cache.append(A[num])
    return Z, A

def process_layer(W, b, A_cache, activation):
    ''' Function to process a layer of NN '''
    z = np.dot(W.T, A_cache) + b
    a =  activation_function(activation)(z)
    return z, a  

def loss_function(Y_hat, Y):
    ''' Function to return the residual (loss or error) '''
    return -(Y * np.log(Y_hat) + (1 - Y) * (np.log(1 - Y_hat)))

def cost_function(Y_hat, Y):
    ''' Function to return the cost '''
    return (1 / Y.shape[1]) * np.sum(loss_function(Y_hat, Y))

def backward_propagation(X, Y, W, Z, A, dWs, dbs, layers):
    ''' Function to backpropagate using gradient descent '''
    dZ_cache = [A[-1] - Y]
    for num in range(layers - 1, -1, -1):
        dZ = dZ_cache.pop()
        a = X if num == 0 else A[num-1]
        dWs[num] = (1 / a.shape[1]) * (np.dot(dZ, a.T).T)
        dbs[num] = (1 / a.shape[1]) * np.sum(dZ, axis=1, keepdims=True)
        if num - 1 >= 0:
            dZ_cache.append(np.dot(W[num], dZ) * dtanh(Z[num-1]))
    return dWs, dbs
 
def optimize(W, b, alpha, dWs, dbs, layers):
    ''' Function to optimize (or update) the weights and bias '''
    for num in range(layers):
        W[num] = W[num] - alpha * dWs[num]
        b[num] = b[num] - alpha * dbs[num]
    return W, b

@function_dispatcher
def activation_function(fn):
    return AttributeError('No such function Exists!!!')

@activation_function.register('tanh')
def activate_tanh(Z):
    ''' Function to return the activated values '''
    return np.tanh(Z)

@activation_function.register('sigmoid')
def activate_sigmoid(Z):
    ''' Function to return the activated value matrix '''
    return 1 / (1 + np.exp(-Z))
 
def dtanh(Z):
    ''' Function to return gradient descent value of the matrix '''
    return 1 - activate_tanh(Z)**2

def dsigmoid(Z):
    return np.exp(-Z) * (activate_sigmoid(Z) ** 2)
