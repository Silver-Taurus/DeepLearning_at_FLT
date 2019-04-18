''' LSTMs '''

import numpy as np

class RNN:
    def __init__(self, xs, ys, rl, eo, lr):
        # initial input (first word)
        self.x = np.zeros(xs)
        # input size
        self.x_size = xs
        # expected output
        self.y = np.zeros(ys)
        # output size
        self.y_size = ys
        # weight matrix for interpreting results from LSTM cell
        self.weights = np.random.random((ys, ys))
        # matrix used in RMSprop (technique of gradient descent that decays the learning rate)
        self.gradient = np.zeros_like(self.weights)
        # length of the recurrent network - number of recurrences, i.e., number of words we have
        self.recur_length = rl
        # learning rate
        self.learning_rate = lr
        # array for storing inputs
        self.input_arr = np.zeros((rl+1, xs))
        # array for storing cell states
        self.cell_arr = np.zeros((rl+1, ys))
        # array for storing outputs
        self.output_arr = np.zeros((rl+1, ys))
        # array for storing hidden states
        self.hidden_arr = np.zeros((rl+1, ys))
        # forget gate
        self.forget_gate_arr = np.zeros((rl+1, ys))
        # input gate
        self.input_gate_arr = np.zeros((rl+1, ys))
        # cell state
        self.cur_cell_arr = np.zeros((rl+1, ys))
        # output gate
        self.output_gate_arr = np.zeros((rl+1, ys))
        # array of expected output values
        self.expected_output = np.vstack((np.zeros(eo.shape[0]), eo.T))
        # declare LSTM cell (input, output, amount of recurrence, learning rate)
        self.LSTM = LSTM(xs, ys, rl, lr)
        
    # activation function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    # the derivative of the sigmoid function, used to compute the gradient
    def dsigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    # apply a series of matrix operation to our input
    def forward_propagate(self):
        for i in range(1, self.rl+1):
            self.LSTM.x = np.hstack((self.hidden_arr[i-1], self.x))
            cs, hs, f, inp, c, o = self.LSTM.forward_propagate()
            # store computed cell state
            self.cell_arr[i] = cs
            self.hidden_arr[i] = hs
            self.forget_gate_arr[i] = f
            self.cur_cell_arr[i] = c
            self.output_gate_arr[i] = o
            self.output_arr[i] = self.sigmoid(np.dot(self.weights, hs))
            self.input_gate_arr[i] = inp
            self.x = self.expected_output[i-1]
        return self.output_arr

    def back_propagate(self):
        # update our weight matrices (Both in RNN and LSTM)    
        # initialise an empty error 
        total_err = 0
        # initialise matrices for gradient updates
        # First for RNN level gradient
        # cell state
        diff_current_state = np.zeros(self.y_size)
        # hidden state
        diff_hidden_state = np.zeros(self.y_size)
        # weight matirx
        total_update = np.zeros((self.y_size, self.y_size))
        # Next for LSTM level gradient
        # forget gate
        total_forget_update = np.zeros((self.y_size, self.x_size + self.y_size))
        # input gate
        total_input_update = np.zeros((self.y_size, self.x_size + self.y_size))            
        # cell unit
        total_cell_update = np.zeros((self.y_size, self.y_size + self.x_size))
        # output gate
        total_output_update = np.zeros((self.y_size, self.x_size + self.y_size))
        #loop backwards thorugh recurrences
        for i in range(self.recur_length, -1, -1):
            # error = calcOutput - expOutput
            error = self.output_arr[i] + self.expected_output[i]
            # calculate and update for weight matrix (error * derivative of the output * hidden state)
            total_update += np.dot(np.atleast_2d(error * self.dsigmoid(self.output_arr[i])), np.atleast_2d(self.hidden_arr[i]).T)
            # Propagate error back to exit of LSTM cell
            # 1. error * RNN wweight matrix
            error = np.dot(error, self.weights)
            # 2. set input values of LSTM cell for recurrence i (horizontal stack of array)
            self.LTSM.x = np.hstack((self.hidden_arr[i-1], self.input_arr[i]))
            # 3. set cell state of LSTM cell's for recurrence i (pre-updates)
            self.LTSM.cell_state = self.cell_arr[i]
            # Finally call the LSTM cell's back_propagate, retreive gradient updataes
            fu, iu, cu, ou, diff_current_state, diff_hidden_state = self.LSTM.back_propagate(error, self.cell_arr[i-1],
                                                                                             self.forget_gate_arr[i], 
                                                                                             self.input_gate_arr[i],
                                                                                             self.cur_cell_arr[i],
                                                                                             self.output_gate_arr[i], 
                                                                                             diff_current_state,
                                                                                             diff_hidden_state)
            # calculate the total error
            total_err += np.sum(error)
            # accumulate all gradient updates
            # forget gate
            total_forget_update += fu
            # input gate
            total_input_update += iu
            # cell state
            total_cell_update += cu
            # output gate
            total_output_update += ou
        # update the LSTM matrices with the average of accumulated gradient updates
        self.LSTM.update(total_forget_update / self.recur_length, total_input_update / self.recur_length,
                         total_cell_update / self.recur_length, total_output_update / self.recur_length)
        # update the weight matrix with average of accumulated gradient update
        self.update(total_update / self.recur_length)
        # return total error of this iteration
        return total_err
        
    def update(self, update):
        # implementation of RMSprop
        self.gradient = (0.9 * self.gradient) + (0.1 * update ** 2)
        self.weights -= self.recur_length / np.sqrt(self.gradient + 1e-8) * update
        return
    
    def sample(self):
        for i in range(1, self.rl+1):
            # set input for LSTM cell
            self.LSTM.x = np.hstack((self.hidden_arr[i-1], self.x))
            # run forward propagation on LSTM cell, retreive cell state and hidden state
            cs, hs, f, inp, c, o = self.LSTM.forward_propagate()
            # store input as vector
            maxI = np.argmax()
            self.x = np.zeros_like(self.x)
            self.x[maxI] = 1
            self.input_arr[i] = self.x
            # store cell state
            self.cell_arr[i] = cs
            # store hidden statea
            self.hidden_arr[i] = hs
            # forget gate
            self.forget_gate_arr[i] = f
            # cell state
            self.cur_cell_arr[i] = c
            # output gate
            self.output_gate_arr[i] = o
            # input gate
            self.input_gate_arr[i] = inp
            # calculate output by multiplying hidden state with weight matrix
            self.output_arr[i] = self.sigmoid(np.dot(self.weights, hs))
            # compute new input
            maxI = np.argmax(self.output_arr[i])
            newX = np.zeros_like(self.x)
            newX[maxI] = 1
            self.x = newX
        return self.output_arr
      
    class LSTM:
        def __init__(self, xs, ys, rl, lr):
            # input is word length * word length
            self.x = np.zeros(xs + ys)
            # input size = word length + word length
            self.x_size = xs + ys
            # output
            self.y = np.zeros(ys)
            # output size
            self.y_size = ys
            # how often to perform recurrence
            self.recur_length = rl
            # learning rate
            self.learning_rate = lr
            # init weight matrices for our gates
            # forget gate
            self.f = np.random.random((ys, xs + ys))
            # input gate
            self.i = np.random.random((ys, xs + ys))
            # cell state
            self.c = np.random.random((ys, xs + ys))
            # output state
            self.o = np.random.random((ys, xs + ys))
            # forget gate gradient
            self.f_gradient = np.zeros_like(self.f)
            # input gate gradient
            self.i_gradient = np.zeros_like(self.i)
            # cell state gradient
            self.c_gradient = np.zeros_like(self.c)
            # output gate gradient
            self.o_gradient = np.zeros_like(self.o)
        
        # activation function to activate our forward propagation
        def sigmoid(self, x):
            return 1 / (1 + np.exp(-x))

        # derivative of sigmoid to help compute gradients  
        def dsigmoid(self, x):
            return self.sigmoid(x) * (1 - self.sigmoid(x))
        
        # another activation function often used by LSTM cells
        def hyperbolic_tanget(self, x):
            return np.tanh(x)
        
        # derivative of tanh for computing gradients
        def dhyperbolic_tanget(self, x):
            return 1 - np.tanh(x) ** 2
        
        # computer a series of matrix mulitiplication to convert our input into output
        def forward_propagate(self):
            f = self.sigmoid(np.dot(self.f, self.x))
            self.cs *= f
            i = self.sigmoid(np.dot(self.i, self.x))
            c = self.hyperbolic_tanget(np.dot(self.c, self.x))
            self.cs += i * c
            o = self.sigmoid(np.do(self.o, self.x))
            self.y = o * self.hyperbolic_tanget(self.cs)
            return self.cs, self.y, f, i, c, o
        
        