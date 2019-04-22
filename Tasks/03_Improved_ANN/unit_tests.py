''' Script for Unit Testing '''

import numpy as np
from copy import deepcopy
import unittest
from ann_utils import create, forward_propagation, loss_function, cost_function, backward_propagation, optimize

class TestAnnUtils(unittest.TestCase):
    ''' Test Class to make the methods for testing the original methods
    in the ann_utils module '''
    
    def initialise(self):
        ''' Method for initialising the reusable entities '''
        # age = [30, 30, 77, 78, 83, 65, 45]
        normalize_age = [-1.2303, -1.2303, 0.8140, 0.8575, 1.0750, 0.2920, -0.5779]
        # percentage = [60, 40, 40, 60, 50, 25, 40]
        normalize_percentage = [1.1921, -0.3974, -0.3974, 1.1921, 0.3974, -1.5894, -0.3974]
        self.X = np.array([normalize_age, normalize_percentage])
        self.Y = np.array([[1, 1, 0, 0, 0, 1, 1]])
        self.layers_units = [2, 3, 1]
        self.layer_inputs = [2, 2, 3]
        self.sample_size = self.X.shape[1]
        self.layers = len(self.layer_inputs)
        self.learning_rate = 0.05
    
    def test_create(self):
        ''' test method for testing create function '''
        self.initialise()
        self.W, self.B, self.Z, self.A, self.dWs, self.dbs = create(self.X, self.layers_units)
        
        for w, b, z, a, dw, db, units, inps in zip(self.W, self.B, self.Z, self.A, self.dWs, self.dbs, self.layers_units, self.layer_inputs):
            self.assertEqual(w.shape, (inps, units))
            self.assertEqual(b.shape, (units, 1))
            self.assertEqual(z.shape, (units, self.sample_size))
            self.assertEqual(a.shape, (units, self.sample_size))
            self.assertEqual(dw.shape, (inps, units))
            self.assertEqual(db.shape, (units, 1))
        
    def test_forward_propagation(self):
        ''' test method for testing forward propagation function'''
        self.initialise()
        self.test_create()
        self.Z, self.A = forward_propagation(self.X, self.W, self.B, self.Z, self.A, self.layers)   
        z1 = np.dot(self.W[0].T, self.X) + self.B[0]
        a1 = np.tanh(z1)
        z2 = np.dot(self.W[1].T, a1) + self.B[1]
        a2 = np.tanh(z2)
        z3 = np.dot(self.W[2].T, a2) + self.B[2]
        a3 = 1 / (1 + np.exp(-z3))   
        Z = [self.Z[0].flatten(), self.Z[1].flatten(), self.Z[2].flatten()]
        A = [self.A[0].flatten(), self.A[1].flatten(), self.A[2].flatten()]
        checkZ = [z1.flatten(), z2.flatten(), z3.flatten()]
        checkA = [a1.flatten(), a2.flatten(), a3.flatten()]   
        
        for valZ, valCheckZ, valA, valCheckA in zip(Z, checkZ, A, checkA):
            for v11, v21, v12, v22 in zip(valZ, valCheckZ, valA, valCheckA):
                self.assertEqual(v11, v21)
                self.assertEqual(v12, v22)
    
    def test_loss_function(self):
        ''' test method for testing loss function '''
        self.initialise()
        self.test_create()
        self.test_forward_propagation()
        L = loss_function(self.A[-1], self.Y).flatten()
        checkL = (-(self.Y * np.log(self.A[-1]) + (1 - self.Y) * (np.log(1 - self.A[-1])))).flatten()
        
        for v1, v2 in zip(L, checkL):
            self.assertEqual(v1, v2)
    
    def test_cost_function(self):
        ''' test method for testing cost function '''
        self.initialise()
        self.test_create()
        self.test_forward_propagation()
        C = cost_function(self.A[-1], self.Y)
        checkC = (1 / self.sample_size) * np.sum(-(self.Y * np.log(self.A[-1]) + (1 - self.Y) * (np.log(1 - self.A[-1]))))
        self.assertEqual(C, checkC)
        
    def test_backward_propagation(self):
        ''' test method for testing backward propation function '''
        self.initialise()
        self.test_create()
        self.test_forward_propagation()
        self.dWs, self.dbs = backward_propagation(self.X, self.Y, self.W, self.Z, self.A, self.dWs, self.dbs, self.layers)
        dz3 = self.A[-1] - self.Y
        dw3 = (1 / self.sample_size) * (np.dot(dz3, self.A[1].T).T)
        db3 = (1 / self.sample_size) * np.sum(dz3, axis=1, keepdims=True)
        dz2 = np.dot(self.W[2], dz3) * (1 - np.tanh(self.Z[1])**2)
        dw2 = (1 / self.sample_size) * (np.dot(dz2, self.A[0].T).T)
        db2 = (1 / self.sample_size) * np.sum(dz2, axis=1, keepdims=True)
        dz1 = np.dot(self.W[1], dz2) * (1 - np.tanh(self.Z[0])**2)
        dw1 = (1 / self.sample_size) * (np.dot(dz1, self.X.T).T)
        db1 = (1 / self.sample_size) * np.sum(dz1, axis=1, keepdims=True)
        check_dWs = [dw1.flatten(), dw2.flatten(), dw3.flatten()]
        check_dbs = [db1.flatten(), db2.flatten(), db3.flatten()]
        dWs = [self.dWs[0].flatten(), self.dWs[1].flatten(), self.dWs[2].flatten()]
        dbs = [self.dbs[0].flatten(), self.dbs[1].flatten(), self.dbs[2].flatten()]
        
        for val_dW, val_Check_dW, val_db, val_Check_db in zip(dWs, check_dWs, dbs, check_dbs):
            for v11, v21, v12, v22 in zip(val_dW, val_Check_dW, val_db, val_Check_db):
                self.assertEqual(v11, v21)
                self.assertEqual(v12, v22)
        return [dw1, dw2, dw3], [db1, db2, db3]

    def test_optimize(self):
        ''' test method for testing optimize function '''
        self.initialise()
        self.test_create()
        self.test_forward_propagation()
        check_dWs, check_dbs = self.test_backward_propagation()
        check_W = deepcopy(self.W)
        check_B = deepcopy(self.B)
        self.W, self.B = optimize(self.W, self.B, self.learning_rate, self.dWs, self.dbs, self.layers)
       
        for num in range(self.layers):
            check_W[num] = check_W[num] - self.learning_rate * check_dWs[num]
            check_B[num] = check_B[num] - self.learning_rate * check_dbs[num]
        
        for val_W, val_Check_W, val_b, val_Check_b in zip(self.W, check_W, self.B, check_B):
            for v11, v21, v12, v22 in zip(val_W, val_Check_W, val_b, val_Check_b):
                for We, CWe, Be, CBe in zip(v11, v21, v12, v22):
                    self.assertEqual(We, CWe)
                    self.assertEqual(Be, CBe)

if __name__ == '__main__':
    unittest.main()