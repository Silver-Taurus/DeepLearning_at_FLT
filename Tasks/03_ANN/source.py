''' Using a L-layer Neural Network (for Binary Classification) with 
Careful Weight Initialisation, Regularisation and Mini-Batch Gradient Descent'''

# Importing the libraries
import h5py
import numpy as np
import matplotlib.pyplot as plt
from ann import ANN

# Importing training dataset
train_file = h5py.File('train_catvnoncat.h5', 'r')
train_features = np.array(train_file['train_set_x'].value)
train_targets = np.array(train_file['train_set_y'].value)

# Importing testing dataset
test_file = h5py.File('test_catvnoncat.h5', 'r')
test_features = np.array(test_file['test_set_x'].value)
test_targets = np.array(test_file['test_set_y'].value)

# Flattening and Reshaping the features
train_features_flatten = train_features.reshape(train_features.shape[0], -1).T
test_features_flatten = test_features.reshape(test_features.shape[0], -1).T

# Normalize the features
from sklearn.preprocessing import normalize
train_set_x = normalize(train_features_flatten)
test_set_x = normalize(test_features_flatten)

# Reshaping the targets
train_set_y = train_targets.reshape((1, train_targets.shape[0]))
test_set_y = test_targets.reshape((1, test_targets.shape[0]))

# Initialising and Fitting the ANN
ann = ANN(layers_units=[3, 2, 1], learning_rate=0.05, lambda_=0, momentum=0.98, batch_size=16, epochs=300, show_cost=True)
ann.fit(train_set_x, train_set_y)

# Predicting the output on test_set_x
y_pred = ann.predict(test_set_x)
y_pred = y_pred.reshape((y_pred.shape[1],))

# Making the confuse matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_targets, y_pred)

# Giving the accuracy
print(f'train accuracy: {ann.train_acc} %')
print(f'test accuracy: {100 - np.mean(np.abs(y_pred - test_targets)) * 100} %')

# Get the cost of all epochs and visualise the training
costs = ann.costs
plt.plot(costs)
plt.title('Visualising the training (learning rate = 0.005)')
plt.xlabel('epochs')
plt.ylabel('costs')
plt.show()
