''' Self-Organizing Maps (SOMs) '''

# Training the SOM
#   Step1: We start with a dataset composed of n-features independent variables.
#   Step2: We create a grid composed of nodes, each one having a weight vector of n-features elements.
#   Step3: Randomly initialise the values of the weight vectors to small nnumbers close to 0 (but not 0).
#   Step4: Select one random observation point from the dataset.
#   Step5: Compute the eucledian distances from this point to the different neurons in the network.
#   Step6: Select the neuron that has the minimum distance to the point. This neuron is the winning node.
#   Step7: Update the weights of the winning node to move it closer to the point.
#   Step8: Using the Gaussian neighbourhood function of mean the winning node, also update the weights of the winning node
#          neighbours to move them closer to the point. The neighbourhood radius is the sigma in the Gaussian function.
#   Step9: Repeat Steps1 to 5 and update the weights after each observation (Reinforcement Learning) or after a batch of
#          observations (Batch Learning), until the networkconverges to a point where the neighbourhood stops decreasing.

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')

# Making the feature matrix and target vector
features = dataset.iloc[:, :-1].values
target = dataset.iloc[: , -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
features = sc.fit_transform(features)

# Training the SOM
from minisom import MiniSom
som = MiniSom(x=10, y=10, input_len=features.shape[1], learning_rate=0.5)
som.random_weights_init(features)
som.train_random(features, num_iteration=100)

# Visualising the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(features):
    w = som.winner(x)
    plot(w[0] + 0.5, w[1] + 0.5, markers[target[i]], markeredgecolor = colors[target[i]], markerfacecolor = 'None',
         markersize = 10, markeredgewidth=2)
show()

# Finding the frauds
mappings = som.win_map(features)
frauds = mappings[(1, 8)]
frauds = sc.inverse_transform(frauds)