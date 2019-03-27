'''  Artificial Neural Network (ANN)'''

# We have the following breakdown for supervised v.s. unsupervised implementaions of Deep learning:
# ---------------------------------------------------------------------------------------
#           Supervised                           |           Unsupervised
# -----------------------------------------------|---------------------------------------
#   -> Artificial Neural Networks                |   -> Self-Organizing Maps
#       - Used for Regression and Classification |       - Used for Feature Detection
#   -> Convolutional Neural Networks             |   -> Deep Boltzmann Machines
#       - Used for Computer Vision               |       - Used for Recommendation Systems
#   -> Recurrent Neural Networks                 |   -> AutoEncoders
#       - Used for Time Series Analysis          |       - Used for Recommendation Systems
# ---------------------------------------------------------------------------------------

# The main concept of Deep Learning is to mimic human brain and the parts of human brain we are interestes in are:
#   - Temporal Lobe (represents the long-term memory - ANN(trained weights))
#   - Occipital Lobe (represents vision - CNN)
#   - Frontal Lobe (represents short-term memory - RNN) 
#   - Parietal Lobe (represents sensation and perception and constructing a spatial co-ordination system)

# ***** The Neuron *****
# The main motive of Deep Learning is to mimc the way human brain works. In the human brain, the signal transmitting units are
# the Neurons which conveys the message and helps in interpreting their meaining on hte basis of prior learning or connections
# that are developed by those neurons over time.
# The Neuron consists of the central body, Dendrites and Axon. On it's own single neuron is not so powerful but when neurons
# group together they can do some pretty heavy work. So, here the dendrites are the signal receiver and the Axon is the
# transmitter. The Axons from one neuron transmits some data in the form of electrical signals to the other neuron but without
# actually getting in touch with the other neuron. The gap between the two at the point of connection where the data is being
# transferred is called `Synapse`.
# Now in Computer science we are going to represent neuron as a node, which has some input signals and can have one or more
# output signals. So, we are going to denote the input signals (or input values) by `X` (X1, X2, ...). The input values in a
# neuron are said to be given by the `Input Layer` where Input Layer being the first layer of the Artificial Neural Network.
# After getting the input and processing the input with or without hidden layers (intermediate neurons) the output is being
# provided by the neuron (`The Output Layer`).
#
# ***** Structure and Elements of Neural Network *****
# The Input Layer
#   - These are the independent varaibles for just one particular observation and they are needed to be standardised or
#     normalised.
#
# The Output Layer or Value
#   - This is the output that is given by the neuron after analyzing the inputs. The output value can be Continuous or Binary
#     or Categorical.
#
# The Synapses
#   - Now every Value or Feature in the Input Layer is connected to a Neuron it is through a Synapse. Each synapse has their
#     own weights and these weights are essential as they are the one which defines the route or the path which is to be
#     taken from the input layer to give an output value. The weights help us to identify that which features in an observation
#     are of more importance for predicting the output.
#
# Inside the Neuron
#   - Now what is happening inside the neuron. Inside the neuron there is `Activation Function` which in general case is the
#     summation of products of weights and the input values. If the value is higher than some threshold value then the signal
#     will be going to be passed further from that neuron.
#
# ``` Activation Function ```
# So, we are going to see the most generally used four types of activation functions:
#   - Threshold Function {f(x) = 1 if x >= 0 and 0 if x < 0}
#   - Sigmoid Function {f(x) = 1 / (1 + e**(-x))}
#   - Rectifier {f(x) = max(x, 0)}
#   - Hyperbolic Tangent {f(x) = (1 - e**(-2x)) / (1 + e**(-2x) }
# So, how happens in a Artificial Neural Network is that in First Input Layer we have got the input values which are given
# the weights at the synapses and then pased down to the hidden layer where the activation function (rectifier function) is
# applied at the each neuron in the hidden layer and then further the singals (or values) are passed to the output layer
# where the final activation function is applied (say: sigmoid function) which gives the output in the desire form as we
# need.

# How do Neural Networks Work?
# So, suppose we have some input parameters X1, X2, X3, X4 and maybe moreoever. So in a simple form we just have the input
# and the output layers only. So in this case the weights are being assigned and evaluated and we get output y as:
#   --> y = w1*x1 + w2*x2 + w3*x3 + w4*x4
# Now, without the hidden layers also it is very powerful, but with the hidden layers included the power level of the
# neural network increses to a much greater extent. Since, say for example we have 5 neurons in the hidden layer all are
# fully-connected to the input layers and then the input values are being evalueated along with the weights in each of
# those neurons in the hidden layer. But not all the hidden layer neurons will get activated from every features. By this
# way the Neural Networks make their pattern of giving the output based on the values of input features.
# The more the hidden layer is present the more complex the predictions are going to performed.

# How do Neural Network learn?
# For Neural Network to learn we just give the training set inputs ans training set outputs in which they are gonna be train
# and the rest of the mapping they figures out on their by updating the weights such that they can build a predicting and
# performing model.
# Now this done as, the output value given by the neural network after first iteration is y^ and the actual value is y, then
# the cost function is calculated (C = sum(0.5 * ((y^ - y) ** 2))), where sum denotes the adding of cost function values for
# each row in the dataset and then we are gonna feed this information back to the neural network so that the weights can be
# updated and more accurate results can be provided in the next iteration - this process is known as `Backpropogation`. 
# The lower the cost function value is the better the model is.

# Gradient Descent (or Batch Gradient Descent)
# Now, as we know that the cost function value is evaluated at the end of each iteration. So, if we are going to calculate
# the cost function for each row, than brute forcing the minimum value of the cost function (even if there are only 1000 
# rows which is very low), for say even 25 weights which are also low in numbers, then we have:
#    1000 * 1000 ... * 1000 = 1000 ** 25 combinations 
# which is a very very large computation that is not feasible. This is the `Curse of Dimensionality`.
# So, to overcome this problem, in place of brute forcing every weight to check the minimisation of the cost function, we
# go with the `Gradient Descent`. In this method, in simple terms, for C v.s. y^ graph, for any point on the curve of this
# graph, we find the tangent and hence tries to go downward where the minimum value can be achieved. 
# Now there is the problem in which we can get the local minima and not the global one. This could result in reduced 
# accuracy. Also, the Gradient Descent requires the curve to be convex. Here comes the `Stochastic Gradient Descent` to 
# the rescue!

# Stochastic Gradient Descent
# In this we update the weights with each row and not after the one full iteration, this helps in achieving the global
# minima. This at first seems to be something that will take more computation but actually it helps in converging faster.

# Batch Gradient Descent is the deterministic method while the Stochastic Gradient Descent is the Non-deterministic method.

# Mini-Batch Gradient Descent
# In this we implement the Stochatic Gradient Descent's methodlogy on a particular size of batch rather than on every line.

# ***** Training the ANN with Stochastic Gradient Descent *****
# Step1: Randomly initialise the weights to small numbers close to 0 (but not 0).
# Step2: Input the  first observation of your dataset in the input layer, each feature in one input node.
# Step3: Forward Propagation from left to right, the neurons are activated in a way that the impact of each neuron's
#        activation is limited by the weights. Propagate the activations until getting the predicted result y.
# Step4: Compare tge predicted result to the actual result. Measure the generated error.
# Step5: Back-Propagation: from right to left, the error is back-propagated. Update the weights according to how much they
#        are responsible for the error. The learning rate decides by how much we update the weights.
# Step6: Repeat Steps 1 to 5 and update the weights after each observation (Reinforcement learning). Or: Repeat Steps 1 to
#        5 but update the weights only after a batch of observation (Batch Learning).

  # ---------------------- Part-1: Data Preprocessing -----------------------------------------------
# Importing the libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset = pd.read_csv('Churn_Modelling.csv')

# Making Features matrix and Target vector
features = dataset.iloc[:, 3:13].values
target = dataset.iloc[:, 13].values

# Encoding Categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
features_label_encoder1 = LabelEncoder()
features[:, 1] = features_label_encoder1.fit_transform(features[:, 1])
features_label_encoder2 = LabelEncoder()
features[:, 2] = features_label_encoder2.fit_transform(features[:, 2])
onehotencoder = OneHotEncoder(categorical_features=[1])
features = onehotencoder.fit_transform(features).toarray()
features = features[:, 1:]

# Splitting the dataset into Training set and Test set
from sklearn.model_selection import train_test_split
trained_features, test_features, trained_target, test_target = train_test_split(features, target, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
feature_ssc = StandardScaler()
trained_features = feature_ssc.fit_transform(trained_features)
test_features = feature_ssc.transform(test_features)


# ---------------------- Part-2: Building the ANN -------------------------------------------------
# Importing the Keras libraries and packages
from keras import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layers
classifier.add(Dense(units=6, activation='relu', kernel_initializer='uniform', input_dim=11))
classifier.add(Dropout(rate=0.1))

# Adding more hidden layers
classifier.add(Dense(units=6, activation='relu', kernel_initializer='uniform'))
classifier.add(Dropout(rate=0.1))

# Adding the output layer
classifier.add(Dense(units=1, activation='sigmoid', kernel_initializer='uniform'))

# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#   - Here, `optimizer` is your algorithm for updating weights, that is: Stochastic Gradient Descent (SGD) based `adam` 
#     algorithm.
#   - Also, the loss function which, in normal case, is ordinary least sqaure (ols) function, here for SGD it will be 
#     modified to a logarithmic bassed loss function named `cross_entropy`. Now if we are traning our model for binary 
#     categorical outputs, we have - `binary_crossentropy` function and in case of more than two categorical ouputs, we 
#     have - `categorical_crossentropy` function.

# Fiiting the ANN to the Training set
classifier.fit(trained_features, trained_target, batch_size=10, epochs=100)


# ---------------------- Part-3: Making the Predictions and evaluating the model ------------------
# Predicting the Test set
predicted_target_prob = classifier.predict(test_features)
predicted_target = predicted_target_prob > 0.5

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_target, predicted_target)


# Predicting for a new single observation
#   - The Person is having the following details:
#       Geography --> France
#       Credit Score --> 600
#       Gender --> Male
#       Age --> 40
#       Tenure --> 3
#       Balance --> 60000
#       No. of Products --> 2
#       Has CrCard --> Yes
#       Is Active Member --> Yes
#       Est_Salary --> 50000
new_prediction_prob = classifier.predict(feature_ssc.transform(np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = new_prediction_prob > 0.5


# ---------------------- Part-4: Evaluating, Improving and Tuning the ANN -------------------------
# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
def build_classifier():
    classifier = Sequential()    
    classifier.add(Dense(units=6, activation='relu', kernel_initializer='uniform', input_dim=11))    
    classifier.add(Dense(units=6, activation='relu', kernel_initializer='uniform'))    
    classifier.add(Dense(units=1, activation='sigmoid', kernel_initializer='uniform'))    
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier        
classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=100)
accuracies = cross_val_score(estimator=classifier, X=trained_features, y=trained_target, cv=10)
mean = accuracies.mean()
variance = accuracies.std()

# Improving the ANN
#    - This is done by adding the Dropouts.

# Parameter Tuning (getting the Hyperparameters)
from sklearn.model_selection import GridSearchCV
def build_classifier(optimizer):
    classifier = Sequential()    
    classifier.add(Dense(units=6, activation='relu', kernel_initializer='uniform', input_dim=11))    
    classifier.add(Dense(units=6, activation='relu', kernel_initializer='uniform'))    
    classifier.add(Dense(units=1, activation='sigmoid', kernel_initializer='uniform'))    
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier        
classifier = KerasClassifier(build_fn=build_classifier)
parameters = {'batch_size': [25, 32], 'epochs': [100, 500], 'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10)
grid_search = grid_search.fit(trained_features, trained_target)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

# So, the results we got is: batch_size=25, epochs=500 and optimizer='adam'.
