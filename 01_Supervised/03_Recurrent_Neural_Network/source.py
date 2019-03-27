''' Recurrent Neural Network (RNN) '''

# ***** Recurrent Neural Networks *****
# RNNs can keep track of arbitary short-term memory, i.e., the event that occured in while ago and thus can interpret the current
# happening event, just like the short-term memory of our brain. Now the RNN also consists of the three layers - Input, Hidden and
# Output but the difference the Hidden layer has a recurrent loop on it which shows that the hidden layer just not only gives the 
# output in the present time but also contribute to the output in the future time by contributing to the hidden layer of future.
#
# Types of Recurrent Neural Networks:
#   - One to Many (Picture Interpretation)
#   - Many to one (Sentiment Analysis)
#   - Many to Many (language translation, subtitle movies)

# ***** Vanishing Gradient Problem *****
# Now, as we know, there is recurrent loop on the hidden layer due to which the current NN has an effect over the Future NN. But
# when the time of updation of weight comes the problem arises: Suppose we have a output node in the middle of the timeline, then
# after calculating the cost function we need to back-propagate the weights which are in this case, not just the hidden layer and
# the input layer but all the hidden layers of the past. But the real problem lies here, the more we go to the past hidden layer
# the more multiplication with weights are being done and since the weights are very small in size the output will go on 
# decreasing and hence the learning rate. So in the RNN as we go to past hidden layer nodes the updation of weights occur slowly
# due to which the initial or past nodes learn at a slower rate while the newer nodes update at high rate which leads to the
# formation of an inappropriate model because if the past reference is not correct or much accurate then the future one will not
# be able to train properly as well and on top of that their value keeps no changing at a higher rate leading to the divergence
# from the expected learning output. 
# Now, if you have in this case, large weights rather than smaller weights then you will gonna get `Exploding Gradien Problem`.
#
# Solutions:
#   -> Exploding Gradient (Wrec > 1):
#       - Truncated Back-propagation
#       - Penalties
#       - Gradient Clipping
#   -> Vanishing Gradient (Wrec < 1):
#       - Weight initialisation
#       - Echo State Networks
#       - Long Short-Term Memory Networks (LSTMs)

# ------------------------------------ (LSTM) -----------------------------------------------------
# ---------------------- Part-1: Data Preprocessing -----------------------------------------------
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
training_dataset = pd.read_csv('Google_Stock_Price_Train.csv')
trained_inputs = training_dataset.iloc[:, [1]].values

# Feature Scaling (Normalisation)
from sklearn.preprocessing import MinMaxScaler
features_sc = MinMaxScaler(feature_range=(0, 1))
trained_inputs = features_sc.fit_transform(trained_inputs)

# Creating a data structure with 60 timesteps and 1 output
feature_memory = np.array([trained_inputs[i-60: i, 0] for i in range(60, 1258)])
current_feature = np.array([trained_inputs[i, 0] for i in range(60, 1258)])

# Reshaping
feature_memory = np.reshape(feature_memory, (feature_memory.shape[0], feature_memory.shape[1], 1))


# ---------------------- Part-2: Building the RNN -------------------------------------------------
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout Regularisation
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(feature_memory.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=False))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units=1, ))

# Compiling the RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(feature_memory, current_feature, batch_size=32, epochs=100)


# ---------------------- Part-3: Making the predictions and visualising the results
# Getting the real stock price at 2017
testing_dataset = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = testing_dataset.iloc[:, [1]].values

# Getting the predicted stock price of 2017
dataset = pd.concat((training_dataset['Open'], testing_dataset['Open']), axis=0)
test_inputs = features_sc.transform(dataset[len(dataset) - len(testing_dataset) - 60: ].values.reshape(-1, 1))
test_feature_memory = np.array([test_inputs[i-60: i, 0] for i in range(60, 80)])
test_feature_memory = np.reshape(test_feature_memory, (test_feature_memory.shape[0], test_feature_memory.shape[1], 1))
predicted_stock_price = features_sc.inverse_transform(regressor.predict(test_feature_memory))

# Visualisation the results
plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
