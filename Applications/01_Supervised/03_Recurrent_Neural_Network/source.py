''' Recurrent Neural Network (RNN) '''

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
