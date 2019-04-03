'''  Artificial Neural Network (ANN)'''

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

del build_classifier

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
