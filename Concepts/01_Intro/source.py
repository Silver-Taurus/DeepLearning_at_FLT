''' Introduction to Deep Learning '''

# Deep learning is a specific subfield of machine learning: a new take on learning representations
# from data that puts an emphasis on learning successive layers of increasingly meaningful repres-
# entations. The `deep` in deep learning stands for the idea of successive layers of representati-
# ons. How many layers contrivute to a model of the data is called the `depth` of the model.
# In deep learning, these layered representations learned via models called `neural networks`, st-
# ructured in literal layers stacked on top of each other.

# What deep learning has achieved so far
#   - Near human level image classification
#   - Near human level speech recognition
#   - Near human level handwritting recognition
#   - Improved machine translation
#   - Improved text-to-speech conversion
#   - Digital assistants such as Google Now and Amazon Alexa
#   - Near human level autonomous driving
#   - Improved ad targeting as used by Google, Bing, etc.
#   - Improved search results on the web
#   - Ability to answer natural language questions
#   - Superhuman Go playing

# Probabilistic Modelling
#   - It is the application of the principles of statistics to data analysis. It was one of the 
#     earliest forms of machine learning, and it's still widely used to this day. One of the best
#     known algorithms in this category is the `Naive Bayes` algorithm. A closely related model is
#     the logistic regression (logreg for short), which is sometimes considered to be the "hello
#     world" of modern machine learning.

# What makes deep learning different
#   - It offers better performance on many problems.
#   - It makes the problem solving much easier, because it completely automates the most important
#     step in a machine learning workflow: `Feature Engineering`



# -------------------------------- Example of neural network -------------------------------------
#   - The MNIST dataset comes preloaded in Keras, in the form of a set of four Numpy arrays.
#   - The images are encoded as Numpy arrays, and the labels are an array of digits, ranging from
#     0 to 9. The images and labels have a one to one correspondence.

# Loading the dataset 
from keras.datasets import mnist
(train_imgs, train_lables), (test_imgs, test_labels) = mnist.load_data()

# Network architecture
from keras.models import Sequential
from keras.layers import Dense
network = Sequential()
network.add(Dense(512, activation='relu', input_shape=(28*28,)))
network.add(Dense(10, activation='softmax'))

# To make the network ready for training, we need to pick three more things, as part of the
# `compilation` step:
#   - A `loss` function --> How the network will be able to measure it's performance on the
#     training data, and thus how it will be able to steer itself in the right direction.
#   - An `optimizer` --> The mechanism through which the network will update itself based on
#     the data it sees and it's loss function.
#   - Metrics to monitor during training and testing

# Compilation step
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Before training the data, we will processes the data by reshaping it into the shape of the network
# expects and scaling it so that all values in the [0, 1] interval of type float32.

# Preparing the data
train_imgs = train_imgs.reshape((60000,28*28)).astype('float32') / 255
test_imgs = test_imgs.reshape((10000,28*28)).astype('float32') / 255

# We also need to categorically encode the labels so that the model can train on it by taking a matrix
# in which only one column is activated which represents the number as well as will contribute in
# matrix operations --> activation(input * weights + bias)

# Preparing the labels
from keras.utils import to_categorical
train_lables = to_categorical(train_lables)
test_labels = to_categorical(test_labels)

# Fitting the network (or model) to its trainiing set data
network.fit(train_imgs, train_lables, epochs=5, batch_size=128)

# Check the model performance on test set data
test_loss, test_acc = network.evaluate(test_imgs, test_labels)
