''' Convolutional Neural Network (CNN) '''

# ***** What Convolutional Neural Network does? *****
# It takes the input image, and predicts the output label with the help of trained CNN. It can also helps in telling the
# emotions after seeing the input image. Their are also many other uses of it. What a CNN gives is a probability and not a 
# fixed output. It is over us that what value of threshold we choose for any particular classifier.

# ***** How does the CNN works? *****
# Suppose we have given an black and white input image of pixel matrix - 2x2, then: the visual representation of this picture
# for a CNN is 2d array with every pixel having a value between 0 and 255 (Here, 0 will be completely black pixel and 255 will
# be the completely white pixel and between that will the shades of black, grey and white). While the Colored image of pixel
# matrix 2x2 is stored in a 3d-array, in which the 3rd dimension is for 3 main colors - RGB where each of that consists their
# color intensity in a 2d array.

# Steps involving the formation of CNN:
#   Step1: Convolution
#   Step2: Max Pooling (or DownSampling)
#   Step3: Flatenning
#   Step4: Full Connection
#
# Convolution
#   - Now, suppose we are having an input image, then the pixel matrix of input image are filled with some values in a 2d or
#     3d array based on whether the image is black-and-white or colored. Now, storing the every part of the pixel matrix as a
#     feature will give the detection of the exact figure only (overfitting), so in order to get a generalize model, what we 
#     need is just a particular feature(s) through which we can recognize that image (or that category of images). So, what we
#     are gonna do is to derive a `Feature Detector (or Filter or Kernel)` which is a pixel matrix of a specified size. This
#     feature matrix is then convoluted over the input image matrix dimension by dimension, if some features matches or not we 
#     record it into a resultant matrix known as `Feature Map (or Convolved Feature or Activation Map)` which stores how many
#     of features from the kernel matches in each step convolution.
#     Now, making a feature map keeps some of the feature out of the full image, so to preserve the maximum of the image we 
#     make multiple feature maps which constitutes a `Convolutional Layer`.
#
#   - ReLU Layer
#     Now after getting the input image and applying feature detectors to get feature maps forms the convolutional layer,
#     this is the first layer and then there comes the `Activation Function`, which is in this case is Rectifier activation
#     function which performs best in this case. Since it gives the: f(x) = max(x, 0) and thus preserves x and hence provides
#     non-linearity and we need non-linearity because the images consists of it.    
#
# Max Pooling
#   - There could be an image of some particular animal but in different forms: diagonal, normal, squashed, etc. and we want
#     our NN to detect that animal in every image. So, if a NN learns the features from some images and tries to find that
#     feature in a new image at the exact location then the probability of finding an exact match is very few. So, we have 
#     to make sure that our NN has a property called `spatial invariance` upto some extent and this is done by the `pooling`
#     method. There can be many types of pooling - Min, Max, Sum etc.
#     So what we do in Max Pooling is that, from the Feature Map we roll over a window of particular dimension and gets the 
#     max pixel value in that window (but in this column doesn't overlap). After this we got the `Pooled Feature Map`. In 
#     this, we are not only preserving the features over a wide range of expectations by introducing the spatial variance
#     and also reducing the size (or parameters), hence preventing overfitting.
#     As we have multiple Feature Maps, and each are going to be pooled, so we also going to have multiple pooling feature
#     maps which forms the `Pooling Layer`.
#    
# Flatenning
#   - Now what we have with us till now is the Pooling Layer, which are then going to flattened into the input layer of a
#     future ANN (Input Nodes).
#
# Full Connection
#   - In CNN the hidden layers are called Fully Connected Layers as all the hidden layer neurons are full connected unlike the
#     ANN where the hidden layers are not fully connected.

# ***** Softmax function *****
#   --> f(z) = e**z / sum(e**z)
# Since form the CNN we are getting the probabilities for say two categories - dog and cat, since both are the resultant of 
# some activations of neuron, then their is calculated by those neurons and since both the dog and cat in the output layer are
# not directly connected to each other we have no means of getting the sum of their probabilities as 1 and hence their value
# can add upto any number greater or less than 1. What makes it equal to 1 is the application of softmax function.

# ---------------------- Part-1: Building the CNN -------------------------------------------------
# Importing the libraries and packages
from keras import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step-1: Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))

# Step-2: Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding two more convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Convolution2D(64, 3, 3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step-3: Flattening
classifier.add(Flatten())

# Step-4: Full Connections
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

# Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# ---------------------- Part-2: Fitting the CNN to the layers ------------------------------------
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale= 1./255,
                                   shear_range=0.2,
                                   zoom_range= 0.2,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')
test_set = train_datagen.flow_from_directory('dataset/test_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')
classifier.fit_generator(training_set, steps_per_epoch=8000, epochs=3, validation_data=test_set, validation_steps=2000)

# ---------------------- Part-3: Making new predicitons -------------------------------------------
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
value = classifier.predict(test_image)
result = [res for res, val in training_set.class_indices.items() if val == value]

test_image2 = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg', target_size=(64, 64))
test_image2 = image.img_to_array(test_image2)
test_image2 = np.expand_dims(test_image2, axis=0)
value2 = classifier.predict(test_image2)
result2 = [res for res, val in training_set.class_indices.items() if val == value2]
    