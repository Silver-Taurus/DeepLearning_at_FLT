''' Data Representation and Opeartions '''

# -------------------------------- Data Representation --------------------------------------------
# In the above example, we started from the data stored in multidimensional Numpy arrays, also called
# `tensors`. In general, all current machine-learning systems use tensors as their basic data structure.
# Tensors are funcdamental to the fields. At it's core, a tensor is a container for data - almost always
# numerical data.
#
# Scalars (0-D tensors)
#   - A tensor that contains only one number is called a scalar. In Numpy, a float32 or float64 number
#     is a scalar tensor. We can display the number of axes of a Numpy tensor via the `ndim` attribute,
#     a scalar tensor has 0 axes (ndim == 0). The number of axes of a tensor is also called the its `rank`.
import numpy as np
x = np.array(12)
x.ndim
#
# Vectors (1-D tensors)
#   - An array of numbers is called a vector or 1-D tensor. A 1-D tensor is said to have exactly one axis.
x = np.array([12, 231, 5, 2345])
x.ndim
# This vector has four entries and so it also called as 4-D vector. So, a 4-D vector is different from a
# 4-D tensor. A 4-D vector has only one axis and has four dimensions along its axis.
#
# Dimensionality can denote either the number of entries along a specific axes or the number of axes in a
# tensor, which can be confusing at times. So, for tensor it is more relevant to talk about the rank of a
# tensor.
#
# Matrices (2-D tensors)
#   - An array of vectors is a matrix or 2-D tensor. A matrix has two axes (often referred to `rows` and 
#     `columns`).
x = np.array([[1,2,3,4,5],
              [10,20,30,40,50],
              [100,200,300,400,500]])
x.ndim
# The entries of the first axis are called the `rows` and the entries of the second axis are called the 
# `columns`.
#
# 3-D tensors and higher dimensional tensors
#   - If you pack matrices in a new array, you obtain a 3-D tensor.
x = np.array([[[10,20,30,40,50],
               [100,200,300,400,500]], 
            [[11, 21, 31, 41, 51],
             [101,201,301,401,501]]])
x.ndim
# Similarly by packing 3-D tensors in an array, we can create a 4-D tensor and so on.


# A tensor is defined by 3 key attributes:
#   - Number of axes (rank)
#   - Shape (refers to the dimension a tensor has alson each axis)
#   - Data type (dtype)


# Displaying the digit
from keras.datasets import mnist
(train_imgs, train_lables), (test_imgs, test_labels) = mnist.load_data()
import matplotlib.pyplot as plt
digit = train_imgs[1]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()


# Training on the dataset using batches
#   - Full batch
#     In this we provide full dataset for the training and the weights are updated at the
#     end of each epoch.
#   - Stochastic
#     This is reinforcement type approach in which the model is trained for each row and
#     the weights are updated after each row is trained.
#   - Mini_batch
#     This is the practical way that we use in we train the model on the dataset of some
#     fix batch size after which we update the weights. This is more effective as in this
#     the data we are passing is less in memory as well as helps is converging faster as
#     for each epoch, weights are update by (number of rows / batch_size) -> number of
#     times. Also, in comparison to stochastic it gives more reliable answers as in case
#     of stochastic each row affects largely on the updation of the weights that include
#     the case of outliers, but in case of mini-batch, the updation is done on the basis
#     of majority of inputs.


# Real-world examples of data tensors
#   - Vector data: 2D tensors of shape (samples, features)
#   - Timeseries data or sequence data: 3D tensors of shape (samples, timesteps, features)
#   - Images: 4D tensors of shape (samples, height, width, channels) or (samples, channels,
#             height, width)
#   - Video: 5D tensors of shape (samples, frames, height, width, channels)
#
# Vector Data
#   - This is the most common case. In such a dataset, each single data point can be encoded
#     as a vector and thus a batch of data will be encoded as 2D tensor (i.e., an array of
#     vectors), where the first axis is the samples and the second axis is the features axis.
#
# Timeseries data or Sequence data
#   - Whenever time matters in your data (or the notion of sequence order), it makes sense to
#     store it in a 3D tensor with an explicit time axis. Each sample can be encoded as a seq.
#     of vectors w.r.t time (a 2D tensor), and thus a batch of data will be encoded as a 3D
#     tensor.
#
# Image data
#   - Image typically have three dimensions: height, weight and color depth (for colored data).
#     For greyscale images we have two dimensions: height and weight (thus can be stored as 2D
#     tensor). But for a colored image we have a 3D tensor and a batch of these images will be
#     encoded as a 4D tensor.
#
# Video data
#   - Video data is one of the few types of real world data for which we'll need 5D tensors.
#     A video can be understoos as a sequence of frames, each frame being a color image (i.e.,
#     a 3D tensor), so the video will be a sequence of 3D tensors i.e., a 4D tensor.
#     Thus, a batch of different videos can be stored in a 5D tensor.



# -------------------------------- Gears of NN: Tensor Operations ---------------------------------
# Much as any computer program can be ultimately reduced to a small set of binary operations on
# binary inputs (AND, OR, NOR and so on), all transformations learned by deep neural networks can
# be reduced to a handful of tensor oprations applied to a tensors of numeric data.
# So, whenever we calulcate the output of a layer of nueral network, we gets the outputs on the
# following basis:
#       output = activation(dot(input, weights) + bias)
# Normally we use the activation function as `relu`: 
#       relu(x) = max(x, 0)
#
# Element-wise operations
#   - The relu operation and the addition are element-wise operations.
#   - These are the opeartions that are applied independently to each entry in the tensors.
#
# Brodacasting
#   - As, we know the in our dense layer, we added a 2D tensor with a vector. When possible,
#     and if there's no ambiguity, the smaller tensor will be broadcasted to match the shape
#     of the larger tensor. Broadcasting consists of two steps:
#       - Axes (called `broadcase axes`) are added to the smaller tensor to match the ndim
#         of the largest tensor.
#       - The smaller tensor is repeated alongside these new axes to match the full shape
#         of the largest tensor.
#
# Tensor product (dot operator)
#   - The dot operation also called a `tensor product` is the most common, most useful tensor
#     operation. It combines entries in the input tensors.
#   - An element-wise product is done using * operator.
#
# Tensor Reshaping
#   - A third type of tensor operation that's essential to understand is `tensor reshaping`.
#   - A special case of reshaping that's commonly encountered is `transposition`.



# -------------------------------- The engine of NNs: Gradient based opt. -------------------------
# As we know, each neural layer transforms its input data as follows:
#       output = relu(dot(input, weights) + bias)
# In this expression, weights and bias are tensors that are attributes of the layer. They are called
# the weights or trainable parameters of the layer (the kernel and bias attributes, respectively).
# These weights contain the information learned by the network from exposure to training data.
#
# Initially, these weight matrices are filled with small random values (a step called `random
# initialisation`). Ofcourse, there no reason that relu(dot(input, weights) + bias) will yield
# any useful representations. So, the resulting representations are meaningless - but they're
# starting points. What comes next is to gradually adjust the weights, based on a feedback signal.
# This gradual adjustment, also called `training`.
#
# This happens within what’s called a training loop (epoch), which works as follows:
#   - Draw a batch of training samples x and corresponding targets y.
#   - Run the network on x (a step called the forward pass) to obtain predictions y_pred.
#   - Compute the loss of the network on the batch, a measure of the mismatch between y_pred and y.
#   - Update all weights of the network in a way that slightly reduces the loss on this batch.
#
# Trying to find the next value manually after getting the loss for every weight for every sample
# is too much of a work for the computer itself (Curse of dimensionality).
#
# So, what we use is the method of gradient descent, in which we calculate the gradient for the
# tensor operations. Because we’re dealing with a differentiable function, we can compute its gradient,
# which gives you an efficient way to implement step 4. If you update the weights in the opposite
# direction from the gradient, the loss will be a little less every time:
#   - Draw a batch of training samples x and corresponding targets y.
#   -  Run the network on x to obtain predictions y_pred.
#   - Compute the loss of the network on the batch, a measure of the mismatch between y_pred and y.
#   - Compute the gradient of the loss with regard to the network’s parameters (a backward pass).
#   - Move the parameters a little in the opposite direction from the gradient 
#     ex: W -= learning_rate * gradient, thus reducing the loss on the batch a bit.
#
# It is called mini-batch stochastic gradient descent (minibatch SGD). The term stochastic refers to
#the fact that each batch of data is drawn at random (stochastic is a scientific synonym of random).