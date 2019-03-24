''' Recurrent Neural Network (RNN) '''

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
