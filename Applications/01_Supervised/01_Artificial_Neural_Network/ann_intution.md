## **ANN Intution**

<br>

We have the following breakdown for supervised v.s. unsupervised implementaions of Deep learning:

 ---------------------------------------------------------------------------------------
           Supervised                           |           Unsupervised
 -----------------------------------------------|---------------------------------------
   -> Artificial Neural Networks                |   -> Self-Organizing Maps
       - Used for Regression and Classification |       - Used for Feature Detection
   -> Convolutional Neural Networks             |   -> Deep Boltzmann Machines
       - Used for Computer Vision               |       - Used for Recommendation Systems
   -> Recurrent Neural Networks                 |   -> AutoEncoders
       - Used for Time Series Analysis          |       - Used for Recommendation Systems
 ---------------------------------------------------------------------------------------

The main concept of Deep Learning is to mimic human brain and the parts of human brain we are interestes in are:
- Temporal Lobe (represents the long-term memory - ANN(trained weights))
- Occipital Lobe (represents vision - CNN)
- Frontal Lobe (represents short-term memory - RNN) 
- Parietal Lobe (represents sensation and perception and constructing a spatial co-ordination system)

<br>

#### **The Neuron**
The main motive of Deep Learning is to mimc the way human brain works. In the human brain, the signal transmitting units are
the Neurons which conveys the message and helps in interpreting their meaining on hte basis of prior learning or connections
that are developed by those neurons over time.
The Neuron consists of the central body, Dendrites and Axon. On it's own single neuron is not so powerful but when neurons
group together they can do some pretty heavy work. So, here the dendrites are the signal receiver and the Axon is the
transmitter. The Axons from one neuron transmits some data in the form of electrical signals to the other neuron but without
actually getting in touch with the other neuron. The gap between the two at the point of connection where the data is being
transferred is called `Synapse`.
Now in Computer science we are going to represent neuron as a node, which has some input signals and can have one or more
output signals. So, we are going to denote the input signals (or input values) by `X` (X1, X2, ...). The input values in a
neuron are said to be given by the `Input Layer` where Input Layer being the first layer of the Artificial Neural Network.
After getting the input and processing the input with or without hidden layers (intermediate neurons) the output is being
provided by the neuron (`The Output Layer`).

<br>

#### **Structure and Elements of Neural Network**
_The Input Layer_
These are the independent varaibles for just one particular observation and they are needed to be standardised or
normalised.

_The Output Layer or Value_
This is the output that is given by the neuron after analyzing the inputs. The output value can be Continuous or Binary
or Categorical.

<br>

#### **The Synapses**
Now every Value or Feature in the Input Layer is connected to a Neuron it is through a Synapse. Each synapse has their
own weights and these weights are essential as they are the one which defines the route or the path which is to be
taken from the input layer to give an output value. The weights help us to identify that which features in an observation
are of more importance for predicting the output.

<br>

#### **Inside the Neuron**
Now what is happening inside the neuron. Inside the neuron there is `Activation Function` which in general case is the
summation of products of weights and the input values. If the value is higher than some threshold value then the signal
will be going to be passed further from that neuron.

<br>

####  **Activation Function**
So, we are going to see the most generally used four types of activation functions:
- Threshold Function {f(x) = 1 if x >= 0 and 0 if x < 0}
- Sigmoid Function {f(x) = 1 / (1 + e**(-x))}
- Rectifier {f(x) = max(x, 0)}
- Hyperbolic Tangent {f(x) = (1 - e**(-2x)) / (1 + e**(-2x) }
So, how happens in a Artificial Neural Network is that in First Input Layer we have got the input values which are given
the weights at the synapses and then pased down to the hidden layer where the activation function (rectifier function) is
applied at the each neuron in the hidden layer and then further the singals (or values) are passed to the output layer
where the final activation function is applied (say: sigmoid function) which gives the output in the desire form as we
need.

<br>

#### **How do Neural Networks Work?**
So, suppose we have some input parameters X1, X2, X3, X4 and maybe moreoever. So in a simple form we just have the input
and the output layers only. So in this case the weights are being assigned and evaluated and we get output y as:
> y = w1*x1 + w2*x2 + w3*x3 + w4*x4 <br>
Now, without the hidden layers also it is very powerful, but with the hidden layers included the power level of the
neural network increses to a much greater extent. Since, say for example we have 5 neurons in the hidden layer all are
fully-connected to the input layers and then the input values are being evalueated along with the weights in each of
those neurons in the hidden layer. But not all the hidden layer neurons will get activated from every features. By this
way the Neural Networks make their pattern of giving the output based on the values of input features.
The more the hidden layer is present the more complex the predictions are going to performed.

<br>

#### **How do Neural Network learn?**
For Neural Network to learn we just give the training set inputs ans training set outputs in which they are gonna be train
and the rest of the mapping they figures out on their by updating the weights such that they can build a predicting and
performing model.
Now this done as, the output value given by the neural network after first iteration is y^ and the actual value is y, then
the cost function is calculated (C = sum(0.5 * ((y^ - y) ** 2))), where sum denotes the adding of cost function values for
each row in the dataset and then we are gonna feed this information back to the neural network so that the weights can be
updated and more accurate results can be provided in the next iteration - this process is known as `Backpropogation`. 
The lower the cost function value is the better the model is.

<br>

#### **Gradient Descent (or Batch Gradient Descent)**
Now, as we know that the cost function value is evaluated at the end of each iteration. So, if we are going to calculate
the cost function for each row, than brute forcing the minimum value of the cost function (even if there are only 1000 
rows which is very low), for say even 25 weights which are also low in numbers, then we have:
> 1000 * 1000 ... * 1000 = 1000 ** 25 combinations <br>
which is a very very large computation that is not feasible. This is the `Curse of Dimensionality`.
So, to overcome this problem, in place of brute forcing every weight to check the minimisation of the cost function, we
go with the `Gradient Descent`. In this method, in simple terms, for C v.s. y^ graph, for any point on the curve of this
graph, we find the tangent and hence tries to go downward where the minimum value can be achieved. 
Now there is the problem in which we can get the local minima and not the global one. This could result in reduced 
accuracy. Also, the Gradient Descent requires the curve to be convex. Here comes the `Stochastic Gradient Descent` to 
the rescue!

<br>

#### **Stochastic Gradient Descent**
In this we update the weights with each row and not after the one full iteration, this helps in achieving the global
minima. This at first seems to be something that will take more computation but actually it helps in converging faster.

`Batch Gradient Descent` is the deterministic method while the `Stochastic Gradient Descent` is the Non-deterministic method.

_Mini-Batch Gradient Descent_
In this we implement the Stochatic Gradient Descent's methodlogy on a particular size of batch rather than on every line.

<br>

#### **Training the ANN with Stochastic Gradient Descent**
1. Randomly initialise the weights to small numbers close to 0 (but not 0).
2. Input the  first observation of your dataset in the input layer, each feature in one input node.
3. Forward Propagation from left to right, the neurons are activated in a way that the impact of each neuron's 
    activation is limited by the weights. Propagate the activations until getting the predicted result y.
4. Compare tge predicted result to the actual result. Measure the generated error.
5. Back-Propagation: from right to left, the error is back-propagated. Update the weights according to how much they
    are responsible for the error. The learning rate decides by how much we update the weights.
6. Repeat Steps 1 to 5 and update the weights after each observation (Reinforcement learning). Or: Repeat Steps 1 to
    5 but update the weights only after a batch of observation (Batch Learning).
