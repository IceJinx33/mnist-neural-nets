# mnist-neural-nets

In this project, I explored training a neural network on the MNIST dataset using TensorFlow.
I compared the performance of a traditional neural network with 2 linear dense layers with ReLU
activation, followed by another linear layer with output size equal to the number of classes in the
dataset and softmax activation, with a Convolutional Neural Network.

For the traditional neural network, I experimented with Stochastic Gradient Descent as an optimizer
with and without momentum, as well as Batch Normalization. I also used ADAM as an optimizer
for the traditional neural network. The experiments with adding momentum, batch normalization
and using ADAM was to see if the model could be made to converge faster, which it did for each of
the techniques mentioned., with ADAM converging the fastest. However, these methods for faster
convergence take more time to run as compared to normal SGD. Out of all the configurations for the
traditional neural network, Batch Normalization had the highest validation and test accuracy, but was
outperformed by the Convolutional Neural Network.

I also experimented with the values of hyperparameters to check if I could get better results than
the baseline. I used Grid Search and Random Search, both covering the same search space for a SGD
with Momentum model. In Grid Search, as all possible combinations of hyperparameter values are
taken, the number of such values that can be tested is limited, as the number of evaluations required
increases exponentially with each added value. On the other hand, Random Search simulates the
distribution of each hyperparameter, and chooses random combinations to evaluate. us Random
Search is shown to have equal, sometimes even beer hyperparameter results than Grid Search, in
lesser time.
