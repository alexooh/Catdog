import numpy as np
import math
class Net(nn.Module):
     def __init__(self, x, y):
        self.input      = x
        #starting value of weights doesn't matter as long as
        self.weights1   = np.random.rand(self.input.shape[1],4) 
        self.weights2   = np.random.rand(4,1)                 
        self.y          = y
        self.output     = np.zeros(self.y.shape)
    def feedforward(self):
        #1/(1+e^-x) sigmoid function
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        # apply chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # add or subtract off of weights based on derivative
        self.weights1 += d_weights1
        self.weights2 += d_weights2
    