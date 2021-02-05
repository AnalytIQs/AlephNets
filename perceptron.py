########################
### PERCEPTRON CLASS ###
########################

# Import libraries.
import numpy as np
import pandas as pd
import math
import random
from itertools import product

"""
- When the perceptron criterion or hinge loss are used to determine the loss function, it is assumed that the target values are -1 or 1.
- Sigmoid activation function is used with log_likelihood loss function. This model is known as logistic regression, since our objective is to predict a probability.
- Finally, the linear activation function is used with least square regression loss functions. This model is knows as linear regressios; its objective is to predict real values.
"""

# Clase Perceptrón.
class Perceptron: 
  
    # Método constructor.
    def __init__(self, num_neurons, activation = "sign", criterion = "perceptron_criterion", learning_rate = 1):

        if(criterion == "hinge_loss" or criterion == "perceptron_criterion") and (activation != "sign" and activation != "linear"):
            raise Exception(criterion.replace("_", " ").capitalize() + " can only be use with sign or linear activation function.")
        if(criterion == "log_likelihood" and activation != "sigmoid"):
            raise Exception(criterion.replace("_"," ").capitalize() + " can only be used with Sigmoid activation function")
        if(criterion == "least_squares_regression" and activation != "linear"):
            raise Exception(criterion.replace("_"," ").capitalize() + " can only be used with linear activation function")
        if(criterion == "logistic_regression_alternate" and activation != "linear"):
            raise Exception(criterion.replace("_"," ").capitalize() + " can only be used with linear activation function")

        self.num_neurons = num_neurons
        self.activation = getattr(self, "_" + activation)
        self.weights = np.random.rand(1, num_neurons)[0]
        self.learning_rate = learning_rate
        self.criterion = getattr(self, "_" + criterion)
        self.loss_function = getattr(self, "_" + criterion + "_loss")
        
    # Forward method: computational step.
    def forward(self, vector_x): 
        dotproduct = np.dot(vector_x, self.weights)
        return dotproduct
        
    """
    Funciones de activación.
    """
    # Sign activation.
    def _sign(self, x): 
        return np.sign(x)
    
    # Linear activation.
    def _linear(self, x):
        return x

    # Sigmoid activation.
    def _sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
  
    # Tanh activation.
    def _tanh(self, x):
        return 2 * self.sigmoid(2*x)-1

    # Relu activation.
    def _relu(self, x): 
        return max(0, x)
  
    # HardTanh activation.
    def _hardTanh(self, x):
        return max(min(x, 1), -1) 
    
    """
    Loss functions.
    """
    # Perceptron criterion.
    def _perceptron_criterion_loss(self,y_true,y_predicted):
        loss = max(0, -y_true * y_predicted)
        return loss

    # Hinge Loss.
    def _hinge_loss_loss(self,y_true,y_predicted):
        loss = max(0, 1 + (-y_true * y_predicted))
        return loss
    
    # Logistic regression.
    def _logistic_regression_loss(self,y_true,y_predicted):
        loss = -math.log(math.abs(y_true*(1/2)-0.5+y_predicted))
        return loss

    # Least squares regression.
    def _least_squares_regression_loss(self,y_true,y_predicted):
        loss = (y_true-y_predicted)**2
        return loss

    # Logistic regression (alternate).
    def _logistic_regression_alternate_loss(self, y_true, y_predicted):
        loss = math.log(1 + math.exp(-y_true * y_predicted))
        return loss
    
    # Widrow hoff loss.
    def _widrow_hoff_loss(self,y_true,y_predicted):
        loss = (y_true-y_predicted)**2
        return loss

    """
    Gradient update functions.
    """
    # 1. Perceptron criterion.
    def _perceptron_criterion(self, y_true, y_predicted, vector_x): 
        if y_true != y_predicted: 
            gradient = np.dot((y_true - y_predicted), vector_x)
            return gradient
        return 0
  
    # 2. Hinge Loss.
    def _hinge_loss(self, y_true, y_predicted, vector_x): 
        gradient = np.dot(y_true, vector_x)
        return gradient

    #3. Log likelihood.
    def _log_likelihood(self, y_true, y_predicted, vector_x):
        gradient = -(y_true*vector_x)/(1+math.exp(y_true*y_predicted))
        return gradient
    
    #4. Least squares.
    def _least_squares_regression(self, y_true, y_predicted, vector_x): 
        gradient = np.dot((y_true - y_predicted), vector_x)
        return gradient
    
    #4. Least squares.
    def _widrow_hoff(self, y_true, y_predicted, vector_x): 
        gradient = np.dot((y_true - y_predicted), vector_x)
        return gradient
    
    """
    Train and predict methods.
    """
    # Train the perceptron.
    def fit(self, X, Y, max_iterations = 100): 

        # Stochastic Gradient Descent.
        for i in range(max_iterations):
     
            # Select random point.
            randomPoint = random.randint(0, len(X) - 1)
            vector_x = X[randomPoint].copy()

            # Predict point category.
            y_predicted = self.forward(vector_x)
            y_true = Y[randomPoint]

            # Compute gradient.
            gradient = self.criterion(y_true, y_predicted, vector_x)

            # Update weights.
            self.weights = self.weights + self.learning_rate * gradient
            
    # Predict value/category.
    def predict(self, vector_x): 
        dotproduct = np.dot(vector_x, self.weights)
        y_predicted = self.activation(dotproduct)
        return y_predicted

    # Get weights.
    def getWeights(self): 
        return self.weights

# Return dataset with bias included.
def add_bias(dataset):
    for element in dataset: 
        element.append(1)
    return dataset

# Predict list.
def predictList(perceptron, X, Y, show_results = False): 

    error_criterion = np.zeros(len(perceptron.weights))
    accuracy = 0

    for i in range(0, len(X)):

        vector_x = X[i].copy()
        
        y_true = Y[i]
        y_predicted = perceptron.predict(vector_x)
        error = perceptron.loss_function(y_true, y_predicted)
        correct = (y_true == y_predicted)

        if show_results: 
            print("Point: ", vector_x)
            print("True Label: ", y_true)
            print("Prediction: ", y_predicted)
            print("Error: ", error)
            print("")

        error_criterion += error
        if correct: 
            accuracy += 1

    print("Criterion Error: ", error_criterion)
    print("Accuracy: ", accuracy / len(X))    