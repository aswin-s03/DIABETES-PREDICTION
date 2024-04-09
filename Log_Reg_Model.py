import numpy as np

class Logistic_Regression():

    def __init__(self, Learning_Rate, no_of_iterations):
        self.Learning_Rate = Learning_Rate
        self.no_of_iterations = no_of_iterations
    
    def fit(self, X, Y):
        self.X = X
        self.Y = Y

        self.m, self.n = X.shape

        self.w = np.zeros(self.n)
        self.b = 0

        for i in range(self.no_of_iterations):
            self.update_weights()
        
    def update_weights(self):
        
        Y_hat = 1 / (1 + np.exp(-(self.X.dot(self.w) + self.b)))

        dw = (np.dot(self.X.T, (Y_hat - self.Y))) / self.m
        db = (np.sum(Y_hat - self.Y)) / self.m

        self.w -= self.Learning_Rate * dw
        self.b -= self.Learning_Rate * db

    def predict(self, X):
        
        Y_hat = 1 / (1 + np.exp(-(X.dot(self.w) + self.b)))

        return np.where(Y_hat > 0.5, 1, 0)