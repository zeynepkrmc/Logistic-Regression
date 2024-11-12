import numpy as np
class LogisticRegression:

    def __init__(self, alpha = 0.01, regLambda=0.01, epsilon=0.0001, maxNumIters = 10000):
        self.alpha = alpha
        self.regLambda = regLambda
        self.epsilon = epsilon
        self.maxNumIters = maxNumIters
        self.theta = None
        '''
        Constructor
        '''

    

    def computeCost(self, theta, X, y, regLambda):
        n = len(y)  # Number of training examples
        h = self.sigmoid(X @ theta)  # Hypothesis function

        cost = (-1 / n) * (y @ np.log(h) + (1 - y) @ np.log(1 - h))
        regularization = (regLambda / (2 * n)) * np.sum(theta[1:] ** 2)
        total_cost = cost + regularization

        return total_cost
        '''
        Computes the objective function
        Arguments:
            theta is d-dimensional numpy vector
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            regLambda is the scalar regularization constant
        Returns:
            a scalar value of the cost  ** make certain you're not returning a 1 x 1 matrix! **
        '''

    
    
    def computeGradient(self, theta, X, y, regLambda):
        n = len(y)
        h = self.sigmoid(X @ theta)
        
        gradient = (1 / n) * (X.T @ (h - y))
        gradient[1:] += (regLambda / n) * theta[1:]
        
        return gradient
        '''
        Computes the gradient of the objective function
        Arguments:
            theta is d-dimensional numpy vector
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            regLambda is the scalar regularization constant
        Returns:
            the gradient, an d-dimensional vector
        '''
    


    def fit(self, X, y):
        n, d = X.shape
        # Augment X with a column of ones for the intercept term
        X = np.c_[np.ones((n, 1)), X]
        self.theta = np.zeros(d + 1)
        
        
        for i in range(self.maxNumIters):
            gradient = self.computeGradient(self.theta, X, y, self.regLambda)
            new_theta = self.theta - self.alpha * gradient
            

            if np.linalg.norm(new_theta - self.theta, ord=2) < self.epsilon:
                print(f"After {i+1} iterations.")
                break
            
            self.theta = new_theta
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
        ** the d here is different from above! (due to augmentation) **
        '''


    def predict(self, X):
        n = X.shape[0]
        X = np.c_[np.ones((n, 1)), X]  # Augment X with ones for the intercept term
        predictions = self.sigmoid(X @ self.theta)
        return (predictions >= 0.5).astype(int)
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy matrix
        Returns:
            an n-dimensional numpy vector of the predictions, the output should be binary (use h_theta > .5)
        '''


    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))
        '''
        Applies the sigmoid function on every element of Z
        Arguments:
            Z can be a (n,) vector or (n , m) matrix
        Returns:
            A vector/matrix, same shape with Z, that has the sigmoid function applied elementwise
        '''
