import numpy as np
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers


class SVM():
    def __init__(self, kernel, C=1):
        """
        Constructor for SVM classifier
        
        Args:
            kernel (function):
                The kernel function, it needs to take
                TWO vector input and returns a float

            C (float): Regularization parameter

        Returns:
            An initialized SVM object

        Example usage:
            >>> kernel = lambda x, y: numpy.sum(x * y)
            >>> C = 0.1
            >>> model = SVM(kernel, C) 

        ---
        """
        self.kernel = kernel 
        self.C = C
    
    def fit(self, X, y):
        """
        Learns the support vectors by solving the dual optimization
        problem of kernel SVM.

        Args:
            X (numpy.array[float]):
                Input feature matrix with shape (N, d)
            y (numpy.array[float]):
                Binary label vector {-1, +1} with shape (N,)

        Returns:
            None

        """

        # Your code goes here
        m, d = X.shape
        assert (y.shape == (m,))

        X_id = np.identity(m)
        ones = np.ones(m)
        zeros = np.zeros(m)

        K = np.zeros((m,m))
        for i in range(m):
            for j in range(m):
                K[i,j] = self.kernel(X[i], X[j])

        P = cvxopt_matrix(np.outer(y,y) * K)
        q = cvxopt_matrix(ones * -1)
        A = cvxopt_matrix(y.reshape(1, -1))
        b = cvxopt_matrix(0.0)
        dia = np.diag(ones * -1)

        if self.C is None:
            G = cvxopt_matrix(dia)
            h = cvxopt_matrix(zeros)
        else:
            G = cvxopt_matrix(np.vstack((dia, X_id)))
            h = cvxopt_matrix(np.hstack((zeros, ones * self.C)))

        solution = cvxopt_solvers.qp(P, q, G, h, A, b)

        #get alpha
        alphas = solution['x']

        # adjusting alphas
        factor = self.C * 0.000001
        for i in range(len(alphas)):
            if (alphas[i] < factor):
                alphas[i] = 0
            elif (alphas[i] > (self.C - factor)):
                alphas[i] = self.C
        

        valid_alpha = []
        sv_x = []
        sv_y = []
        sv_counter = 0
        b = 0


        for i in range(len(alphas)):
            if (alphas[i] > 0):
                valid_alpha.append(alphas[i])
                sv_x.append(X[i])
                sv_y.append(y[i])
                if (alphas[i] < self.C):
                    sv_counter += 1
                    b += y[i]
                    for j in range(len(alphas)):
                        if alphas[j] > 0:
                            b -= alphas[j] * y[j] * K[i,j]
   

        self.b = b / sv_counter
        self.alpha = valid_alpha
        self.SV = sv_x
        self.SVLabel = sv_y

        return
        
    
    def predict(self, X):
        """
        Predict the label {-1, +1} of input data points using the learned
        support vectors

        Args:
            X (numpy.array):
                The data feature matrix of shape (N,d)
        Returns:
            y_hat (numpy.array):
                The {-1, +1} label vector of shape (N,)
        
        """
        
        # Your code goes here
        m, d = X.shape
        y = []
        l = len(self.SV)
        for i in range(m):
            c = 0
            for j in range(l):
                c += self.alpha[j] * self.SVLabel[j] * self.kernel(self.SV[j], X[i])
            label = np.sign(c + self.b)
            y.append(label)
        return np.array(y)

        
