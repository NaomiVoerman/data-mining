'''
Created on Tue Sep 11 12:20:40 2018

@author: Naomi 

'''

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt

np.random.seed(123)

# reading in the data
movies = pd.read_csv("datasets/movies.dat", sep="::", header = None)
ratings = pd.read_csv("datasets/ratings.dat", sep="::", header = None)
users = pd.read_csv("datasets/users.dat", sep="::", header = None)

ratings_array = np.concatenate((ratings[0], ratings[1], ratings[2])).reshape((-1, 3), order='F')
global_average_rating = ratings.mean(0)[2]

movies.columns = ["MovieID","MovieTitle","Genre"]
ratings.columns = ["UserID","MovieID","Rating","TimeStamp"]
users.columns = ["UserID","Sex","AgeGroup","OccupationGroup","ZipCode"]

print('shape of movies-matrix:', movies.shape)
print('shape of ratings-matrix:', ratings.shape)
print('shape of users-matrix:', users.shape)

# create X-matrix from ratings data
I = max(ratings_array[:,0]) # number of users
J = max(ratings_array[:,1]) # number of movies
X = np.zeros(shape=(I, J)) # TODO calculate the correct X-matrix

for i in range(len(ratings_array)):
    index1 = ratings_array[i,:][0] - 1
    index2 = ratings_array[i,:][1] - 1
    X[index1, index2] = ratings_array[i,:][2]

# count all the cells that are nonzero
if np.count_nonzero(X) == ratings.shape[0]:
    print("The X-matrix is filled in correctly")

# fill in the zero-spots with the gobal average
X[X<1] = global_average_rating
if np.count_nonzero(X) == I*J:
    print("The X-matrix contains no zeros")

num_factors = 10
num_iter = 75
regularization = 0.05
learn_rate = 0.005

# initialize U and M
U = np.empty([I, num_factors])
M = np.empty([num_factors, J])

# apply the algorithm
# loop until the terminal condition is met:
#   that is when the RMSE does not decrease 
#   during two iterations.
def RMSE(x_actual, x_predicted):
    '''
    Calculates the RMSE of two matrices.

    Arguments
    ---------
    x_actual : 
    x_predicted : 
    '''

    return sqrt(mean_squared_error(x_actual, x_predicted))

def calculate_eij(x, x_pred):
    '''
    Calculate the error for each element
    and store in an numpy array.

    Arguments
    --------
    x : np.array
        This is an IxJ numpy array.
    x_pred : np.array
        This is an IxJ numpy array.

    Return
    -------
    error : np.array
        This is an IxJ numpy array.
    '''
    
    error = np.subtract(x, x_pred)
    return error

def calculate_gradient(error, m, u):
    '''
    Calculate the gradient for each error
    and store in an numpy array.

    Arguments
    --------
    error : np.array
        This is an IxJ numpy array.
    m : np.array
        This is an IxK numpy array.
    u : np.array
        This is an KxJ numpy array.

    Return
    -------
    gradient_m : np.array
        This is an ... numpy array.
    gradient_u
        This is an ... numpy array.
    '''

    gradient_m = np.dot(-2*error, m.T)
    gradient_u = np.dot(-2*error.T, u)
    return gradient_m, gradient_u

def update_matrices(u, m, eta, lr, gradient_m, gradient_u, regularization):
    '''

    Return
    -------
    U_update : np.array
        This is an IxK array.
    M_update : np.array
        This is an KxJ array.
    '''
    difference_u = np.subtract(-1 * gradient_u, regularization*u)
    difference_m = np.subtract(-1 * gradient_m.T, regularization*m)

    U_update = u + eta*difference_u
    M_update = m + eta*difference_m
    return U_update, M_update

iter = 0
while iter < num_iter:
    # first calculate predicted X
    X_predicted = np.dot(U, M)
    RMSE = RMSE(X, X_predicted)

    # compute eij
    error = calculate_eij(X, X_predicted)

    # compute the gradient of eij^2
    gradient_m, gradient_u = calculate_gradient(error, M, U)

    # update the ith row of U and the jth column of M
    U_update, M_update = update_matrices(U, M, learn_rate, gradient_m, gradient_u, regularization)

    # calculate the RMSE on the probe subset
    X_predicted = np.dot(U_update, M_update)
    RMSE_update = RMSE(X, X_predicted)


