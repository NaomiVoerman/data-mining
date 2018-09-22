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

ratings_array = np.concatenate((ratings[0], ratings[1], 
    ratings[2])).reshape((-1, 3), order='F')
global_average_rating = ratings.mean(0)[2]

movies.columns = ["MovieID","MovieTitle","Genre"]
ratings.columns = ["UserID","MovieID","Rating","TimeStamp"]
users.columns = ["UserID","Sex","AgeGroup","OccupationGroup",
                                                "ZipCode"]

movies['MovieID'] = movies['MovieID'].apply(pd.to_numeric)

print('shape of movies-matrix:', movies.shape)
print('shape of ratings-matrix:', ratings.shape)
print('shape of users-matrix:', users.shape)

# create X-matrix from ratings data
#I = max(ratings_array[:,0]) # number of users
#J = max(ratings_array[:,1]) # number of movies
#X = np.zeros(shape=(I, J))

#for i in range(len(ratings_array)):
#    # subtract 1 so index wont get
#    # out of bounds.
#    index1 = ratings_array[i,:][0] - 1 
#    index2 = ratings_array[i,:][1] - 1
#    X[index1, index2] = ratings_array[i,:][2]

## count all the cells that are nonzero
#if np.count_nonzero(X) == ratings.shape[0]:
#    print("The X-matrix is filled in correctly")

## fill in the zero-spots with the gobal average
#X[X<1] = global_average_rating
#if np.count_nonzero(X) == I*J:
#    print("The X-matrix contains no zeros")

R_df = ratings.pivot(index = 'UserID', columns ='MovieID', 
    values = 'Rating').fillna(global_average_rating)

X = R_df.as_matrix()
print("The shape of matrix X is:", X.shape)

num_factors = 10
num_iter = 10 # TODO: adjust to 75
regularization = 0.05
learn_rate = 0.005

# initialize U and M
U = np.random.rand(X.shape[0], num_factors)
M = np.random.rand(num_factors, X.shape[1])

def RMSE(x_actual, x_predicted):
    '''
    Calculates the RMSE of two matrices.

    Arguments
    ---------
    x_actual : numpy array
        This is the training part of
        the matrix X.

    x_predicted : numpy array
        This is the training part of
        the predicted matrix X.
    '''

    return np.sqrt(np.mean((x_predicted-x_actual)**2))

def calculate_eij(x, x_pred):
    '''
    Calculate the error for each element
    and store in an numpy array. Each
    element of the output array consists of
    the prediction error for the rating of the
    ith-user and the j-th movie.

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

def calculate_gradient(error, m, u, eta, regularization):
    '''
    Calculate the gradient for each error
    and store in an numpy array.

    Arguments
    --------
    error : np.array
        This is an IxJ numpy array.
    m : np.array
        This is an KxJ numpy array.
    u : np.array
        This is an IxK numpy array.

    Return
    -------
    gradient_m : np.array
        This is an KxJ numpy array.
    gradient_u
        This is an IxK numpy array.
    '''

    # gradient of matrix U
    gradient_u = eta * np.subtract((2 * np.dot(error, m.T)), (regularization*u))

    # gradient of matrix M
    gradient_m = eta * (np.subtract((2 * np.dot(error.T, u).T), (regularization * m)))

    return gradient_m, gradient_u

def update_matrices(u, m, gradient_m, gradient_u):
    '''
    Update the ith row and the jth column
    of the matrix U and M.

    Return
    -------
    U_update : np.array
        This is an IxK array.
    M_update : np.array
        This is an KxJ array.
    '''

    U_update = u + gradient_u
    M_update = m + gradient_m
    return U_update, M_update

# apply the algorithm
# loop until the terminal condition is met:
#   that is when the RMSE does not decrease 
#   during two iterations.

# TODO: check where goes wrong in algorithm

X_predicted = np.dot(U, M)

itr = 0
rmse_updated = 0
rmse = 0.1
while (itr < num_iter and rmse_updated != rmse) == True:
    # calculate 'previous' rmse
    rmse = RMSE(X, X_predicted)

    # compute eij
    error = calculate_eij(X, X_predicted)

    # compute the gradient of eij^2
    gradient_m, gradient_u = calculate_gradient(error, M, U,
        eta=learn_rate, regularization=regularization)

    # update the ith row of U and the jth column of M
    U_update, M_update = update_matrices(U, M, gradient_m=gradient_m, 
        gradient_u=gradient_u)

    # calculate the RMSE on the probe subset
    X_predicted = np.dot(U_update, M_update)
    rmse_updated = RMSE(X, X_predicted)
    print(rmse_updated)
    itr = itr + 1