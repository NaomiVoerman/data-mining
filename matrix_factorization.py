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
I = len(users) # number of users
J = len(movies) # number of movies
X = np.empty(shape=(I, J)) # TODO calculate the correct X-matrix

combinations = tuple(zip(*[ratings_array[:, 0], ratings_array[:, 1]]))
#count = 0

#for i in range(I):
#    for j in range(J):
#        try:
#            index = combinations.index((i, j))
#            test[i ,j] = ratings_array[index, 2]
#
#        except ValueError as err:
#            test[i, j] = global_average_rating
#            count=count+1
#print(count)

num_factors = 10
num_iter = 75
regularization = 0.05
learn_rate = 0.005

# initialize U and M
U = np.empty([I, num_factors])
M = np.empty([num_factors, J])

# apply the algorithm
# loop until the terminal condition is met
# that is when the RMSE does not decrease during two iterations
def RMSE(x_actual, x_predicted):
    '''
    Calculates the RMSE of two arrays.

    Arguments
    ---------
    x_actual : 
    x_predicted : 
    '''

    return sqrt(mean_squared_error(x_actual, x_predicted))

def calculate_eij(x, x_pred):
    '''
    '''
    error = x-x_pred
    return error

def calculate_gradient(error, m, u):
    '''
    '''
    gradient_m = -2*error*m
    gradient_u = -2*error*u
    return gradient_m, gradient_u

def equation_6(u, eta, error, m, lr):
    '''
    '''
    return u + eta * (2*error*m - lr*u)

def equation_7(m, eta, error, u, lr):
    '''
    '''
    return m + eta * (2*error*u - lr*m)

iter = 0
while iter < num_iter:

    # compute eij
    # compute the gradient of eij^2
    # update the ith row of U and the jth column of M
    # calculate the RMSE on the probe subset

