'''
Created on Tue Sep 11 12:20:40 2018

@author: Naomi 

Naive Approach used to estimate the accuracy.
The naive approach that is used is the "global
average rating".

Estimate accuracy with RMSE and MAE.
Use 5-fold cross-validation.
'''

import numpy as np
import pandas as pd

# set seed
np.random.seed(123)

## read in the data with pandas
movies = pd.read_csv('datasets/movies.dat', delimiter='::', header=None)
ratings = pd.read_csv('datasets/ratings.dat', delimiter='::', header=None)
users = pd.read_csv('datasets/users.dat', delimiter='::', header=None)

# extract some useful characteristics
print('shape of movies-matrix:', movies.shape)
print('shape of ratings-matrix:', ratings.shape)
print('shape of users-matrix:', users.shape)

# Calculate the global average rating
global_average_rating = ratings.mean(0)[2]

## make sure that you clip the values.
## they cannot lie outside the range of [1, 5].

## use 5-fold cross-validation to estimate the accuracy