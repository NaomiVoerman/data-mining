'''
@author: Naomi 

Naive Approach used to estimate the accuracy.
The naive approach that is used is an optimal
linear combination of three averages.

Estimate accuracy with RMSE and MAE.
Use 5-fold cross-validation.
'''

import numpy as np
import pandas as pd

# set seed
np.random.seed(123)

# read in the data with pandas
movies = pd.read_csv('datasets/movies.dat', delimiter='::', header=None)
ratings = pd.read_csv('datasets/ratings.dat', delimiter='::', header=None)
users = pd.read_csv('datasets/users.dat', delimiter='::', header=None)

# extract some useful characteristics
print('shape of movies-matrix:', movies.shape)
print('shape of ratings-matrix:', ratings.shape)
print('shape of users-matrix:', users.shape)

## need to calculate the global average.
## this is used when there are mising values.

## make sure that you clip the values.
## they cannot lie outside the range of [1, 5].

## Apply 5-fold cross-validation, where need to
## split all the available data in 5 sections
## at random in more or less equal sizes.
## Apply the model to the part that was not used
## in the training process.
## Retrieve 5 different estimates of the accuracy.
## Use the average as estimate of the error on 
## future data.