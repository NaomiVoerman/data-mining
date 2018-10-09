import pandas as pd
import numpy as np
import time

movies = pd.read_csv("datasets/movies.dat", sep="::", header = None)
ratings = pd.read_csv("datasets/ratings.dat", sep="::", header = None)
users = pd.read_csv("datasets/users.dat", sep="::", header = None)

print('shape of movies-matrix:', movies.shape)
print('shape of ratings-matrix:', ratings.shape)
print('shape of users-matrix:', users.shape)

movies.columns = ["MovieID","MovieTitle","Genre"]
ratings.columns = ["UserID","MovieID","Rating","TimeStamp"]
users.columns = ["UserID","Sex","AgeGroup","OccupationGroup","ZipCode"]

# 1. the global average
def glb_avg(data, ind_train):
    global_average_rating = data.Rating[ind_train].mean(0)
    rating_vec = np.repeat(global_average_rating, data.shape[0])
    return(rating_vec)
    
ratings.Rating.mean()

global_avg_user = ratings.groupby(["UserID"]) [["Rating"]].mean()

print(time.time(), time.clock())