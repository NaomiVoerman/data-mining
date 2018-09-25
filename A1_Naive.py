import pandas as pd
import numpy as np

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
    '''
    Calculates the global average based
    on the data and the train indices.
    Returns a vector of length(Ratings)
    including the global mean.
    '''
    global_average_rating = data.Rating[ind_train].mean(0)
    rating_vec = np.repeat(global_average_rating, data.shape[0])
    return(rating_vec)
    
ratings.Rating.mean()

global_avg_user = ratings.groupby(["UserID"]) [["Rating"]].mean()#()

# 2. the average per user
def glb_avg_user(data,ind_train):
    global_avg_user = data[ind_train].groupby(["UserID"])[["Rating"]].mean() #.transform(mean)
    
    #FallBack 
    rating_vec = glb_avg(data, ind_train)
    
    for id in global_avg_user.index:
        ind = np.where(data["UserID"]==id)
        rating_vec[ind] = global_avg_user.loc[id]
     
    return(rating_vec)

#~1 in global_avg_user.index   
global_avg_user.head(5)

# 3 the average per movie
global_avg_movie = ratings.groupby(["MovieID"])[["Rating"]].mean()

def glb_avg_movie(data, ind_train):
    global_avg_movie = data[ind_train].groupby(["MovieID"])[["Rating"]].mean() #.transform(mean)
    
    #FallBack 
    rating_vec = glb_avg(data, ind_train)
    
    for id in global_avg_movie.index:
        ind = np.where(data["MovieID"]==id)
        rating_vec[ind] = global_avg_movie.loc[id]
     
    return(rating_vec)

global_avg_movie.head(5)

# 4
np.random.seed(610)
Nfolds = 5

#Create folds grouping vector
Nrep = ratings.shape[0] // Nfolds
a = np.repeat(np.arange(Nfolds),Nrep)
b = np.arange(ratings.shape[0] % Nfolds)
folds_vec = np.concatenate([a,b])
np.random.shuffle(folds_vec)

#initialize error vecs
cols = ["global_train", "global_test", "user_train", "user_test", 
    "movie_train", "movie_test", "combined_train", "combined_test"]
err_rmse = pd.DataFrame(index = range(0, Nfolds), columns = cols)
err_mae = pd.DataFrame(index = range(0, Nfolds), columns = cols)

for fold in range(Nfolds):
    ind_test = folds_vec == fold
    ind_train = ~ ind_test
    Ytest = ratings.Rating[ind_test]
    Ytrain = ratings.Rating[ind_train]
      
    global_avg_vec = glb_avg(ratings, ind_train)
    movie_avg_vec = glb_avg_movie(ratings, ind_train)
    user_avg_vec = glb_avg_user(ratings, ind_train)
    
    #training errors
    # rmse
    err_rmse.global_train[fold] = np.sqrt(np.mean((Ytrain - global_avg_vec[ind_train])**2))
    err_rmse.movie_train[fold] = np.sqrt(np.mean((Ytrain - movie_avg_vec[ind_train])**2))
    err_rmse.user_train[fold] = np.sqrt(np.mean((Ytrain - user_avg_vec[ind_train])**2))

    # mae
    err_mae.global_train[fold] = np.mean(abs(Ytrain - global_avg_vec[ind_train]))
    err_mae.movie_train[fold] = np.mean(abs((Ytrain - movie_avg_vec[ind_train])))
    err_mae.user_train[fold] = np.mean(abs((Ytrain - user_avg_vec[ind_train])))
    
    #testing errors
    #rmse
    err_rmse.global_test[fold] = np.sqrt(np.mean((Ytest-global_avg_vec[ind_test])**2))
    err_rmse.movie_test[fold] = np.sqrt(np.mean((Ytest-movie_avg_vec[ind_test])**2))
    err_rmse.user_test[fold] = np.sqrt(np.mean((Ytest-user_avg_vec[ind_test])**2))

    # mae
    err_mae.global_test[fold] = np.mean(abs((Ytest-global_avg_vec[ind_test])))
    err_mae.movie_test[fold] = np.mean(abs((Ytest-movie_avg_vec[ind_test])))
    err_mae.user_test[fold] = np.mean(abs((Ytest-user_avg_vec[ind_test])))

    #linear combination method:
    #training
    #Design matrix
    X1 = user_avg_vec[ind_train]
    X2 = movie_avg_vec[ind_train]
    X3 = np.repeat(1, len(X1))
    X = np.vstack([X1, X2, X3]).T
    
    betas = np.linalg.lstsq(X, Ytrain)[0]
    Ypred = np.dot(X, betas)
    
    #improving Ypred
    Ypred[Ypred > 5] = 5
    Ypred[Ypred < 1] = 1
    
    #training error
    err_rmse.combined_train[fold] = np.sqrt(np.mean((Ytrain-Ypred)**2))
    err_mae.combined_train[fold] = np.mean(abs(Ytrain-Ypred))
    
    #testing
    #Design matrix
    X1 = user_avg_vec[ind_test]
    X2 = movie_avg_vec[ind_test]
    X3 = np.repeat(1, len(X1))
    X = np.vstack([X1, X2, X3]).T
    Ypred = np.dot(X, betas)
    
    #improving Ypred
    Ypred[Ypred>5] = 5
    Ypred[Ypred<1] = 1

    #test error
    err_rmse.combined_test[fold] = np.sqrt(np.mean((Ytest - Ypred)**2))
    err_mae.combined_test[fold] = np.mean(abs(Ytest-Ypred))
    
    print("\nRMSE Fold " + str(fold) + ":")
    print(err_rmse.loc[fold])
    #print("Fold " + str(fold) + ": RMSE_train=" + str(err_train[fold]) + "; RMSE_test=" + str(err_test[fold]))

    print("\nMAE Fold " + str(fold) + ":")
    print(err_mae.loc[fold])

print(err_rmse)
print(err_rmse.mean(0))

print(err_mae)
print(err_mae.mean(0))