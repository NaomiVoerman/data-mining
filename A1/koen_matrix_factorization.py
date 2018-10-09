import pandas as pd
import numpy as np

#Loading data
# movies = pd.read_csv("../data/ml-1m/movies.dat", sep="::", header = None)
ratings = pd.read_csv("../data/ml-1m/ratings.dat", sep="::", header = None)
# users = pd.read_csv("../data/ml-1m/users.dat", sep="::", header = None)

#col names
# movies.columns = ["MovieID","MovieTitle","Genre"]
ratings.columns = ["UserID","MovieID","Rating","TimeStamp"]
# users.columns = ["UserID","Sex","AgeGroup","OccupationGroup","ZipCode"]

#setseed
np.random.seed(348)

def train_MF(ratings, N):
    
    num_factors=10 #
    #num_iter= 75
    Lambda = 0.05 #regularization
    eta = 0.005 #learning rate

    # initialize U and M
    U_rows = ratings["UserID"].unique() 
    M_cols = ratings["MovieID"].unique()
    U = np.random.rand(len(U_rows), num_factors)
    M = np.random.rand(num_factors, len(M_cols))

    num_iter = N
    num_elem = ratings.shape[0]

    RMSE = np.zeros(num_iter)
    MAE = np.zeros(num_iter)

    for i in range(num_iter):
        SE = 0; AE = 0
        print("\nIteration", i+1)
        for j in range(num_elem):

            #time consuming? initalize proper index matrix (j*2)?
            ind_i = np.where(U_rows == ratings.UserID.iloc[j])[0]
            ind_j = np.where(M_cols == ratings.MovieID.iloc[j])[0]

            x_hat = np.dot(U[ind_i,:],M[:,ind_j])
            eij = ratings.Rating.iloc[j] - x_hat
            SE = SE + (eij)**2
            AE = AE + abs(eij)

            #Update with gradients:
            U[ind_i,:] = U[ind_i,:] + eta * (2 * eij * np.transpose(M[:,ind_j]) - Lambda * U[ind_i,:])
            M[:,ind_j] = M[:,ind_j] + eta * (2 * eij * np.transpose(U[ind_i,:]) - Lambda * M[:,ind_j])

            if j % 200000 == 0:
                print("element number:", j)

        RMSE[i] = np.sqrt(SE/num_elem)
        MEA[i] = AE / num_elem
        print("RMSE =", RMSE[i])
        print("MAE =", MAE[i])
    
    return(U, M, RMSE, MAE)

#train_MF(ratings,10)

#Cross validation:
Nfolds = 5
No_train_iter = 20

#Create folds grouping vector
Nrep = ratings.shape[0] // Nfolds
a = np.repeat(np.arange(Nfolds),Nrep)
b = np.arange(ratings.shape[0] % Nfolds)
folds_vec = np.concatenate([a,b])
np.random.shuffle(folds_vec)

RMSE_train = np.zeros(Nfolds)
RMSE_test = np.zeros(Nfolds)

for k in range(Nfold):
    print("\nFOLD:",k+1)
    
    ind_test = folds_vec == k
    ind_train = ~ ind_test
    
    rat_train = ratings[ind_train]
    rat_test = ratings[ind_test]
    
    print("Training on trainset....")
    U_train, M_train, RMSE, MAE = train_MF(rat_train, No_train_iter)
    U_rows = rat_train["UserID"].unique() 
    M_cols = rat_test["MovieID"].unique()
    
    RMSE_train[k] = RMSE[-1]
    MAE_train[k] = MAE[-1]
    
    print("Predicting on testset....")
    #fallback
    u_mean = U_train.mean(0)
    m_mean = M_train.mean(1)
    
    no_elem = len(rat_test.Rating)
    SE = 0
    AE = 0
    for j in range(no_elem):
        m_col = np.where(M_cols == rat_test.MovieID.iloc[j])[0]
        if np.any(m_col):
            m = M_train[:,m_col]
        else:
            m = m_mean
        
        u_row = np.where(U_rows == rat_test.UserID.iloc[j])[0]
        if np.any(m_col):
            u = M_train[u_row,:]
        else:
            u = u_mean
            
        x_hat = np.dot(m,u)
        eij = rat_test.Rating.iloc[j] - x_hat
        SE = SE + eij**2
        AE = AE + abs(eij)
        
    RMSE_test[k]  = np.sqrt(SE/no_elem)
    MAE_test[k] = AE / no_elem
    print("RMSE_test =", RMSE_test[k])
    print("MAE_test =", MAE_test[k])

print(RMSE_train)
print(RMSE_test)

print(MAE_train)
print(MAE_test)