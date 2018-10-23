'''
@Author: Koen de Jong & Naomi Voerman
The goal is to find similar users of Netflix
with help of minhasing and LSH.
Minhashing converts large sets to short
signatures, while preserving similarity.
These are short integer vectors that 
represent the sets and reflect their
similarity.
LSH focusses on pairs of signatures
likely to be similar. Similarity is measured
with Jaccard Similarity.
'''

import sys
import time 
import numpy as np
import scipy.sparse as sc

def combinations(mat):
    n = mat.shape[0]
    if n == 2: 
        return(mat)
    
    out = mat[0:2]
    for i in range(n - 2):
        out = np.vstack((out, np.array((mat[0], mat[i+2])))) 
    
    if n > 3:
        out2 = combinations(mat[1:n])
        out = np.vstack((out, out2))       
    return(out)

def Jsim(pair):
    #FUNTION HAS DEPENDENCY ON THE Sparse_MU_csc FROM OUTSIDE THE FUNCTION ENVIRONMENT!!!
    real_sim = np.unique(np.sum(np.hstack((Sparse_MU_csc[:,pair[0]].toarray(),
                                          Sparse_MU_csc[:,pair[1]].toarray())) == [0.,0.], axis = 1),
                        return_counts = True)
    
    indA = real_sim[0]==0
    if sum(indA) == 0:
        A = 0
    else:
        A = real_sim[1][indA]

    indB = real_sim[0]==1
    if sum(indB) == 0:
        B = 0
    else:
        B = real_sim[1][indB]
    sim = A / (A + B)
    
    return(sim)

if __name__ == '__main__':
    total_start = time.time()
    seed = int(sys.argv[1])
    path = sys.argv[2]
    np.random.seed(seed=seed)

    user_movie = np.load(path)

    Sparse_MU = sc.coo_matrix((np.ones(user_movie.shape[0]), (user_movie[:,1], user_movie[:, 0])))
    Sparse_MU_csr = Sparse_MU.tocsr()
    #Sparse_MU_csc = sc.csc_matrix((np.ones(user_movie.shape[0]), (user_movie[:,1], user_movie[:, 0])))

    I = 80 # number of signatures
    M = np.ones((I, Sparse_MU.shape[1]))*np.Inf
    list_permutations = [np.random.permutation(Sparse_MU.shape[0]) for i in np.arange(I)]

    #Minhasing
    print("1. Minhashing: Constructing signature matrix M")
    start = time.time()
    for i in np.arange(I):
        Sparse_temp = Sparse_MU_csr[list_permutations[i], :]
        
        for rowN in np.arange(Sparse_MU.shape[0]):
            ind_nonzero = Sparse_temp[rowN, :].nonzero()[1]
            # minimum of currect cell in matrix M and proposed value
            M[i, ind_nonzero] = np.minimum(M[i, ind_nonzero], rowN+1)
            
            if sum(M[i, :]) < np.Inf:
                break
    end = time.time()
    print("   The time it takes to make signature matrix M is", end - start,"seconds")
    Sparse_MU_csr = None
    Sparse_temp = None
    Sparse_MU_csc = Sparse_MU.tocsc()

    #Local sensitve hashing
    #To buckets:
    print("2. Locale Sensitive hashing: Placing users in buckets per band")
    k = 8
    B = 10
    vec = 10**(np.arange(k))
    mat_extra = np.transpose(np.tile(vec, (M.shape[1], 1)))
    buckets = np.zeros((B, M.shape[1]))
    for i in range(B):
        buckets[i, :] = np.sum(M[i*k:(i*k+k),:]*mat_extra, axis = 0)
        #buckets[i,:] = np.sum(M[i*k:(i*k+k), :], axis = 0)

    #Finding possible similar pairs from buckets:
    print("3. Finding candidate pairs in buckets")
    start = time.time()
    out = np.array((0, 0))
    for i in range(B):
        # find all the considered buckets: at least two movie-indices in bucket
        realised_buckets = np.unique(buckets[i, :], return_counts = True)
        ind = np.where(realised_buckets[1] > 1)[0]
        bucket_values_mult = realised_buckets[0][ind]
        for j in range(bucket_values_mult.shape[0]):
            mat = np.where(buckets[i, :] == bucket_values_mult[j])[0]
            out = np.vstack((out, combinations(mat)))
    out = out[1:, :]
    out_uniq = np.unique(out, axis=0)
    end = time.time()
    print("   The time it takes to create the candidate pairs", end-start,"seconds")

    #Checking similarity in M
    print("Checking for similarity in signature matrix M")
    ind_high_pos = np.array([sum(M[:,out_uniq[i,0]] == M[:,out_uniq[i,1]]) / I for i in range(out_uniq.shape[0])]) > 0.46
    #sum(np.array([sum(M[:,out_uniq[i,0]] == M[:,out_uniq[i,1]]) / Mrow for i in range(out_uniq.shape[0])]) > 0.5)

    #Checking actual similarity in Sparse Matrix
    print("Checking for similarity in actual sparse matrix")
    out_uniq = out_uniq[ind_high_pos,:]
    count = 0
    for i in range(out_uniq.shape[0]):
        out_uniq = np.sort(out_uniq, axis=1)
        sim = Jsim(out_uniq[i, :])
        if sim > 0.5:
            if count == 0:
                f = open('results.txt', 'w')
                f.write(str(out_uniq[i, 0])+','+str(out_uniq[i, 1]))
                f.close()
            else:
                f = open('results.txt', 'a')
                f.write('\n' + str(out_uniq[i, 0])+','+str(out_uniq[i, 1]))
                f.close()
            count += 1
 
    total_end = time.time()
print("Total time:", total_end-total_start, "\nPairs found:", count)
