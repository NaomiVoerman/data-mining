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
likely to be similar.
'''

import sys
import time 
import numpy as np
import scipy.sparse as sc

def combinations(mat):
    n = mat.shape[0]
    if n == 2: return(mat)
    out = mat[0:2]
    for i in range(n-2):
        out = np.vstack((out, np.array((mat[0], mat[i+2])))) 
    if n > 3:
        out2 = combinations(mat[1:n])
        out = np.vstack((out, out2))       
    return(out)

def Jsim(pair):
    real_sim = np.unique(np.sum(np.hstack((Sparse_MU[:,pair[0]].toarray(),
                                          Sparse_MU[:,pair[1]].toarray())) == [0.,0.], axis = 1),
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
    seed = int(sys.argv[1])
    path = sys.argv[2]
    np.random.seed(seed=seed)

    user_movie = np.load(path)

    Sparse_MU = sc.csr_matrix((np.ones(user_movie.shape[0]), (user_movie[:,1], user_movie[:, 0])))

    I = 80
    M = np.ones((I,Sparse_MU.shape[1]))*np.Inf
    list_permutations = [np.random.permutation(Sparse_MU.shape[0]) for i in np.arange(I)]

    #start = time.time()
    for i in np.arange(I):
        Sparse_temp = Sparse_MU[list_permutations[i], :]
        
        for rowN in np.arange(Sparse_MU.shape[0]): #np.arange(1):
            
            ind_nonzero = Sparse_temp[rowN,:].nonzero()[1]
            M[i,ind_nonzero] = np.minimum(M[i,ind_nonzero],rowN+1)
            
            if sum(M[i,:]) < np.Inf: break
    #end = time.time()
    #print(end - start)
                            
    #Local sensitve hashing
    #To buckets:
    i=0
    k=10
    B=8
    vec = 10**(np.arange(k))
    mat_extra = np.transpose(np.tile(vec, (M.shape[1],1)))
    buckets = np.zeros((B,M.shape[1]))
    for i in range(B):
        buckets[i,:] = np.sum(M[i*k:(i*k+k),:]*mat_extra, axis = 0)
        #buckets[i,:] = np.sum(M[i*k:(i*k+k),:], axis = 0)

    #Finding possible similar pairs from buckets:
    out = np.array((0,0))
    for i in range(B):
        realised_buckets = np.unique(buckets[i,:], return_counts = True)
        ind = np.where(realised_buckets[1]>1)[0]
        bucket_values_mult = realised_buckets[0][ind]
        for j in range(bucket_values_mult.shape[0]):
            mat = np.where(buckets[i,:] == bucket_values_mult[j] )[0]
            out = np.vstack((out, combinations(mat)))
    out = out[1:,:]
    out_uniq = np.unique(out, axis = 0)

    #Checking similarity in M
    Mrow = M.shape[0]
    ind_high_pos = np.array([sum(M[:,out_uniq[i,0]] == M[:,out_uniq[i,1]]) / Mrow for i in range(out_uniq.shape[0])]) > 0.5
    #sum(np.array([sum(M[:,out_uniq[i,0]] == M[:,out_uniq[i,1]]) / Mrow for i in range(out_uniq.shape[0])]) > 0.5)

    #Checking actual similarity in Sparse Matrix

    #Further specified candidate pairs:
    out_uniq2 = out_uniq[ind_high_pos,:]
    out_uniq2 = list(map(tuple,out_uniq2))
    np.array(map(Jsim, out_uniq2))

    #Writing to txt
    '''
    ## create text file results.txt
    # for each similar pair
    # 1. order such that u1 < u2
    # 2. append to results.txt
    # create text file as output file
    f = open('results.txt','w')
    f.write(str(377)+','+str(103531))
    f.close()

    # append
    f = open('results.txt','a')
    f.write('\n' + str(413)+','+str(3821))
    f.close()
    '''
