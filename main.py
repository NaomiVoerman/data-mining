'''
@Author: Naomi Voerman

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
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
from collections import defaultdict

def jaccard_similarity(S1, S2):
    '''
    Calculates the jaccard similarity
    of two sets of movies watched by
    two (different) users.

    S1: np.array
    S2: np.array
    '''
    S1 = set(S1)
    S2 = set(S2)

    intersection = S1 & S2
    union = S1 | S2
    return len(intersection) / len(union)

def probability_similarity(s, r, b):
    '''
    This function is the probability 
    that the signatures will be compared 
    for similarity.

    s: jaccard similarity of two sets
    r: number of rows in each band
    b: number of bands
    '''

    return 1 - ((1 - s**r)**b)

if __name__ == '__main__':
    seed = int(sys.argv[1])
    path = sys.argv[2]
    np.random.seed(seed=seed)

    # load the data
    data = np.load(path)

    # use the sparse package to make a sparse matrix: csc_matrix
    rows = data[::, 1]
    columns = data[::, 0]
    fill = np.ones((data.shape[0]), dtype=int)
    sparse_matrix = csc_matrix((fill, (rows, columns)), dtype=int)
    sparse_array = sparse_matrix.toarray()

    # apply minhashing: create the signature matrix M
    signatures = 50
    r, c = sparse_matrix.shape
    #M = np.full((signatures, sparse_matrix.shape[1]), np.inf)
    M = np.random.randint(0, 250, (signatures, c))

    '''
    list_permutations = [np.random.permutation(r) for i in np.arange(signatures)]

    start = time.time()
    for i in range(signatures):
        shuffled = sparse_matrix[list_permutations[i], :]

        for row in np.arange(r):
            ind_nonzero = shuffled[row, :].nonzero()[1]
            M[i, ind_nonzero] = np.minimum(M[i, ind_nonzero], row+1)

            if sum(M[i,:]) < np.Inf: break
    end = time.time()
    print("The time it takes to create the signature matrix M is", end - start,"seconds")
    print(M)
    '''

    # In order to compare the signatures of all pairs of
    #  columns apply LSH (Locality Sensitive Hashing).

    # calculate trade-off between b and r.
    t = 0.5 # the minimum Jaccard similarity
    '''
    S = np.arange(0, 1, 0.1)
    Sigs = np.arange(50, 150, 10)
    
    for sig in Sigs:
        B = np.arange(5, 15, 1)
        RB = sig/B 

        for j in range(len(B)):
            plt.plot(S, probability_similarity(S, B[j], RB[j]))
            plt.axvline(t, color='black')
            plt.title("n ="+str(sig))
            plt.show()
    '''
    
    b = int(signatures/20) # bands
    rb = int(M.shape[0]/b) # the number of rows per band
    print("the number of bands is",b,"and the number of rows per band is:", rb)
    print("It is", b*rb == signatures, "that the number of signatures is equal to b*r")
    
    buckets = np.zeros((b, c))
    for i in range(b):
        buckets[i, :] = np.sum(M[i*rb:(i*rb+rb), :], axis = 0)

    k = np.zeros((b))
    actual_length = np.zeros((b))
    for i in range(b):
        actual_length[i] = len(buckets[i, :])
        k[i] = len(np.unique(buckets[i, :]))
    print(actual_length)
    print(k) # the number of buckets per band

    ## find the movie pairs with jaccard similarty > 0.5
    start = time.time()
    movie_pairs = []
    for elem in np.unique(buckets[0, :]):
        index_elem = np.where(buckets[0, :] == elem)
        # select the considered pairs
        if (len(index_elem[0])) > 1:
            for i in range(len(index_elem[0])):
                for j in range(len(index_elem[0])):
                    # calculate jaccard similarity
                    if (index_elem[0][i] != index_elem[0][j]):
                        if jaccard_similarity(M[:, i], M[:, j]) > t:
                            # save these pairs of movies
                            movie_pairs.append([index_elem[0][i], index_elem[0][j]])
    print(movie_pairs)
    end = time.time()
    print("The time it takes to create the movie pairs with JS>0.5 is", end-start,"seconds")

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

## --------------------------------------------------------
    ## calculate buckets
    #buckets = [defaultdict(list) for b in range(b)]
    #M_split = np.split(M, b, axis=0)
    #for i in range(b): # b is the number of bands
    #    tempsum = np.sum(M_split[i], axis=0)
    #    for col in range(c):
    #        buckets[i][tempsum[col]].append(col)
