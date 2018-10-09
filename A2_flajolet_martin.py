'''
Author: @Naomi Voerman

Can simulate a stream by generating a long sequence 
of random 32-bit long integers (they will look like 
32-bit long hashes of distinct objects). Therefore
there is no need for any hash function.

This script runs multipe experiments for:
- various numbers of distinct elements
- various number of buckets
- various number of setups

The goal is to establish/verify the trade-offs
between the number of "hashes"/"buckets", errors,
amount of required memory as a function of the
number of distinct elements in the stream.

m = the number of vectors in the bitmap
k = number of buckets
size = the size of the stream
num_bits = the number of bits

Apply serveral tests for:
- N: size
- M: memory
- RAE: ..
'''

import random
import numpy as np

np.random.seed(123)

def trailing_zeroes(num, bits):
  '''
  Counts the number of trailing 0 bits in num.

  Arguments
  ---------
  num | binary bit number
    A sequence of solely 0's and 1's.
  bits | integer

  
  Returns
  --------
  p | integer
    The number of trailing 0's.
  '''

  if num == 0:
    return bits
  p = 0
  while (num >> p) & 1 == 0:
    p += 1
  return p

# implement the algorithm for some
# random properties
ks = [4,5,6,7,8,9,10,11,12,13,14,15,16]
sizes = [20**6]
num_bits = 32

# empty arrays to store results in
RAEs = np.zeros([len(sizes), len(ks)])
Estimates = np.zeros([len(sizes), len(ks)])
Alphas = np.zeros([len(ks)])

# simulate random 'num_bits'-data of length 'size'
streams = [[random.getrandbits(num_bits) for i in range(N)] for N in sizes]
true_count = [len(list(set(stream))) for stream in streams]

for j in range(len(streams)):
  stream = streams[j]

  for k in range(len(ks)):
    #print(ks[k])
    m = 2**ks[k]
    bitmap = np.zeros([m, (num_bits-ks[k])])

    # fill in the bitmap as follows:
    #   take a hash-value:
    #       - idxi: use the first k bits (right to left)
    #         as index.
    #       - idxj: use the remainder to determine the
    #         trailing zeros.

    for s in range(len(stream)):
      # s: individual hash value

      bin_bit = "{0:b}".format(stream[s])
      # determine the row index
      rev = bin_bit[::-1]
      idx = rev[0:ks[k]][::-1]
      idxi = int(idx, 2)

      # count trailing zeros of the remaining binary's
      # this represents the column index.
      end = ks[k]
      try:
        hashv = int(bin_bit[0:-end])
      except ValueError:
        hasv = 0
      idxj = trailing_zeroes(hashv, num_bits)

      # assign the index of the bitmap to a 1
      bitmap[idxi, idxj] = 1

    # calculate R as follows:
    #   for every row in the bitmap:
    #   - if it only consists of 1's:
    #       row i = 0
    #   - else:
    #       give the index of where the
    #       first zero

    R = []
    for i in range(m):
      # check which the indexes of each individual row 
      # that is equal to zero.
      r = np.where(bitmap[i, ] == 0)[0]

      # take the smallest index
      r_min = np.amin(r)
      R.append(r_min)   

    phi = 0.77351
    alpha = 0.78 / np.sqrt(m) # Estimated error
    Alphas[k] = alpha
    est_count = m/phi * 2**(np.mean(R)) # Probabilistic_Counting error
    Estimates[j, k] = est_count
    RAEs[j, k] = np.absolute(true_count[j] - est_count)/true_count[j] # Final Prediction Error

print(true_count)
print(Estimates)
print(RAEs)
print(Alphas)

print(2.13962509e-01 == 0.213962509)
print(4.36688484e-03 == 0.00436688484)
print(1.00530251e+01 == 10.0530251)