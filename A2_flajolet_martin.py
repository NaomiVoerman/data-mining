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

m = number of bitmaps
k = number of buckets
size = the size of the stream
num_bits = the number of bits

Apply serveral tests for:
- N: size
- M: num_bits
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
k = 5
m = 2**k
size = 100
num_bits = 32

# simulate data from which later on will be sampled again
simulated_data = [random.getrandbits(num_bits) for i in range(size*100)]
simulated_data = np.array(simulated_data)
data = np.unique(simulated_data)

# simulate random 'num_bits'-data of length 'size'
simulated_sequence = np.random.choice(data, size, replace=False)
true_count = len(np.unique(simulated_sequence))
bitmap = np.zeros([m, num_bits-k]) # this is a matrix of shape i*j

# fill in the bitmap as follows:
#   take a hash-value:
#       - idxi: use the first k bits (right to left)
#         as index.
#       - idxj: use the remainder to determine the
#         trailing zeros.

for s in range(size): # len(simulated_sequence)
    # s: individual hash value

    bin_bit = "{0:b}".format(simulated_sequence[s])
    # determine the row index
    rev = bin_bit[::-1]
    idx = rev[0:k][::-1]
    idxi = int(idx, 2)

    # count trailing zeros of the remaining hash values
    # this represents the column index.
    hashv = int(bin_bit[0:k])
    idxj = trailing_zeroes(hashv, num_bits-k)
    
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

phi = 0.77351 # Constant
alpha = 0.78 / np.sqrt(m) # Estimated error
est_count = m/phi * 2**(np.mean(R)) # Probabilistic_Counting error
RAE = np.absolute(true_count - est_count)/true_count # Final Prediction Error
print(RAE)