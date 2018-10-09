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

add test
'''

import random
import numpy as np

np.random.seed(123)

## try to simulate part of the stream
#print(random.getrandbits(32))
#print("{0:b}".format(random.getrandbits(32)))

def trailing_zeroes(num):
  '''
  Counts the number of trailing 0 bits in num.

  Arguments
  ---------
  num | binary bit number
    A sequence of 32 contain solely 0's and 1's
    representing a number in 32-bits.
  
  Returns
  --------
  p | integer
    The number of trailing 0's.
  '''

  if num == 0:
    return 32 # Assumes 32 bit integer inputs!
  p = 0
  while (num >> p) & 1 == 0:
    p += 1
  return p

## the actual algorithm
#   input: stream with a single pass
#     - apply hash function h(x)
#     - write down the binary bit calculation
#     - determine the trailing zero
#   the maximum number of trailing zeros = r
#   output: R = 2**r

## the implemented algorithm
#   input: a stream that are hash values
#     - determine the trailing zero
#   the maximum number of trailing zeros = r
#   output: R = 2**r

def flajolet_martin(stream):
  '''
  Calculate the approximation of the
  number of distinct elements.

  Arguments
  ---------
  stream | list
    A long sequence of random 32-bit long 
    integers.

  Return
  --------
    R | integer 
      The approximated number of distinct
      elements.
  '''
  
  max_zero = [0] * len(stream)
  for i in range(len(stream)):

    # determine trailing zero
    max_zero[i] = trailing_zeroes(stream[i])

  # determine r
  r = max(max_zero)
  R = 2**r

  return R

if __name__ == "__main__":
  # run several times and take the mean/median
  #   try this for 10, 20, 30, 40 and 50 runs
  # also run for different number of 'distinct values'

  tries = [10, 20, 30, 40, 50]
  distinct = [1000, 10000, 100000, 1000000]

  fm_means = np.empty([len(tries), len(distinct)])
  fm_median = np.empty([len(tries), len(distinct)])
  for i in range(len(tries)):
    for j in range(len(distinct)):
      # mean
      fm_output = np.array([flajolet_martin([random.getrandbits(32) for k in range(distinct[j])]) for l in range(tries[i])])
      fm_means[i, j] = fm_output.mean()

      # median
      fm_median[i, j] = np.median(fm_output)

  print(fm_means)
  print(fm_median)

  # combine the mean and median
  #   - use k times l sequences of data
  #   say 10 times 1000 sequences
  #   - split into k-groups of size l
  #   - use the median to aggregate the
  #   l results together
  #   - take the mean of group k as estimate

  k = tries[i] 
  j = distinct[j]
  total = k*j
  np.array([flajolet_martin([random.getrandbits(32) for k in range(k*j)]))


  
