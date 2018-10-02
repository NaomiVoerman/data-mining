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

# simulate stream
print(random.getrandbits(32))
print("{0:b}".format(random.getrandbits(32)))

def trailing_zeroes(num):
  '''
  Counts the number of trailing 0 bits in num.

  Arguments
  ---------
  num |
    A 
  
  Returns
  --------
  p | 
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

print([100000/flajolet_martin([random.getrandbits(32) for i in range(100000)]) for j in range(10)])