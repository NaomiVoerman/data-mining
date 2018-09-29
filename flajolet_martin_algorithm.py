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

def trailing_zeroes(num):
  """Counts the number of trailing 0 bits in num."""
  if num == 0:
    return 32 # Assumes 32 bit integer inputs!
  p = 0
  while (num >> p) & 1 == 0:
    p += 1
  return p