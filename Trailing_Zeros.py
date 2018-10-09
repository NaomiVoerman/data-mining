# Import libraries
import numpy as np


# Generate a hash result
# Simulating results of hash_function using numpy.random.randint which provides binary stream uniformly distributed.

# Generate a matrix with 500 rows and 10 columns with binary(0,1) bitstreams for the input data
#num = np.random.randint(2 , size=(1 , 10)) 

def trailing_zeroes(num):
    bit_length = len(num)
    if 1 in num:
        i = 0
        while num[-(i+1)] == 0 and i < bit_length:
            i += 1
    else:
        print ("The bitstream does not include a '1'")
   
    return i   
