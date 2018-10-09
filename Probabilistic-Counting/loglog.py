# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# -*- coding: utf-8 -*-


import random
import numpy as np
from operator import setitem

#we just convert the string to integer
def getNumeric(prompt):
    while True:
        response = input(prompt)
        try:
            return int(response)
        except ValueError:
            print("Please enter a number.")

#Give the number of buckets
k = getNumeric("Give the number of buckets:")
#Create a matrix with random integers binary [0,1] range of bitstream
test=[[random.randint(0, 1) for _ in range(25)] for _ in range(100000)]
#Distinct unique elements
unique_rows = np.unique(test, axis=0)
d_elem= (len(unique_rows))
#counting the approximate count of buckets
b= 2**k

bb= k-1
#create a list of binary
binary=[]
valueb=[]
value=[]
#create the bitmap
bitmap = np.zeros(shape=(b,1))
#counting the tailing zeroes in the matrix
for y in range(0,len(test)):
  binary=[]
  valueb=[]
  value=[]
#couting the buckets and convert to decimal
  for num,x in enumerate(test[y]):
        binary.append(x)
        if num==bb:
         s = "".join(map(str, binary))
         decimal=int(s,2)
#counting the tailing zeroes of the bitstream without the buckets
  for num,x in enumerate(test[y]):
         if num >bb:
          valueb.append(x)
#compare the bitstream and print the one with the highest zeroes
  for num,x in enumerate(valueb):
         if x==1:
          tet=(num)
          value.append(num)
          p=decimal,value[0]
          if value[0] > bitmap[decimal]:
#set the number of the tailing zeroes in the list
           try:
            setitem(bitmap,decimal,value[0])
           except IndexError:
              continue
#magic number
phi=0.79402
#estimated error of loglog
alpha = 1.30 / np.sqrt(b)
#the real counting according to the algorithm
DV=phi*2** k * 2**(np.mean(bitmap))
#the real error of distinct elements
Error = np.absolute(d_elem - DV)/d_elem
#print distinct elements
print(d_elem)
#print the distinct elements of loglog
print(DV)
#print the estimated error
print(alpha)
#print the RAE
print(Error)