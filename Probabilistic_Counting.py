import random
import numpy as np
from operator import setitem

def getNumeric(prompt):
    while True:
        response = input(prompt)
        try:
            return int(response)
        except ValueError:
            print("Please enter a number.")
print("\n")            
print("Probabilistic_Counting made by Sakis & Georgios")    
print("________________________________________________")        
steam = getNumeric("Give the matrix of steam:")
num_bits = getNumeric("Give the number of bits:")
#m = getNumeric("Number of elements we want to distinct:")
k = getNumeric("Give the number of bits for bucket_id:")
b= 2**k
bb= k-1
# Giorgos

test=[[random.randint(0, 1) for _ in range(num_bits)] for _ in range(steam)]
bitmap = np.zeros(shape=(b, num_bits - k))
bitmap = bitmap.astype(int)
#tes= np.array(test)
unique_rows = np.unique(test, axis=0)
d_elem= (len(unique_rows))
binary=[]
valueb=[]
value=[]
for y in range(0,len(test)):
  binary=[]
  valueb=[]
  value=[]
  for num,x in enumerate(test[y]): # iterate over all the elements in the data
    binary.append(x)
    if num==bb: # if the element is equal to k-1
     s = "".join(map(str, binary))
     decimal=int(s,2) # create the index i
  for num,x in enumerate(test[y]):
     if num >bb:
      valueb.append(x) # create index j
  for num,x in enumerate(valueb): 
     if x==1:
      tet=(num) 
      value.append(num) # create index j
      p=decimal,value[0]
      try:
        setitem(bitmap,p,1) # set the coordinates equal to 1
      except IndexError:
          continue

print(bitmap)

R=[]
for i in range(bitmap.shape[0]): # for each line of bitmap matrix
    
    r = np.where(bitmap[i] == 0)[0] # find the index of the ones
    
    if r.size == 0: # fill in with 0 if there is no zero in the row
        r = num_bits - k
        
    try:    
        rmin = np.amin(r) # take the minimum of the indexing (first 0 in the row)
    except ValueError: # Hack the error...
        pass        
    R.append(rmin)   # list the final indexing of zeroes.   

phi=0.77351 # Constand
alpha = 0.78 / np.sqrt(b) # Estimated error
DV=b/phi * 2**(np.mean(R)) # Probabilistic_Counting error
Error = np.absolute(d_elem - DV)/d_elem # Final Prediction Error
# Printing...
print("\n")
print("The number of unique elements is: " + str(d_elem))
print("\n")
print("Number of distinct elements accornding to the Probabilistic_Counting: " + str(DV))
print("\n")
print("Estimated Error 'alpha': " + str(alpha))
print("\n")
print("The Final Prediction Error is: " + str(Error) + " in range: " + str(steam) + " for number of bits: " + str(num_bits) + " and number of bits for bucket_id: " + str(k))

