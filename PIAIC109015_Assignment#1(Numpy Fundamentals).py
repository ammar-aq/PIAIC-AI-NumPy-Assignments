#!/usr/bin/env python
# coding: utf-8

# # **Assignment For Numpy**

# Difficulty Level **Beginner**

# 1. Import the numpy package under the name np

# In[2]:


import numpy as np


# 2. Create a null vector of size 10 

# In[3]:


np.zeros((10,10))


# 3. Create a vector with values ranging from 10 to 49

# In[5]:


a=np.arange(10,50)
a


# 4. Find the shape of previous array in question 3

# In[7]:


a.shape


# 5. Print the type of the previous array in question 3

# In[8]:


# type shows data type in python,dtype will show type in NumPy.
print(type(a)) 
print(a.dtype)


# 6. Print the numpy version and the configuration
# 

# In[9]:


print(np.version.version)
print(np.show_config())


# 7. Print the dimension of the array in question 3
# 

# In[10]:


print(a)
a.ndim


# 8. Create a boolean array with all the True values

# In[40]:


arr = np.ones(10, dtype=bool)
arr


# 9. Create a two dimensional array
# 
# 
# 

# In[12]:


a=np.array([[1,2,3,5],[1,2,3,4]])
a


# 10. Create a three dimensional array
# 
# 

# In[13]:


a=np.array([[[1,2,3],[3,4,5],[6,7,8]]])
a


# Difficulty Level **Easy**

# 11. Reverse a vector (first element becomes last)

# In[15]:


x=np.array([1,2,3,4,5])
print(x)
print(x[-1::-1])
x = x[::-1]
print(x)


# 12. Create a null vector of size 10 but the fifth value which is 1 

# In[16]:


a=np.zeros(10)
a[4]=1
a


# 13. Create a 3x3 identity matrix

# In[17]:


np.eye(3,3)


# 14. arr = np.array([1, 2, 3, 4, 5]) 
# 
# ---
# 
#  Convert the data type of the given array from int to float 

# In[19]:


arr = np.array([1, 2, 3, 4, 5])
print(arr.dtype)
arr = np.array([1, 2, 3, 4, 5])*1.0
print(arr.dtype)


# 15. arr1 =          np.array([[1., 2., 3.],
# 
#                     [4., 5., 6.]])  
#                       
#     arr2 = np.array([[0., 4., 1.],
#      
#                    [7., 2., 12.]])
# 
# ---
# 
# 
# Multiply arr1 with arr2
# 

# In[20]:


arr1 = np.array([[1., 2., 3.],[4., 5., 6.]])  
arr2 = np.array([[0., 4., 1.],[7., 2., 12.]])
arr1*arr2


# 16. arr1 = np.array([[1., 2., 3.],
#                     [4., 5., 6.]]) 
#                     
#     arr2 = np.array([[0., 4., 1.], 
#                     [7., 2., 12.]])
# 
# 
# ---
# 
# Make an array by comparing both the arrays provided above

# In[21]:


arr1 = np.array([[1., 2., 3.],[4., 5., 6.]])  
arr2 = np.array([[0., 4., 1.],[7., 2., 12.]])
arr=arr1==arr2
arr


# 17. Extract all odd numbers from arr with values(0-9)

# In[38]:


arr=np.arange(0,10)
print(arr)
arr[1::2]


# 18. Replace all odd numbers to -1 from previous array

# In[39]:


arr=np.arange(0,10)
print(arr)
arr[1::2]=1
print(arr)


# 19. arr = np.arange(10)
# 
# 
# ---
# 
# Replace the values of indexes 5,6,7 and 8 to **12**

# In[24]:


arr = np.arange(10)
arr[5:9]=12
arr


# 20. Create a 2d array with 1 on the border and 0 inside

# In[25]:


x=np.ones((5,5))
x[1:-1,1:-1]=0
x


# Difficulty Level **Medium**

# 21. arr2d = np.array([[1, 2, 3],
# 
#                     [4, 5, 6], 
# 
#                     [7, 8, 9]])
# 
# ---
# 
# Replace the value 5 to 12

# In[26]:


arr2d = np.array([[1, 2, 3],[4, 5, 6], [7, 8, 9]])
arr2d[1][1]=12
arr2d


# 22. arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# 
# ---
# Convert all the values of 1st array to 64
# 

# In[29]:


arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
#print(arr3d[0])
arr3d[0]=64
print(arr3d)


# 23. Make a 2-Dimensional array with values 0-9 and slice out the first 1st 1-D array from it

# In[31]:


arr2d=np.array([[0,1,2,3,4],[5,6,7,8,9]])
print(arr2d[0][0:5])
print(arr2d[0])


# 24. Make a 2-Dimensional array with values 0-9 and slice out the 2nd value from 2nd 1-D array from it

# In[32]:


arr2d=np.array([[0,1,2,3,4],[5,6,7,8,9]])
arr2d[1][1]


# 25. Make a 2-Dimensional array with values 0-9 and slice out the third column but only the first two rows

# In[33]:


arr2d=np.array([[0,1,2,3,4],[5,6,7,8,9]])
arr2d[:2,2]


# 26. Create a 10x10 array with random values and find the minimum and maximum values

# In[36]:


a=np.random.randint(100,size=(10,10))
print(a)
print(np.min(a))
print(np.max(a))


# 27. a = np.array([1,2,3,2,3,4,3,4,5,6]) b = np.array([7,2,10,2,7,4,9,4,9,8])
# ---
# Find the common items between a and b
# 

# In[44]:


a= np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])
print(np.unique((a,b)))


# 28. a = np.array([1,2,3,2,3,4,3,4,5,6])
# b = np.array([7,2,10,2,7,4,9,4,9,8])
# 
# ---
# Find the positions where elements of a and b match
# 
# 

# In[45]:


a= np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])
print(np.intersect1d(a,b))
np.where(a==b)


# 29.  names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])  data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will**
# 

# In[51]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) 
data = np.random.randn(7, 4)
print(data)
print(names!="Will")
print(data[names!="Will"])


# 30. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will** and **Joe**
# 
# 

# In[48]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) 
data = np.random.randn(7, 4)
print(names=="Bob")
print(data[names=="Bob"])


# Difficulty Level **Hard**

# 31. Create a 2D array of shape 5x3 to contain decimal numbers between 1 and 15.

# In[53]:


np.arange(1,16).reshape(5,3)


# 32. Create an array of shape (2, 2, 4) with decimal numbers between 1 to 16.

# In[59]:


a=np.arange(1,17).reshape(2,2,4)
print (a)


# 33. Swap axes of the array you created in Question 32

# In[60]:


print(a.T)


# 34. Create an array of size 10, and find the square root of every element in the array, if the values less than 0.5, replace them with 0

# In[66]:


a=np.arange(10)
print(a)
a=np.sqrt(a)
np.where(a<0.5,0,a)


# 35. Create two random arrays of range 12 and make an array with the maximum values between each element of the two arrays

# In[70]:


a=np.array(np.random.rand(12))
b=np.array(np.random.rand(12))
np.maximum(a,b)


# 36. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
# 
# ---
# Find the unique names and sort them out!
# 

# In[89]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
unames=np.unique(names)
print(unames)
print(np.sort(unames))


# 37. a = np.array([1,2,3,4,5])
# b = np.array([5,6,7,8,9])
# 
# ---
# From array a remove all items present in array b
# 
# 

# In[76]:


a = np.array([1,2,3,4,5])
b = np.array([5,6,7,8,9])
print(np.setdiff1d(a,b))


# 38.  Following is the input NumPy array delete column two and insert following new column in its place.
# 
# ---
# sampleArray = numpy.array([[34,43,73],[82,22,12],[53,94,66]]) 
# 
# 
# ---
# 
# newColumn = numpy.array([[10,10,10]])
# 

# In[80]:


sampleArray=np.array([[34,43,73],[82,22,12],[53,94,66]])
newColumn = np.array([[10,10,10]])
sampleArray = np.delete(sampleArray , 1, axis = 1) 
print(sampleArray)
sampleArray = np.insert(sampleArray, 1, newColumn, axis = 1) 
print(sampleArray)


# 39. x = np.array([[1., 2., 3.], [4., 5., 6.]]) y = np.array([[6., 23.], [-1, 7], [8, 9]])
# 
# 
# ---
# Find the dot product of the above two matrix
# 

# In[81]:


x = np.array([[1., 2., 3.], [4., 5., 6.]])
y = np.array([[6., 23.], [-1, 7], [8, 9]])
np.dot(x,y)


# 40. Generate a matrix of 20 random values and find its cumulative sum

# In[85]:


x=np.random.randn(20)
x.cumsum()


# In[ ]:




