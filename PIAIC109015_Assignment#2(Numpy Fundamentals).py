#!/usr/bin/env python
# coding: utf-8

# # Numpy_Assignment_2::

# ## Question:1

# ### Convert a 1D array to a 2D array with 2 rows?

# #### Desired output::

# array([[0, 1, 2, 3, 4],
#         [5, 6, 7, 8, 9]])

# In[2]:


import numpy as np
a=np.array([[0, 1, 2, 3, 4],[5, 6, 7, 8, 9]])
a


# ## Question:2

# ###  How to stack two arrays vertically?

# #### Desired Output::
array([[0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1]])
# In[8]:


b=np.ones((2,5))
np.vstack((a,b))


# ## Question:3

# ### How to stack two arrays horizontally?

# #### Desired Output::
array([[0, 1, 2, 3, 4, 1, 1, 1, 1, 1],
       [5, 6, 7, 8, 9, 1, 1, 1, 1, 1]])
# In[9]:


np.hstack((a,b))


# ## Question:4

# ### How to convert an array of arrays into a flat 1d array?

# #### Desired Output::
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# In[10]:


arr = np.arange(15).reshape(1,3,5)
print(arr.ndim,"Dimension")
print(arr)
arr=arr.flatten()
print(arr.ndim,"Dimension")
print(arr)


# ## Question:5

# ### How to Convert higher dimension into one dimension?

# #### Desired Output::
array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
# In[ ]:


arr=np.arange(15).reshape(1,3,5)
print(arr.ndim,"Dimension")
print(arr)
arr=arr.flatten()
print(arr.ndim,"Dimension")
print(arr)


# ## Question:6

# ### Convert one dimension to higher dimension?

# #### Desired Output::
array([[ 0, 1, 2],
[ 3, 4, 5],
[ 6, 7, 8],
[ 9, 10, 11],
[12, 13, 14]])
# In[23]:


# -1 corresponds to the unknown count of the row or column. We can think of it as x(unknown).
# x is obtained by dividing the number of elements in the original array by the other value of the ordered
# pair with -1.
arr = np.arange(15).reshape(-1,3)
arr


# ## Question:7

# ### Create 5x5 an array and find the square of an array?

# In[33]:


arr=np.arange(25).reshape(5,5)
print(arr)
np.square(arr)


# ## Question:8

# ### Create 5x6 an array and find the mean?

# In[44]:


np.random.seed(123)
arr=np.random.randint(30,size=(5,6))
print(arr)
arr.mean()


# ## Question:9

# ### Find the standard deviation of the previous array in Q8?

# In[37]:


np.std(arr)


# ## Question:10

# ### Find the median of the previous array in Q8?

# In[40]:


np.median(arr)


# ## Question:11

# ### Find the transpose of the previous array in Q8?

# In[43]:


print(arr.T)
print(np.transpose(arr))


# ## Question:12

# ### Create a 4x4 an array and find the sum of diagonal elements?

# In[47]:


a=np.arange(16).reshape(4,4)
print(a)
print('Sum =',np.trace(a))
# Finding the diagonal elements of a matrix 
diag = np.diagonal(a) 
print(diag)
print(sum(diag)) 


# ## Question:13

# ### Find the determinant of the previous array in Q12?

# In[54]:


np.linalg.det(a)


# ## Question:14

# ### Find the 5th and 95th percentile of an array?

# In[55]:


print(a)
print('\n',np.percentile(a,5))
print(np.percentile(a,95))


# ## Question:15

# ### How to find if a given array has any null values?

# In[56]:


print(a)
print('\n',np.isnan(a))

