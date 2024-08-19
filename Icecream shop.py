#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# In[2]:


df=pd.read_csv("C:/Users/home/Downloads/IceCreamData.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


x=df['Temperature']
y=df['Revenue']


# In[6]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)


# In[7]:


len(xtrain)


# In[8]:


len(xtest)


# In[9]:


model=LinearRegression()


# In[10]:


ytest.ndim


# In[11]:


xtrain.ndim


# In[12]:


np.array([xtrain]).ndim


# In[13]:


#model training

model.fit(np.array([xtrain]).reshape(-1,1),ytrain)


# In[14]:


y_pred=model.predict(np.array([xtest]).reshape(-1,1))


# In[15]:


y_pred


# In[16]:


ytest


# In[17]:


r2_score(ytest,y_pred)


# In[18]:


model.predict([[37]])


# In[19]:


with open('model.pkl','wb') as files:
    pickle.dump(model,files)


# In[ ]:




