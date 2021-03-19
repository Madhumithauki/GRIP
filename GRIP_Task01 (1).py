#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Importing data
url = "http://bit.ly/w-data"
df=pd.read_csv(url)
print("Data imported successfully")
df.head(10)


# In[3]:


#Plotting the data
df.plot(x='Hours', y='Scores',style='s')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()


# In[4]:


#Initialize the variables
X=df.iloc[:, :-1].values
Y=df.iloc[:, 1].values


# In[5]:


#Splitting the data
from sklearn.model_selection import train_test_split  
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0) 


# In[6]:


#Importing Linear Regression or Training the data
from sklearn.linear_model import LinearRegression  
R = LinearRegression()  
R.fit(X_train, Y_train) 
print("Training Complete.")


# In[7]:


#Plotting Regression Line
line=R.coef_*X+R.intercept_

#Plotting for test data
df.plot(x='Hours', y='Scores',style='s')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.scatter(X, Y)
plt.plot(X, line)
plt.show()


# In[8]:


#Testing the data
print(X_test)
Y_pred=R.predict(X_test)


# In[9]:


#Comparing Actual vs Predicted
DF=pd.DataFrame({'Actual':Y_test, 'Predicted':Y_pred})
DF


# In[10]:


#Predict Own Data
hours=9.25
own_pred=R.predict(np.array(hours).reshape(-1,1))
print("Number of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# In[11]:


#Finding mean
from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(Y_test, Y_pred)) 

