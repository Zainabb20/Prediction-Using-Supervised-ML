#!/usr/bin/env python
# coding: utf-8

# # Data Science @GRIPSMay'21

# # Task 1 - Prediction Using Supervised ML:
# 

# # Prediction of Percentage of a Student based on the number of hours he/she studies:

# DataSet Used:http://bit.ly/w-data

# # 1)import all the required libraries:

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pylot as plt


# # 2) Load the DataSet:

# In[6]:


file="http://bit.ly/w-data"
data_load=pd.read_csv(file)
print("Successfully imported data into console")


# In[7]:


print(data_load)


# # 3) head() function to  view the data:

# In[8]:


data_load.head()   #returns the first five rows from the dataset


# # 4) plot the data according to your requirements:

# In[15]:


data_load.plot(x='Hours',y='Scores',style='o')
plt.title('Hours vs Percentage')
plt.xlabel('Hours studied')
plt.ylabel('Percentage Score')
plt.show()


# In[ ]:


#Looking to the plot , we can see that the percentage score linearly increases with the no. of hours studied.


# # 5) Divide data into attributes and labels:

# In[16]:


x=data_load.iloc[:,:-1].values
y=data_load.iloc[:,1].values


# # 6) Splitting data into training and test sets:

# In[17]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2,random_state=0)


# # 7)Train the Algorithm:

# In[18]:


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train, y_train)
print("Training....Completed !")


# # 8)Plot test data using trained test data:

# In[19]:


line=regressor.coef_*x+regressor.intercept_
plt.scatter(x,y)
plt.plot(x,line)
plt.show()


# # 9) Predict the scores:

# In[20]:


print(x_test)
y_pred=regressor.predict(x_test)


# # 10) Actual VS Predicted model:

# In[21]:


df=pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
df


# # Time to test our model

# # If a student studies for 9.25hrs/day, approximately how many marks can he get? 

# In[24]:


hours=[[9.25]]
own_pred=regressor.predict(hours)
print("Number of hours={}".format(hours))
print("Prediction of Score={}".format(own_pred[0]))


# # Final Step--Evaluate the model:

# In[25]:


from sklearn import metrics
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,y_pred))


# # By-Zainab Saherwala
#     Data Science Intern  @Grips
#     znbsaheri@gmail.com
