#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge,LogisticRegression


# In[41]:


df=pd.read_csv("Top Indian Places to Visit.csv")
df.nunique()


# In[42]:


df.drop(["Zone", "State", "City", "Name","Establishment Year","time needed to visit in hrs", "Airport with 50km Radius","Weekly Off","Significance","DSLR Allowed","Best Time to visit"], axis=1, inplace=True)

df = pd.get_dummies(df, dtype='int')


# In[43]:


df


# In[44]:


col=['Type','Google review rating','Number of google review in lakhs']
df.describe()


# In[45]:


df.dropna(inplace=True)
df


# In[46]:


X=df.drop(columns=['Entrance Fee in INR'])
Y=df['Entrance Fee in INR']
X
Y


# In[47]:


X_train_lin, X_test_lin, Y_train_lin, Y_test_lin = train_test_split(X, Y, test_size=0.2, random_state=42)
lin_model = LinearRegression()
lin_model.fit(X_train_lin, Y_train_lin)
lin_model.score(X_train_lin,Y_train_lin)


# In[48]:


lin_model.score(X_test_lin,Y_test_lin)


# In[49]:


from sklearn.linear_model import Lasso

reg2=Lasso(alpha=50,max_iter=100,tol=0.1)
reg2.fit(X_train_lin,Y_train_lin)


# In[50]:


reg2.score(X_train_lin,Y_train_lin)


# In[51]:


reg2.score(X_test_lin,Y_test_lin)


# In[52]:


from sklearn.linear_model import Ridge
import numpy as np 
reg3=Ridge(alpha=50,max_iter=100,tol=0.1)
reg3.fit(X_train_lin,Y_train_lin)


# In[53]:


reg3.score(X_test_lin,Y_test_lin)


# In[54]:


reg3.fit(X_train_lin,Y_train_lin)


# In[55]:


reg3.score(X_train_lin,Y_train_lin)


# In[56]:


from sklearn.linear_model import LogisticRegression
logistic=LogisticRegression(max_iter=1000)
logistic.fit(X_train_lin, Y_train_lin)


# In[57]:


a=logistic.score(X_train_lin,Y_train_lin)
logistic.score(X_train_lin,Y_train_lin)


# In[58]:


logistic.score(X_test_lin,Y_test_lin)


# In[59]:


from sklearn.tree import DecisionTreeClassifier
ds=DecisionTreeClassifier(random_state=42)
ds.fit(X_train_lin, Y_train_lin)


# In[60]:


ds.score(X_train_lin,Y_train_lin)


# In[61]:


ds.score(X_test_lin,Y_test_lin)


# In[ ]:




