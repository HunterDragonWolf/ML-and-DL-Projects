#!/usr/bin/env python
# coding: utf-8

# # INTRODUCTION
# 
# ## We'll learn, practise and compare 6 classification models in this project. So, you'll see in this kernel:
# 
# ### 1.Test-Train Datas Split
# 
# ### 2. Desicion Tree Classification
# 
# ### 3. Logistic Regression Classification
# 
# ### 4.Random Forest Classification 
# 
# ### 5.Support Vector Machine (SVM) Classification
# 
# ### 6.Compare all of these Classification Models
# 
# ### PROBLEM STATEMENT : Gender Recognition by Voice and Speech Analysis
# 

# # Deceptive Opinion Spam Detection

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("deceptive-opinion_01.csv")


# In[3]:


df.head()


# In[4]:


df['deceptive'].value_counts()


# In[5]:


placed_no = [ 800 , 800 ]

stat = ['Deceptive' , 'Truthful']

colors = [ 'brown' , 'orange' ]


# In[6]:


plt.pie(placed_no , labels = stat , colors=colors , autopct= '%0.1f%%')


# In[7]:


Hotel = df['hotel'].unique()


# In[8]:


Source = df['source'].unique()


# In[9]:


Polarity = df['polarity'].unique()


# In[10]:


Hotel


# In[11]:


Source


# In[12]:


Polarity


# In[13]:


df.info()


# ## Finding NaN Values

# In[14]:


df.isnull().sum()


# ## Label Encoding

# In[15]:


from sklearn.preprocessing import LabelEncoder


# In[16]:


le_source = LabelEncoder()

le_hotel = LabelEncoder()

le_polarity = LabelEncoder()

le_deceptive = LabelEncoder()


# In[17]:


le_source


# In[18]:


le_hotel


# In[19]:


le_polarity


# In[20]:


le_deceptive


# ## Input

# In[21]:


X = df.drop(['deceptive', 'text'], axis = 1)


# In[22]:


X.head()


# In[23]:


Hotel = X['hotel'].unique()

Polarity = X['polarity'].unique()

Source = X['source'].unique()


# ## Fitting

# In[24]:


X['hotel'] = le_hotel.fit_transform(X['hotel'])


# In[25]:


X.head()


# In[26]:


X['polarity'] = le_polarity.fit_transform(X['polarity'])


# In[27]:


X.head()


# In[28]:


X['source'] = le_hotel.fit_transform(X['source'])


# In[29]:


X.head()


# In[30]:


X.tail()


# ## Hotel Equivalent

# In[31]:


df1 = pd.DataFrame()


# In[32]:


df1['hotel_df']=df['hotel']


# In[33]:


df1['hotel_X']=X['hotel']


# In[34]:


Hotel


# In[35]:


df1.head()


# ## Polarity Equivalent

# In[36]:


df2 = pd.DataFrame()


# In[37]:


df2['polarity_df']=df['polarity']


# In[38]:


df2['polarity_X']=X['polarity']


# In[39]:


Polarity


# In[40]:


df2.head()


# In[41]:


df2.tail()


# ## Source Equivalent

# In[42]:


df3 = pd.DataFrame()


# In[43]:


df3['source_df']=df['source']


# In[44]:


df3['source_X']=X['source']


# In[45]:


Source


# In[46]:


df3.head()


# In[47]:


df3.tail()


# In[48]:


df3.iloc[802:902]


# ## Output/Target

# In[49]:


y = df[['deceptive']]


# In[50]:


y.head()


# In[51]:


y['deceptive'] = le_deceptive.fit_transform(y['deceptive'])


# In[52]:


y.head()                         # Here 1 denotes Truthful


# In[53]:


y.tail()                         # Here 0 denotes Deceptive


# ## Splitting the data into Training and Testing sets

# In[54]:


from sklearn.model_selection import train_test_split


# In[55]:


algo_scores=[]
algo_names=[]


# In[56]:


X_train , X_test , y_train , y_test   = train_test_split(X,y,test_size=0.2)


# In[57]:


len(X_train)


# In[58]:


len(y_train)


# In[59]:


len(X_test)


# In[60]:


len(y_test)


# In[61]:


X_test.shape


# In[62]:


y_test.shape


# In[63]:


X_test.head()


# In[64]:


X_test.tail()


# ## Model Training

# ### Decision Tree

# In[65]:


from sklearn import tree


# In[66]:


model1 = tree.DecisionTreeClassifier()


# In[67]:


model1.fit(X,y)


# In[68]:


algo_names.append("Decision Tree")
algo_scores.append(model1.score(X_test,y_test))


# ### Logistic Regression

# In[69]:


from sklearn.linear_model import LogisticRegression


# In[70]:


model2 = LogisticRegression()


# In[71]:


model2.fit(X,y)


# In[72]:


algo_names.append("Logistic Regression")
algo_scores.append(model2.score(X_test,y_test))


# ### Random Forest

# In[73]:


from sklearn.ensemble import RandomForestClassifier


# In[74]:


model3 = RandomForestClassifier()


# In[75]:


model3.fit(X,y)


# In[76]:


algo_names.append("Random Forest")
algo_scores.append(model3.score(X_test,y_test))


# ### SVM

# In[77]:


from sklearn.svm import SVC


# In[78]:


model4=SVC()


# In[79]:


model4.fit(X,y)


# In[80]:


algo_names.append("SVM")
algo_scores.append(model4.score(X_test,y_test))


# ## Accuracy

# In[81]:


from sklearn import metrics


# In[82]:


model1.score(X,y)                                            


# ### Model 1 = Score of Decision Tree Model = 0.89625

# In[83]:


model2.score(X,y)


# ### Model 2 = Score of Logistic Regression Model = 0.7975

# In[84]:


model3.score(X,y)


# ### Model 3 = Score of Random Forest Model = 0.89625

# In[85]:


model4.score(X,y)


# ### Model 4 = Score of SVM Model = 0.7975

# ## Prediction

# In[86]:


p1=model1.predict(X_test)


# In[87]:


p2=model2.predict(X_test)


# In[88]:


p3=model3.predict(X_test)


# In[89]:


p4=model4.predict(X_test)


# ## Metrics

# In[90]:


metrics.accuracy_score(y_test,p1)


# In[91]:


metrics.accuracy_score(y_test,p2)


# In[92]:


metrics.accuracy_score(y_test,p3)


# In[93]:


metrics.accuracy_score(y_test,p4)


# #### COMPARING THE CLASSIFICATION SCORE OF DIFFERENT ALGORITHMS USING BAR GRAPH

# In[94]:


plt.figure(figsize=(8,8))
plt.ylim([0,3])
plt.bar(algo_names,algo_scores,width=0.3,color= ['brown'])
plt.xlabel('Algorithm Name')
plt.ylabel('Algorithm Score')


# ## Individual Prediction Depiction of Models

# ### Decision Tree

# In[95]:


comp1=[32,573.6]                                                           #  (5% of X_test, score*X_test)
l1=['Wrong','Correct']
plt.pie(comp1,labels=l1,autopct='%0.1f%%',colors=['Brown','Orange'])
plt.title('DECISION TREE PREDICTIONS')
plt.show()


# ### Logistic Regression

# In[96]:


comp2=[64,510.4]                                                           #  (10% of X_test, score*X_test)
l2=['Wrong','Correct']
plt.pie(comp2,labels=l2,autopct='%0.1f%%',colors=['Brown','Orange'])
plt.title('LOGISTIC REGRESSION PREDICTIONS')
plt.show()


# ### Random Forest

# In[97]:


comp3=[44.8,573.6]                                                          #  (7% of X_test, score*X_test)
l3=['Wrong','Correct']
plt.pie(comp3,labels=l3,autopct='%0.1f%%',colors=['Brown','Orange'])
plt.title('RANDOM FOREST PREDICTIONS')
plt.show()


# ### SVM

# In[98]:


comp4=[83.2,510.4]                                                          #  (13% of X_test, score*X_test)
l4=['Wrong','Correct']
plt.pie(comp4,labels=l4,autopct='%0.1f%%',colors=['Brown','Orange'])
plt.title('SVM PREDICTIONS')
plt.show()


# # Conclusion

# ### BASED ON PRECISION , PREDICTION PIE CHART , ACCURACY SCORE , WE CAN SAY THAT RANDOM FOREST CLASSIFICATION AND DECISION FOREST CLASSIFICATION IS BEST SUITED FOR THIS PROBLEM WHERE WE ARE PREDICTING WHETHER THE TEXTS TO VARIOUS HOTELS THROUGH VARIOUS SOURCES ARE DECEPTIVE OR TRUTHFUL.
