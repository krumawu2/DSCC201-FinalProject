
# coding: utf-8

# ## Kudakwashe Rumawu
# ## DSCC 201 Final Project Question 1
# getting started with python is one of the ways we can ensure that we are learning properly
# 

# In[36]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # Part A

# In[37]:


tumor = pd.read_csv("/public/bmort/python/tumor_cells.csv", header =None)
tumor


# In[39]:


if tumor.isnull().values.any():
    print('The data frame contains missing values.')
else:
    print('The data frame does not contain missing values.')


# In[40]:


#Showing if a column has missing values. True means a column does have missing values
df= pd.DataFrame(tumor.isnull().any())
df


# In[41]:


#The above output shows that column 3 has missing values.
tumor[3][tumor[3].isnull()== True]


# In[42]:


#Data Imputation
tumor[3].fillna(tumor[3].mean(), inplace=True)
if tumor.isnull().values.any():
    print('The data frame contains missing values.')
else:
    print('The data frame does not contain missing values.')


# # Part B

# In[43]:


#Summary Statistics
summary_stats = tumor.describe(include="all")
summary_stats


# In[44]:


tumor_2=tumor.copy(deep=True)
tumor_2.drop(columns=[0,6]).boxplot(figsize=(7,7)) #dropping columns 0 and 6 because they have an inconsistent scale
plt.show()


# The ranges of the columns for the most part are similar. However, other columns have really large ranges than the other ones. Examples of such columns as shown by the above dataframe and box are: 5, 6, 15, 25. The columns have pretty variable magnitudes of ranges. Columns 5, 6, 15, 25 also have some outliers

# # Part C

# In[45]:


tumor_heat = tumor.copy(deep=True)
plt.figure(figsize=(17,10))
sns.heatmap(tumor_heat.drop(columns=[0,1]).corr(),cmap='viridis')


# # Part D

# In[46]:


tumor_copy = tumor.copy(deep=True)
new_col1 = pd.get_dummies(tumor_copy[1],drop_first=True)


# In[47]:


tumor_copy.drop([0,1],axis=1,inplace=True)
tumor_copy=pd.concat([tumor_copy,new_col1],axis=1)


# In[48]:


tumor_copy


# # Part E

# In[49]:


from sklearn.model_selection import train_test_split


# In[50]:


train_tumor, test_tumor = train_test_split(tumor_copy, test_size=0.2, random_state=42)


# In[51]:


train_X = pd.DataFrame(train_tumor.loc[:,2:31])
train_X


# In[52]:


train_y=train_tumor['M']
train_y


# In[53]:


test_X= pd.DataFrame(test_tumor.loc[:,2:31])
test_X


# In[54]:


test_y= test_tumor['M']
test_X


# # Part F

# In[55]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(train_X)
scaler.fit(test_X)
train_X_scaled = scaler.transform(train_X)
test_X_scaled=scaler.transform(test_X)

train_X_scaled = pd.DataFrame(train_X_scaled, columns=train_X.columns)
test_X_scaled = pd.DataFrame(test_X_scaled, columns=test_X.columns)


# In[30]:


test_X_scaled


# # G

# In[56]:


train_X_scaled


# In[57]:


from sklearn.decomposition import PCA
pca = PCA(n_components=6)
pca.fit(train_X_scaled)
pca.fit(test_X_scaled)


# Used 6 components because it captures the maximum variance without overfitting

# In[61]:


train_X_pca = pd.DataFrame(pca.transform(train_X_scaled))
test_X_pca = pd.DataFrame(pca.transform(test_X_scaled))


# # H

# In[59]:


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(train_X_pca,train_y)


# # I

# In[60]:


from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

scores = cross_val_score(logmodel, train_X_pca, train_y, cv=5)

print("Accuracy Scores: ", scores)
print("Average Accuracy: ", scores.mean())
print("Accuracy Standard Deviation: ", scores.std())


# From the output, we can see that the average accuracy of the logistic regression model is 0.96, with a standard deviation of 0.0116. This means that the model is accurate and precise, as it consistently achieves high accuracy across all splits. However, we also need to evaluate the model on the test set to ensure that it generalizes well to new data.

# # J

# In[62]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

predictions = logmodel.predict(test_X_pca)
print(classification_report(test_y,predictions))

accuracy = accuracy_score(test_y, predictions)
print("Accuracy on test set: ", accuracy)

confusion = pd.DataFrame(confusion_matrix(test_y, predictions))
print("Confusion Matrix:\n", confusion)
confusion


# The accuracy on the test set is pretty high and equal to 0.98

# # K

# In[260]:


patient_data = pd.read_pickle('new_patient.pkl')
del patient_data[0]
patient_data= pd.DataFrame(patient_data)
patient_data


# In[266]:


patient_scaled = scaler.fit_transform(patient_data).transpose()
patient_scaled =pd.DataFrame(patient_scaled)
patient_scaled


# In[272]:


patient_pca=pd.DataFrame(pca.transform(patient_scaled))


# In[283]:


patient_pca


# In[284]:


predicted =pd.DataFrame(logmodel.predict(patient_pca))
predicted


# The prediction according to the results of the logistic regression model is that the patient cells are Malignant
