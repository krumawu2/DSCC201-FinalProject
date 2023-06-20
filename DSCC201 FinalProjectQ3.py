
# coding: utf-8

# # Kudakwashe Rumawu
# # DSCC201 Final Project Question 3

# # Part A

# In[80]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[84]:


bank_data = pd.read_csv('bank_data_sk.csv')
bank_data


# # Part B

# In[85]:


bank_data2 = bank_data.drop(columns='CustomerID')
bank_data2


# In[86]:


if bank_data2.isnull().values.any():
    print('The data frame contains missing values.')
else:
    print('The data frame does not contain missing values.')


# In[87]:


df= pd.DataFrame(bank_data2.isnull().any())
df


# In[88]:


bank_data2[bank_data2['CreditScore'].isnull()==True]


# In[89]:


bank_data2[bank_data2['Age'].isnull()==True]


# In[90]:


bank_data2['CreditScore'].fillna(bank_data2['CreditScore'].mean(), inplace=True)
bank_data2['Age'].fillna(bank_data2['Age'].mean(), inplace=True)
if bank_data2.isnull().values.any():
    print('The data frame contains missing values.')
else:
    print('The data frame does not contain missing values.')


# In order to perform the imputation on the age and credit score I used the average of the available age and credit score data respectively

# # Part C

# In[91]:


summary_stats = bank_data2.describe(include="all")
summary_stats


# Balance and estimated salary columns have really large ranges as compared to the other ones. The magnitudes of the other columns are not too different

# # Part D

# In[92]:


from sklearn.preprocessing import OneHotEncoder

# create an instance of OneHotEncoder
encoder = OneHotEncoder(categories=[['New York', 'New Jersey', 'Connecticut']])

# fit and transform the 'State' column using OneHotEncoder
one_hot_encoded = encoder.fit_transform(bank_data2[['State']]).toarray()

# create new column names based on the state names
new_columns = ['New York', 'New Jersey', 'Connecticut']

# create a new DataFrame with the one-hot encoded data
one_hot_df = pd.DataFrame(one_hot_encoded, columns=new_columns)

# concatenate the one-hot encoded DataFrame with the original DataFrame
bank_one_hot = pd.concat([bank_data2, one_hot_df], axis=1)

# remove the original 'State' column
bank_one_hot= bank_one_hot.drop(columns=['State'])

bank_one_hot


# In[93]:


bank_copy = bank_one_hot.copy(deep=True)


# # Part E

# In[94]:


from sklearn.model_selection import train_test_split
train_bank, test_bank = train_test_split(bank_copy, test_size=0.2, random_state=42)


# # Part F

# In[95]:


summary_stats2 = train_bank.describe(include="all")
summary_stats2


# In[52]:


summary_stats3 = test_bank.describe(include="all")
summary_stats3


# Balance and estimated salary columns have really large ranges as compared to the other ones. The magnitudes of the other columns are not too different. I think it is wise to perform normalization on the data

# In[96]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_X_scaled = scaler.fit_transform(train_bank.drop(columns=[]))
test_X_scaled=scaler.fit_transform(test_bank)


# In[97]:


train_X_scaled = pd.DataFrame(train_X_scaled, columns=train_bank.columns)
test_X_scaled = pd.DataFrame(test_X_scaled, columns=test_bank.columns)


# # Part G

# In[98]:


train_X_scaled


# In[99]:


train_y=train_bank['CheckingAcct']


# In[100]:


train_y


# In[101]:


train_X_use=train_X_scaled.drop(columns=['HasCard','Balance','IsActiveMember','CheckingAcct'])
test_X_use=test_X_scaled.drop(columns=['HasCard','Balance','IsActiveMember','CheckingAcct'])


# In[143]:


test_X_use


# In[59]:


train_X_use


# In[102]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(train_X_use,train_y)


# In[103]:


test_y=test_bank['CheckingAcct']
test_y


# In[62]:


train_X_use


# In[104]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
predicted_value=logreg.predict(test_X_use)
print(classification_report(test_y,predicted_value))


# # Part H
# 

# In[105]:


from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

k=5
cv = KFold(n_splits=k, shuffle=True, random_state=42)

scores = cross_val_score(logreg, train_X_use, train_y, cv=cv, scoring='accuracy')

print("Accuracy Scores: ", scores)
print("Average Accuracy: ", scores.mean())
print("Accuracy Standard Deviation: ", scores.std())


# # Part I

# In[106]:


predicted_value = logreg.predict(test_X_use)
print(classification_report(test_y,predicted_value))

accuracy = accuracy_score(test_y, predicted_value)
print("Accuracy on test set: ", accuracy)

confusion = pd.DataFrame(confusion_matrix(test_y, predicted_value))
print("Confusion Matrix:\n", confusion)
confusion


# In[67]:


true_neg=486
false_neg=228
false_pos=93
true_pos=198

false_pos_rate = false_pos / (false_pos + true_neg)
false_neg_rate = false_neg / (false_neg + true_pos)


# In[68]:


false_pos_rate


# The percent of false positives is 16.06%

# In[69]:


false_neg_rate


# The percent of false negatives is 53.52%

# Since the false positive rate is low, this indicates that the model has a low tendency to classify negative instances as positive. The false negative rate is quite and this indicates that the model has a high tendency to classify positive instances as negative. 

# # Part J

# In[107]:


bank_unknown=pd.read_csv('bank_data_unknown.csv')


# In[108]:


bank_unknown


# In[109]:


bank_unknown2=bank_unknown.drop(columns=['CustomerID'])
bank_unknown2


# In[110]:


bank_unknown_in = bank_unknown2
bank_unknown_in 


# In[111]:


# create an instance of OneHotEncoder
encoder = OneHotEncoder(categories=[['New York', 'New Jersey', 'Connecticut']])

# fit and transform the 'State' column using OneHotEncoder
encoded = encoder.fit_transform(bank_unknown_in[['State']]).toarray()

# create new column names based on the state names
new_columns = ['New York', 'New Jersey', 'Connecticut']

# create a new DataFrame with the one-hot encoded data
one_hot_df = pd.DataFrame(encoded, columns=new_columns)

# concatenate the one-hot encoded DataFrame with the original DataFrame
bank_one_hot = pd.concat([bank_unknown_in, one_hot_df], axis=1)

# remove the original 'State' column
bank_one_hot= bank_one_hot.drop(columns=['State'])

bank_one_hot


# In[112]:


bank_unknown_scaled = scaler.fit_transform(bank_one_hot)
bank_unknown_scaled=pd.DataFrame(bank_unknown_scaled, columns=bank_one_hot.columns)
bank_unknown_scaled


# In[113]:


bank_unknown_use = bank_unknown_scaled.drop(columns=['HasCard','Balance','IsActiveMember'])
bank_unknown_use


# In[114]:


predicted_values=logreg.predict(bank_unknown_use) 
predicted_values=pd.DataFrame(predicted_values)
predicted_values


# In[118]:


df=pd.DataFrame(bank_unknown['CustomerID'])
df


# The following customers are likely to sign up for a checking account: 25515560, 25115626, 22815656
