#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import joblib
import warnings
warnings.simplefilter('ignore')
from azureml.core import Run
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import os

run = Run.get_context() 
df = pd.read_csv('diabetes.csv')
df.head()


# In[12]:


X = df.drop(columns=['PatientID', 'Diabetic'])
y = df['Diabetic']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy_score)
run.log('Accuracy', float(accuracy))

# os.makedirs('model', exist_ok=True)
joblib.dump(model, 'Logisticmodel.joblib')
run.complete()


# In[ ]:




