#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import joblib
import numpy as np
from azureml.core.model import Model


# In[3]:


def init():
    global model
    model_path = Model.get_model_path('Logisticmodel')
    model = joblib.load(model_path)
    
def run(raw_data):
    try:
        data = np.array(json.loads(raw_data)['data'])
        #data = np.array(data).reshape(1,-1)
        
        result = model.predict(data)
        
        return json.dumps({'result': result.tolist()})
    except Exception as e:
        return json.dumps({'error': str(e)})


# In[ ]:




