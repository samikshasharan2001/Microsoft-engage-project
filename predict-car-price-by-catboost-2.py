#!/usr/bin/env python
# coding: utf-8

# ## importing modules

# In[25]:


import numpy as np
import pandas as pd
import os
import seaborn as sns


# ## Importing data

# In[26]:


df = pd.read_csv('vehicles.csv', index_col=0)
print("data shape:")
print(df.shape)
print("columns name:")
print(list(df.columns))


# ## Data cleaning

# ### delete unnecessary columns that can't be used

# In[27]:


useless_columns = ['url', 'region_url', 'VIN', 'image_url', 'description', 'state']
df.drop(useless_columns, axis=1, inplace=True)
print("data shape:")
print(df.shape)
print("columns name:")
print(list(df.columns))


# ### drop rows with key missing data

# In[28]:


df.isnull().any(axis=0)


# In[29]:


temp = df.dropna(axis=0, subset=['year', 'manufacturer'])
print("Total delete", df.shape[0]-temp.shape[0], "row")
df = temp


# ### delete invalid data

# In[30]:


def count_invalid_num(series):
    """
   Find invalid data, return the number of invalid data
    """
    invalid_num = series.isnull().sum()
    if series.price <= 0:
        invalid_num += 1
    if series.odometer < 0:
        invalid_num += 1
    return invalid_num

df['invalid_num'] = df.apply(count_invalid_num, axis=1)
df.head()


# In[31]:


ori_rows = df.shape[0]
df.drop(df[df.invalid_num > 6].index, inplace=True)
after_rows = df.shape[0]
print("total deleted", ori_rows - after_rows, "row")


# ### deleting rows with price 0

# In[32]:


ori_rows = df.shape[0]
df.drop(df[df['price'] == 0].index, inplace = True) 
after_rows = df.shape[0]
print("total deleted", ori_rows - after_rows, "Row")


# In[33]:


print("The current data shape is：", df.shape)


# ## Data Imputation

# In[34]:


#Interpolation function 

def fill_by_key(data:pd.DataFrame, tar_col:str, type:str, key:str, fill=False, default='GT'):
    if type == 'median':
        tmp = dict(data.groupby(key)[tar_col].median())
    if type == 'mode':
        tmp = dict(df.groupby(key)[tar_col].agg(lambda x: pd.Series.mode(x)))
        for (k, v) in tmp.items():
            if str(v).find('[') != -1:
                print(k, v, '->', default)
                tmp[k] = default
    if type == 'mean':
        tmp = dict(df.groupby(key)[tar_col].mean())
    if fill:
        df[tar_col] = df[tar_col].fillna(df[key].apply(lambda x: tmp.get(x)))
        df.drop(df[df[tar_col].isna()].index, inplace = True)
    else:
        return tmp
    
def fill_helper(data:pd.DataFrame, diction:dict, tar_col:str, key:str):
    df[tar_col] = df[tar_col].fillna(df[key].apply(lambda x: diction.get(x)))
    df.drop(df[df[tar_col].isna()].index, inplace = True)


# ### interpolating odometer values

# In[35]:


fill_by_key(df, 'odometer', 'median', 'year', True, 1000)  # Imputation based on the median of the year
print("The current data shape is：", df.shape)


# In[36]:


df.isnull().any(axis=0)


# ### interpolating model values

# In[37]:


fill_ = fill_by_key(df, 'model', 'mode', 'manufacturer')
fill_helper(df, fill_, 'model', 'manufacturer')


# In[38]:


df[df.manufacturer=='hennessey'].model


# In[39]:


df.isnull().any(axis=0)


# ###  interpolating condition values

# In[40]:


df.condition.unique()


# In[41]:


def condition2int(condition):
    condition_dict = {'salvage':0, 'fair':1, 'good':2, 'excellent':3, 'like new':4, 'new':5}
    try:
        return condition_dict[condition]
    except:
        return np.nan

df['condition'] = df['condition'].apply(condition2int)


# In[42]:


fill_by_key(df, 'condition', 'median', 'year', True, 2)
df.isnull().any(axis=0)


# ### interpolating cylinders values

# In[43]:


df.cylinders.unique()


# In[44]:


def cylinder2int(cylinders):
    try:
        return int(cylinders[0])
    except:
        return np.nan
df['cylinders'] = df['cylinders'].apply(cylinder2int)


# In[45]:


fill_ = fill_by_key(df, 'cylinders', 'median', 'model', default=4)


# In[46]:


fill_helper(df, fill_, 'cylinders', 'model')


# ### interpolating fuel values

# In[47]:


fill_mode = fill_by_key(df, 'fuel', 'mode', 'model', default='gas')


# In[48]:


fill_helper(df, fill_mode, 'fuel', 'model')
df.isnull().any(axis=0)


# ### interpolating title_status values

# In[49]:


df.title_status.unique()


# In[50]:


fill_ = fill_by_key(df, 'title_status', 'mode', 'model', default='clean')
fill_helper(df, fill_, 'title_status', 'model')
df.isnull().any(axis=0)


# ### interpolating transmission, drive, size, paint_color values

# In[51]:


df.transmission.unique()


# In[52]:


fill_ = fill_by_key(df, 'transmission', 'mode', 'model', default='automatic')


# In[53]:


fill_helper(df, fill_, 'transmission', 'model')


# In[54]:


df.isnull().any(axis=0)


# In[55]:


df.drive.unique()


# In[56]:


fill_ = fill_by_key(df, 'drive', 'mode', 'model', default='fwd')


# In[57]:


fill_helper(df, fill_, 'drive', 'model')
df.isnull().any(axis=0)


# In[58]:


df.paint_color.unique()


# In[59]:


fill_ = fill_by_key(df, 'paint_color', 'mode', 'model', default='white')


# In[60]:


fill_helper(df, fill_, 'paint_color', 'model')
df.isnull().any(axis=0)


# In[61]:


df['size'].unique()


# In[62]:


fill_ = fill_by_key(df, 'size', 'mode', 'model', default='mid-size')


# In[63]:


fill_helper(df, fill_, 'size', 'model')
df.isnull().any(axis=0)


# In[64]:


df['type'].unique()


# In[65]:


fill_ = fill_by_key(df, 'type', 'mode', 'model', default='other')


# In[66]:


fill_helper(df, fill_, 'type', 'model')
df.isnull().any(axis=0)


# ### Interpolate the mean of latitude and longitude 

# In[67]:


df.long.fillna(df.long.mean(), inplace=True)
df.lat.fillna(df.lat.mean(), inplace=True)
df.isnull().any(axis=0)


# ## regression model

# In[68]:


df.info()


# In[69]:


df.condition = df.condition.apply(lambda x: int(x))
df.condition.unique()


# In[70]:


df.info()


# In[71]:


df.cylinders = df.cylinders.apply(lambda x: int(x))
df.cylinders.unique()


# In[72]:


from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor

df.drop(columns=["region","posting_date","invalid_num","county"],inplace=True)


# In[73]:


df.head()


# In[74]:


X_train, X_val, y_train, y_val = train_test_split(df.loc[:, [x for x in list(df.columns) if x not in ['price', 'id']]], df.loc[:, 'price'], test_size=0.2 , random_state=2021)
print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)


# In[75]:


df[df["price"]==0]


# In[76]:


categorical_features_indices = np.where(X_train.dtypes != np.float64)[0]
model = CatBoostRegressor(iterations=1000, depth=5, cat_features=categorical_features_indices,learning_rate=0.05, logging_level='Verbose')


# In[77]:


model.fit(X_train, y_train, plot=True)


# ## Predict

# In[78]:


y_hat = model.predict(X_val)


# In[79]:


import plotly.graph_objects as go
import numpy as np
N = 50
x = np.arange(0,N)
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=np.array(y_hat[:N]),mode='lines+markers',name='y_hat'))
fig.add_trace(go.Scatter(x=x, y=np.array(y_val[:N]),mode='lines+markers',name='y_val'))
fig.show()


# ### It is found that the price predicts a negative number, and the data is modified. Here, the impact of outlier data on the model can be reflected.

# In[80]:


y_train_log, y_val_log = np.log(y_train), np.log(y_val)


# In[81]:


model.fit(X_train, y_train_log, plot=True)


# In[82]:


y_hat_log = model.predict(X_val)
y_hat = np.exp(y_hat_log)
N = 60
x = np.arange(0,N)
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=np.array(y_hat[:N]),mode='lines+markers',name='y_hat'))
fig.add_trace(go.Scatter(x=x, y=np.array(y_val[:N]),mode='lines+markers',name='y_val'))
fig.show()


# In[83]:


from sklearn.metrics import mean_squared_error, mean_absolute_error
rmse_val = np.sqrt(mean_squared_error(y_val_log, y_hat_log))
print("RMSE of Validation is: ", rmse_val)


# In[84]:


y_hat_log = model.predict(X_val)
y_hat = np.exp(y_hat_log)


# In[85]:


model


# In[86]:


import pickle
file_name = "cat_reg.pkl"

# save
pickle.dump(model, open(file_name, "wb"))


# In[87]:


model1 = pickle.load(open(file_name, "rb"))


# In[88]:


y_hat_log = model1.predict(X_val)
y_hat = np.exp(y_hat_log)


# In[89]:


y_hat_log = model1.predict(X_val)
y_hat = np.exp(y_hat_log)
N = 60
x = np.arange(0,N)
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=np.array(y_hat[:N]),mode='lines+markers',name='y_hat'))
fig.add_trace(go.Scatter(x=x, y=np.array(y_val[:N]),mode='lines+markers',name='y_val'))
fig.show()


# In[155]:


arr = ['manufacturer', 'model', 'condition', 'cylinders', 'fuel', 'title_status', 'transmission'
       , 'drive', 'size', 'type', 'paint_color']

dic={}
for i in arr :
    u=list(X_train[i].unique())
    dic[i]=u
d=pd.read_csv("uscities.csv")
dic["city"]=list(d["city"])


# In[157]:


import pickle
with open('unique_value.pickle', 'wb') as handle:
    pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[142]:


from catboost import CatBoostClassifier

model1 = CatBoostClassifier()  
model1.load_model('model_save')


# In[141]:


model.save_model("model_save")


# In[145]:


df.manufacturer.unique()


# In[152]:


a={}
for i in df.manufacturer.unique():
    a[i]=set(df[df["manufacturer"]==i]["model"])


# In[154]:


with open('manufacture_model.pickle', 'wb') as handle:
    pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[ ]:




