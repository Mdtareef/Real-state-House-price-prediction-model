#!/usr/bin/env python
# coding: utf-8

# In[60]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib 
matplotlib.rcParams["figure.figsize"] = (20,10)


# In[61]:


df=pd.read_csv(r"C:\BHP\model\Bengaluru_House_Data.csv")
df


# In[62]:


df.shape


# In[63]:


df.columns


# In[64]:


df['area_type'].unique()


# In[65]:


df['area_type'].value_counts()


# #### Drop features(columns) that are not required to build our model

# In[66]:


df2=df.drop(['area_type','society','balcony','availability'],axis=1)
df2.shape


# ### Data Cleaning: Handle null values.

# In[67]:


df2.isna().sum()


# In[68]:


df2.shape


# In[69]:


df3=df2.dropna()


# In[70]:


df3.isna().sum()


# In[71]:


df3.shape


# ### Feature Engineering

# Add new feature(integer) for bhk (Bedrooms Hall Kitchen)

# In[72]:


df3['bhk']=df3['size'].apply(lambda x: int(x.split(' ')[0]))
df3.bhk.unique()


# In[73]:


df3['bhk']


# Explore total_sqft feature

# In[74]:


def is_float(x):
    try:
        float(x)
    except:
        return False
    return True
        


# In[ ]:





# In[75]:


df3[~df3['total_sqft'].apply(is_float)].head(10)


# In[76]:


def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None 


# In[77]:


df4 = df3.copy()
df4.total_sqft = df4.total_sqft.apply(convert_sqft_to_num)
df4 = df4[df4.total_sqft.notnull()]
df4.head(2)


# In[78]:


df4.loc[30]


# ### Feature Engineering

# Add new feachure called price per square feet

# In[79]:


df5=df4.copy()


# In[80]:


df5['price_per_sqft']=df5['price']*100000/df5['total_sqft']
df5.head()


# In[81]:


df5_stats=df5['price_per_sqft'].describe()
df5_stats


# In[82]:


df5.to_csv("bhp.csv",index=False)


# Examine locations which is a categorical variable. We need to apply dimensionality reduction technique here to reduce number of locations

# In[83]:


df5.location = df5.location.apply(lambda x: x.strip())
location_stats = df5['location'].value_counts(ascending=False)
location_stats


# In[84]:


location_stats.values.sum()


# In[85]:


len(location_stats[location_stats>10])


# In[86]:


len(location_stats)


# In[87]:


len(location_stats[location_stats<10])


# ### Dimensionality Reduction

# Any location having less than 10 data points should be tagged as "other" location. This way number of categories can be reduced by huge amount. Later on when we do one hot encoding, it will help us with having fewer dummy columns

# In[88]:


location_stats_less_then_10=location_stats[location_stats<=10]
location_stats_less_then_10


# In[89]:


location_stats_grater_then_10=location_stats[location_stats>=10]
location_stats_grater_then_10


# In[90]:


len(df5.location.unique())


# In[91]:


df5.location=df5.location.apply(lambda x: 'other' if x in location_stats_less_then_10 else x)
len(df5.location.unique())


# In[92]:


df5.head()


# ### Outlier Removal Using Business Logic

# As a data scientist when you have a conversation with your business manager (who has expertise in real estate), he will tell you that normally square ft per bedroom is 300 (i.e. 2 bhk apartment is minimum 600 sqft. If you have for example 400 sqft apartment with 2 bhk than that seems suspicious and can be removed as an outlier. We will remove such outliers by keeping our minimum thresold per bhk to be 300 sqft

# In[93]:


df5[df5.total_sqft/df5.bhk<300].head()


# Check above data points. We have 6 bhk apartment with 1020 sqft. Another one is 8 bhk and total sqft is 600. These are clear data errors that can be removed safely

# In[94]:


df5.shape


# In[95]:


df6=df5[~(df5.total_sqft/df5.bhk<300)]
df6.shape


# In[96]:


df6.head()


# ## Outlier Removal Using Standard Deviation and Mean

# In[97]:


df6.price_per_sqft.describe()


# In[98]:


def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df7 = remove_pps_outliers(df6)
df7.shape


# ##### Let's check if for a given location how does the 2 BHK and 3 BHK property prices look like

# In[99]:


def plot_scatter_chart(df,location):
    bhk2 = df[(df.location==location) & (df.bhk==2)]
    bhk3 = df[(df.location==location) & (df.bhk==3)]
    matplotlib.rcParams['figure.figsize'] = (20,12)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+', color='green',label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()
    


# In[100]:


plot_scatter_chart(df7,"Rajaji Nagar")


# In[101]:


plot_scatter_chart(df7,"Hebbal")


# In[102]:


def remove_bhk_outlier(df):
    exclude_indices=np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats={}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk]={
                'mean': np.mean(bhk_df.price_per_sqft),
                'std':np.std(bhk_df.price_per_sqft),
                'count':bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats=bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices=np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices, axis='index')
df8 = remove_bhk_outlier(df7)
df8.shape


# Plot same scatter chart again to visualize price_per_sqr for 2 BHK and 3 BHK properties

# In[103]:


plot_scatter_chart(df8, 'Rajaji Nagar')


# In[104]:


plot_scatter_chart(df8, 'Hebbal')


# In[105]:


matplotlib.rcParams['figure.figsize']=(30,16)
plt.hist(df8.price_per_sqft, rwidth=0.9)
plt.xlabel("price per square feet")
plt.ylabel("count" )


# ## Outlier Removal Using Bathrooms Feature

# In[106]:


df8.bath.unique()


# In[107]:


df8[df8.bath>10]


# In[108]:


plt.hist(df8.bath, rwidth=0.8)
plt.xlabel('number of bathroom')
plt.ylabel('count')


# In[109]:


df8[df8.bath>df8.bhk+2]


# In[110]:


df9=df8[df8.bath<df8.bhk+2]
df9.shape


# In[111]:


df9.head()


# In[112]:


df10=df9.drop(['size','price_per_sqft'],axis='columns')
df10.head(3)
         


# ### Use One Hot Encoding For Location

# In[113]:


dummies= pd.get_dummies(df10.location)
dummies.head(3)


# In[114]:


df11=pd.concat([df10,dummies.drop('other',axis='columns')],axis='columns')
df11.head()


# In[115]:


df12 = df11.drop('location',axis='columns')
df12.head(2)


# ## Build a Model Now.

# In[116]:


df12.shape


# In[117]:


X = df12.drop(['price'],axis='columns')
y = df12.price


# In[118]:


X.shape


# In[119]:


y.shape


# In[120]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=4000)



# ### Use K Fold cross validation to measure accuracy of our LinearRegression model

# In[124]:


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
cv= ShuffleSplit(n_splits=5, test_size=0.2,random_state=500)


# In[128]:


#cross_val_score(LinearRegression(), X, y, cv=cv)
cross_val_score(LinearRegression(), X, y, cv=cv)


# In[126]:


from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor  # Corrected import statement
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score


# In[141]:


models={
    'lr_clf':LinearRegression(),
    'lss':Lasso(),
    'rg':Ridge(),
    'Knr':KNeighborsRegressor(),
    'dtr':DecisionTreeRegressor()
}

for name, mod in models.items():
    mod.fit(X_train,y_train)
    y_pred=mod.predict(X_test)
    
    print(f"{name} MSE : {mean_squared_error(y_test,y_pred)} Score {r2_score(y_test,y_pred)}")


# Based on above results we can say that LinearRegression gives the best score and minimum error. Hence we will use that.
# 

# In[153]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression


# In[154]:


lr = LinearRegression()
lr.fit(X_train,y_train)
lr.score(X_test,y_test)


# # Test the model for few properties

# In[155]:


def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr.predict([x])[0]


# In[156]:


predict_price('1st Phase JP Nagar',1000, 2, 2)


# In[157]:


predict_price('1st Phase JP Nagar',1000, 3, 3)


# In[158]:


predict_price('Indira Nagar',1000, 2, 2)


# In[159]:


predict_price('Indira Nagar',1000, 3, 3)


# ### Pickle file

# In[160]:


import pickle
with open('BHPM1.pickle','wb') as f:
    pickle.dump(lr,f)


# ### Export location and column information to a file that will be useful later on in our prediction application

# In[161]:


import json
columns={
    'data_columns' : [col.lower() for col in X.columns]
}
with open('columns.json','w') as f:
    f.write(json.dumps(columns))


# In[ ]:




