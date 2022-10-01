#!/usr/bin/env python
# coding: utf-8

# ## Loading The Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Importing The Datasets

# In[2]:


df_train = pd.read_csv("train.csv", low_memory = False)
df_test = pd.read_csv("test.csv", low_memory = False)
df_stores = pd.read_csv("stores.csv", low_memory = False)
df_features = pd.read_csv("features.csv", low_memory = False)
df_submission = pd.read_csv("sampleSubmission.csv", low_memory = False)


# ### Assessing the 'train.csv' dataset

# In[3]:


df_train.head(10)


# In[4]:


df_train.shape


# In[5]:


df_train.info()


# There are 421570 rows with 5 features in the "train.csv" dataset.

# - Store: the store number
# - Dept: the department number
# - Date: the week
# - Weekly_Sales:  sales for the given department in the given store
# - IsHoliday: whether the week is a special holiday week

# ### Checking for missing values

# In[6]:


sns.heatmap(df_train.isnull())


# In[7]:


df_train.isna().sum()


# Looking at the count it is clear that there are no missing values

# ### Checking for duplicates

# In[8]:


df_train.duplicated(subset=['Store', 'Dept', 'Date']).value_counts()


# There are no duplicate values in the 'train.csv' dataset based on the columns 'Store', 'Dept', & 'Date'.

# ### Assessing the 'stores.csv' dataset

# In[9]:


df_stores.head(10)


# In[10]:


df_stores.shape


# In[11]:


df_stores.info()


# There are 45 rows with 3 features in the "stores.csv" dataset, where only 1 feature is categorical (Type) while the rest two features are numerical.

# - Store: the store number
# - Type: the store type A or B
# - Size: the size of the store

# ### Checking for missing values

# In[12]:


sns.heatmap(df_stores.isnull())


# In[13]:


df_stores.isna().sum()


# Looking at the count it is clear that there are no missing values

# ### Assessing the 'features.csv' dataset

# In[14]:


df_features.head(10)


# In[15]:


df_features.shape


# In[16]:


df_features.info()


# There are 8190 rows with 12 features in the "features.csv" dataset.

# - Store: the store number
# - Date: the week
# - Temperature: average temperature in the region
# - Fuel_Price: cost of fuel in the region
# - MarkDown 1-5: anonymized data related to promotional markdowns that Walmart is running. MarkDown data is only available after Nov 2011, and is not available for all stores all the time. Any missing value is marked with an NA.
# - CPI: the consumer price index
# - Unemployment: the unemployment rate
# - IsHoliday: whether the week is a special holiday week

# ### Checking for missing values

# In[17]:


sns.heatmap(df_features.isnull())


# In[18]:


df_features.isna().sum()


# The features MarkDown1, MarkDown2, MarkDown3, MarkDown4, & MarkDown5 have a lot a missing and even replacing the NaN values with either the mean or 0 could have an adverse effect on the accuracy and the efficiency of the model. So, the best way of dealing with such problem is to drop these features altogether to achieve a higher accuracy.
# 
# The features ‘CPI’ & ‘Unemployment’ have a very few numbers of missing values. Most of the values in these two features are very similar. Therefore, the mean of all the values can be taken to replace the NaN values in these features.

# In[19]:


#Dropping the columns with many NaN values
df_features.drop(['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5'], axis=1, inplace=True)


# In[20]:


#Replacing the NaN values with mean
df_features['CPI'].fillna(value=df_features['CPI'].mean(), inplace=True)
df_features['Unemployment'].fillna(value=df_features['Unemployment'].mean(), inplace=True)


# ### Merging the datasets

# In[21]:


df_merge = df_train.merge(df_stores, how='left').merge(df_features, how='left')


# In[22]:


df_merge.head(10)


# In[23]:


df_merge.shape


# ### Checking for missing values

# In[24]:


sns.heatmap(df_merge.isnull())


# In[25]:


df_merge.isna().sum()


# In[26]:


#Checking for duplicates after merging
df_merge.duplicated(subset=['Store', 'Date', 'Dept']).value_counts()


# ### Splitting the 'Date' column into 'Year', 'Month' & 'Day'

# In[27]:


df_merge['Date'] = pd.to_datetime(df_merge['Date'])
df_merge['Year'] = df_merge.Date.dt.year
df_merge['Month'] = df_merge.Date.dt.month
#df_merge['Month'] = df_merge['Month'].apply(lambda x: calendar.month_abbr[x])
df_merge['Day'] = df_merge.Date.dt.day
#df_merge['WeekOfYear'] = df_merge.Date.dt.isocalendar().week


# In[28]:


df_merge.head(10)


# In[29]:


df_merge['Month'] = df_merge['Month'].astype(str).str.zfill(2)
df_merge['Months'] = df_merge['Year'].astype(str) + " - " + df_merge['Month'].astype(str)
df_merge.head(10)


# ### Correlation Heatmap

# In[30]:


def heatmap_all(combined):
    plt.figure(figsize=(12, 8))
    heatmap = sns.heatmap(combined.corr(), annot=True)
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':20}, pad=12);
heatmap_all(df_merge)


# ### Data Visualization

# In[31]:


#Year-Sales
Sales_Year = df_merge.groupby(['Year'])[['Weekly_Sales']].sum()
Sales_Year


# In[32]:


Sales_Year = Sales_Year.reset_index(level=0)


# In[33]:


sns.set(rc = {'figure.figsize':(6,4)}, font_scale = 1.2)
sns.barplot(x='Year', y='Weekly_Sales', data=Sales_Year, palette="mako")
plt.xlabel("Years")
plt.ylabel("Number of Sales")


# The year 2011 had the maximum number of sales, while 2012 had the minimum.

# In[34]:


#Months-Sales
Sales_YearMonth = df_merge.groupby(['Months'])[['Weekly_Sales']].sum()
Sales_YearMonth.sort_values('Months')


# In[35]:


sns.set(rc = {'figure.figsize':(20,8)}, font_scale = 1.5)
sns.lineplot(data=Sales_YearMonth, x="Months", y="Weekly_Sales").set(title="No. of Sales from 2010-12")
plt.xticks(rotation=60)
plt.xlabel("Months (2010-12)")
plt.ylabel("Number of Sales")
plt.show()


# The graph depicts 3 phases. The 1st phase starts from Feb-2010 till Dec-2010 and shows some kind of linear trend and additive seasonality in the graph. But right after that 1st phase, from the start of the 2nd phase which is from Jan-2011 to Dec-2011, there is a sudden dip in the sales from Dec-2010 to Jan-2011.
# 
# The 2st phase starts from Jan-2011 to Dec-2011 and also shows some kind of linear trend and additive seasonality. The number of sales decreases from Dec-2011 to Jan-2012 right after the 2nd phase, which is quite similar what we see at the end of the 1st phase. The 3rd phase which starts from Jan-2012 and ends in Oct-2012 has no apparent trend but does show additive seasonality.
# 
# The graph shows a strong seasonality within each year and also shows some strong cyclic behavior with a period of about 10-11 months.

# In[36]:


#Store_Types-Monthly_Sales
Type_YearMonth = df_merge.groupby(['Months', 'Type'])[['Weekly_Sales']].sum()
Type_YearMonth.sort_values('Months')


# In[37]:


sns.set(rc = {'figure.figsize':(20,8)}, font_scale = 1.5)
sns.lineplot(data=Type_YearMonth, x="Months", y="Weekly_Sales", hue="Type")
plt.xticks(rotation=60)
plt.xlabel("Number of Sales")
plt.ylabel("Months (2010-12)")
plt.show()


# Store type ‘A’ had the maximum number of sales while store type ‘C’ had the minimum.

# In[38]:


#Store_Type Count
sns.set(rc = {'figure.figsize':(7,4)}, font_scale = 1.2)
sns.countplot(x="Type", data=df_merge)
plt.xlabel("Store Types")
plt.ylabel("Store Count")


# The number of stores for the type ‘A’ were a lot more as compared to the other two types.

# In[39]:


#Store_Type-Size
sns.set(rc = {'figure.figsize':(7,4)}, font_scale = 1.2)
sns.barplot(x='Type', y='Size', data=df_merge, palette="mako")
plt.xlabel("Store Types")
plt.ylabel("Store Sizes")


# Compared to other two store types, the store sizes for the type ‘A’ were larger.

# In[40]:


#Temperature-Months
Temp_YearMonth = df_merge.groupby(['Months'])[['Temperature']].mean()
Temp_YearMonth.sort_values('Months')


# In[41]:


sns.set(rc = {'figure.figsize':(20,8)}, font_scale = 1.5)
sns.lineplot(data=Temp_YearMonth, x="Months", y="Temperature").set(title="Average Temperature from 2010-12")
plt.xticks(rotation=60)
plt.show()


# In[42]:


#Holiday-Sales
IsHol_YearMonth = df_merge.groupby(['Months', 'IsHoliday'])[['Weekly_Sales']].sum()
IsHol_YearMonth.sort_values('Months')


# In[43]:


sns.set(rc = {'figure.figsize':(20,8)}, font_scale = 1.5)
sns.lineplot(data=IsHol_YearMonth, x="Months", y="Weekly_Sales", hue="IsHoliday")
plt.xticks(rotation=60)
plt.xlabel("Months (2010-12)")
plt.ylabel("Number of Sales")
plt.show()


# Even though the number of monthly sales were not zero but were quite low whenever there was a holiday, while the sales numbers were definitely high there weren’t any holidays.

# In[44]:


#StoreType-Holiday
sns.set(rc = {'figure.figsize':(12,6)}, font_scale = 1.4)
sns.countplot(x='IsHoliday',hue='Type', data=df_merge, palette="mako")
plt.xlabel("Holiday")
plt.ylabel("Sales")


# The bar graph shows that sales for store type ‘A’ were higher as compared to the other two types no matter whether there was a holiday or not, while store type ‘C’ had the minimum number of sales and were close to zero for store type ‘C’ when there was a holiday.

# In[45]:


#Unemployment_Rate-Months
U_YearMonth = df_merge.groupby(['Months'])[['Unemployment']].mean()
U_YearMonth.sort_values('Months')


# In[46]:


sns.set(rc = {'figure.figsize':(20,8)}, font_scale = 1.5)
sns.lineplot(data=U_YearMonth, x="Months", y="Unemployment").set(title="Unempyoment Rate from 2010-12")
plt.xticks(rotation=60)
plt.xlabel("Months (2010-12)")
plt.ylabel("Unemployment Rate")
plt.show()


# There appears to be a strong downward decreasing trend with strong seasonality in the unemployment rates during the three-year period from Feb-2010 to Oct-2012, but there is no presence of any kind of cyclic behavior.

# In[47]:


#Fuel_Prices-Months
FP_YearMonth = df_merge.groupby(['Months'])[['Fuel_Price']].mean()
FP_YearMonth.sort_values('Months')


# In[48]:


sns.lineplot(data=FP_YearMonth, x="Months", y="Fuel_Price").set(title="Fuel Prices from 2010-12")
plt.xticks(rotation=60)
plt.xlabel("Months (2010-12)")
plt.ylabel("Fuel Prices")
plt.show()


# There was a sudden increase in the fuel prices from Sep-2010 where the prices were as low as 2.7 dollars to May-2011 where they reached the highest with prices reaching close to 4 dollars. The prices seem to decrease a bit for a period of 8-10 months but then again when back up to 4 dollars in Apr-2012.

# ### Encoding the Categorical Variables

# In[49]:


#Creating a dummy dataset and dropping 'Date' column
df_model = df_merge.copy()
df_model.drop(['Date'], axis=1, inplace=True)


# In[50]:


df_model.head(10)


# ### One-Hot Encoding

# In[51]:


cat_cols = df_model[['IsHoliday', 'Type']] 


# In[52]:


from sklearn.preprocessing import OneHotEncoder
#One-hot-encoding the categorical columns.
encoder = OneHotEncoder(handle_unknown='ignore')
#Converting it to dataframe
df_encoder = pd.DataFrame(encoder.fit_transform(cat_cols).toarray())
df_final = df_model.join(df_encoder)
df_final.head()


# In[53]:


#Dropping the columns that have already been One-Hot-Encoded
df_final.drop(['IsHoliday', 'Type', 'Months'], axis=1, inplace=True)


# In[54]:


col = df_final.pop('Weekly_Sales')
df_final.insert(0, 'Weekly_Sales', col)
df_final.head()


# In[55]:


X = df_final.iloc[:, 1:].values
y = df_final.iloc[:, 0].values


# In[56]:


X


# In[57]:


y


# ### Splitting the dataset

# In[58]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 1)


# In[59]:


X_train


# In[60]:


X_test


# In[61]:


y_train


# In[62]:


y_test


# ### Random Forest Regression Model

# In[63]:


#Training the Random Forest Regression model on the whole dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)


# ### Predicting the Test set results

# In[64]:


y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
pred = np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)
pred[:10, :]


# In[65]:


pred_plot = pd.DataFrame(pred, columns=['Predicted', 'Actual'])
pred_plot.head(15)


# ### Evaluating the model performance

# In[66]:


from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


# In[67]:


from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, y_pred)


# In[68]:


from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred)


# ### LightGBM Regression Model

# In[69]:


#Training the LightGBM model on the whole dataset
import lightgbm as lgb
model = lgb.LGBMRegressor()
model.fit(X_train, y_train)
expected_y  = y_test
predicted_y = model.predict(X_test)


# ### Evaluating the model performance

# In[70]:


#R-Squared
from sklearn import metrics
r2_score(expected_y, predicted_y)


# In[71]:


#Mean Absolute Error
from sklearn.metrics import mean_absolute_error
mean_absolute_error(expected_y, predicted_y)


# In[72]:


#Mean Squared Error
from sklearn.metrics import mean_squared_error
mean_squared_error(expected_y, predicted_y)


# ### Assessing the 'test.csv' dataset

# In[73]:


df_test.head(10)


# In[74]:


df_test.shape


# In[75]:


df_test.info()


# There are 115064 rows with 4 features in the 'test.csv' dataset.

# - Store: the store number
# - Dept: the department number
# - Date: the week
# - IsHoliday: whether the week is a special holiday week

# ### Merging the 'test.csv' dataset with 'stores.csv' & 'features.csv' dataset

# In[76]:


df_result = df_test.merge(df_stores, how='left').merge(df_features, how='left')


# In[77]:


df_result.shape


# ### Checking for missing values

# In[78]:


sns.set(rc = {'figure.figsize':(6,4)}, font_scale = 1)
sns.heatmap(df_result.isnull())


# In[79]:


df_result.isna().sum()


# ### Splitting the 'Date' column into 'Year', 'Month' & 'Day'

# In[80]:


df_result['Date'] = pd.to_datetime(df_result['Date'])
df_result['Year'] = df_result.Date.dt.year
df_result['Month'] = df_result.Date.dt.month
df_result['Day'] = df_result.Date.dt.day


# In[81]:


df_result.head(10)


# ### Encoding the Categorical Columns for 'test.csv'

# In[82]:


# Dropping the 'Date' column
df_result.drop('Date', axis=1, inplace=True)


# ### One-Hot Encoding

# In[83]:


res_cols = df_result[['IsHoliday', 'Type']] 


# In[84]:


from sklearn.preprocessing import OneHotEncoder
#One-hot-encoding the categorical columns.
encoder = OneHotEncoder(handle_unknown='ignore')
#Converting it to dataframe
df_encoder = pd.DataFrame(encoder.fit_transform(res_cols).toarray())
df_res_final = df_result.join(df_encoder)
df_res_final.head()


# In[85]:


#Dropping the columns that have already been One-Hot-Encoded
df_res_final.drop(['IsHoliday', 'Type'], axis=1, inplace=True)


# ### Predicted Sales

# In[86]:


#Predicted Sales for the test set
predicted_test = regressor.predict(df_res_final)


# In[87]:


predicted_test


# In[88]:


pred_sales = pd.DataFrame(predicted_test, columns=['Predicted_Sales'])


# In[89]:


df_graph = pred_sales.copy()


# In[90]:


df_graph.head(10)


# ### Visualization of the Predicted Values

# In[91]:


df_graph["Month"] = df_result["Month"]


# In[92]:


df_graph.head(10)


# In[93]:


Predict_Month = df_graph.groupby(['Month'])[['Predicted_Sales']].sum()
Predict_Month


# In[94]:


Predict_Month['Predicted_Sales'] = Predict_Month['Predicted_Sales'].astype('int')
Predict_Month


# In[95]:


start_date = '2010-11-01'
end_date = '2011-07-31'


# In[96]:


mask_11 = (df_merge['Date'] >= start_date) & (df_merge['Date'] <= end_date)


# In[97]:


df_11 = df_merge.loc[mask_11]
df_11.head(10)


# In[98]:


df_11['Date'] = pd.to_datetime(df_11['Date'])
df_11['Year'] = df_11.Date.dt.year
df_11['Month'] = df_11.Date.dt.month
df_11['Day'] = df_11.Date.dt.day


# In[99]:


df_11.head(10)


# In[100]:


start_date = '2011-11-01'
end_date = '2012-07-31'


# In[101]:


mask_12 = (df_merge['Date'] >= start_date) & (df_merge['Date'] <= end_date)


# In[102]:


df_12 = df_merge.loc[mask_12]
df_12.head(10)


# In[103]:


df_12['Date'] = pd.to_datetime(df_12['Date'])
df_12['Year'] = df_12.Date.dt.year
df_12['Month'] = df_12.Date.dt.month
df_12['Day'] = df_12.Date.dt.day


# In[104]:


df_12.head(10)


# In[105]:


#Creating a table with the sales from every year
compare = df_11.groupby(['Month'])[['Weekly_Sales']].sum()
compare['2010-11'] = compare['Weekly_Sales']
compare.head(10)


# In[106]:


compare.drop('Weekly_Sales', axis=1, inplace=True)


# In[107]:


compare['2010-11'] = compare['2010-11'].astype('int')
compare


# In[108]:


compare['2011-12'] = df_12.groupby(['Month'])[['Weekly_Sales']].sum()
compare['2011-12'] = compare['2011-12'].astype('int')
compare


# In[109]:


compare['2012-13'] = Predict_Month['Predicted_Sales']
compare


# In[110]:


#Plot for the comparison of sales
sns.set(rc = {'figure.figsize':(30,15)}, font_scale = 2.5)
sns.lineplot(data=compare, x="Month", y="2010-11", color='r', label='Sales in 2011').set(title="Comparison of Predicted Sales & Previous Years' Sales")
sns.lineplot(data=compare, x="Month", y="2011-12", color='b', label='Sales in 2012')
sns.lineplot(data=compare, x="Month", y="2012-13", color='g', label='Predicted Sales in 2013')
plt.xticks(rotation=60)
plt.xlabel("Months")
plt.ylabel("Number of Sales")
plt.show()


# The above graph shows the comparison of the predicted sales values in the year 2013 and sales numbers in the years 2011 & 2012. The predicted sales values were very similar to the previous years’ sales values that means if there would have been data about the actual sales values from the ‘test.csv’ dataset, it would have been very close.
