#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


# In[2]:


data = pd.read_csv('measures_v2.csv')
data = data.drop(['profile_id'],axis = 1)


# In[3]:


import seaborn as sns
sns.pairplot(data)


# In[4]:


plt.figure(figsize = (15,15))
sns.heatmap(data.corr(), annot = True)

Out of all 13 the motor_speed (rpm), pm(permanent magnet temp in celcius) & torque is the resulting quantities.
profile_id is the measurment session.1.   coolant, stator_winding, stator_tooth, stator_yoke, ambient & pm are tempearture related quantities with unit in celcius.
2.   u_q, u_d, i_q, i_d are voltage and current quantities.Preparing dataset for temperature related quantities
# In[5]:


data_temp = data.drop(['u_q','u_d','i_q','i_d','torque'],axis = 1)
pm = data_temp.drop(['pm'],axis = 1)
data_temp = pd.concat([pm,data_temp['pm']],axis = 1)
data_temp


# In[6]:


data_temp.describe().transpose()

Finding Spearman Correlation
# In[7]:


para = [data_temp['coolant'], data_temp['stator_winding'], data_temp['stator_tooth'], data_temp['motor_speed'], data_temp['stator_yoke'], data_temp['ambient']]
name = data_temp.columns
for i in range(6) :
    from scipy.stats import spearmanr
    coef, p = spearmanr(data_temp['pm'], para[i])
    print('spearman correlation coefficient for pm and {} is: {}'.format(name[i], coef))
    alpha = 0.05
    if p > alpha:
        print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
    else:
        print('Samples are correlated (reject H0) p=%.3f' % p)

## Conclusion
The dependancy of permanent magnet temperature is maximum on stator_tooth and stator_winding temp that is 0.839 spearman corelation coeffiecient and minimum on ambient temp that is 0.539 spearman corelation coeffiecientPlot for univariate analsysis
# In[8]:


for i in range(6):
    data_temp[[name[i], 'pm']].plot(figsize = (20,12))

Modeling and Preprocessing the data. The data is split in 2 frame (x,y) and (xt,yt). Whereas x,y is data in range of 0 to 1200000 further it is being used for modeling the algo. xt, yt consist of remaining datapoints, which act as compltely unseen data for prediction and testing.
# In[9]:


x = data_temp.iloc[:1200000,:-1]
y = data_temp.iloc[:1200000,-1]


# In[10]:


xt = data_temp.iloc[1200001:,:-1]
yt = data_temp.iloc[1200001:,-1]


# In[11]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3)

Linear Regression
# In[12]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg = reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
linreg_r2 = r2_score(y_pred,y_test)


# In[13]:


train_score_reg = reg.score(x_train, y_train)
test_score_reg = reg.score(x_test, y_test)
unseen_score_reg = reg.score(xt,yt)

print("The train score for reg model is {}".format(train_score_reg))
print("The test score for reg model is {}".format(test_score_reg))
print('The unseen data score for reg model is {}'.format(unseen_score_reg))


# In[40]:


yt_pred = reg.predict(xt)
linreg_r2_us = r2_score(yt_pred,yt)
linreg_r2_us

ridge and lasso
# In[15]:


from sklearn.linear_model import Ridge, RidgeCV, Lasso
ridgeReg = Ridge(alpha=200)
ridgeReg.fit(x_train,y_train)

train_score_ridge = ridgeReg.score(x_train, y_train)
test_score_ridge = ridgeReg.score(x_test, y_test)

print("\nRidge Model............................................\n")
print("The train score for ridge model is {}".format(train_score_ridge))
print("The test score for ridge model is {}".format(test_score_ridge))


# In[41]:


y_pred = ridgeReg.predict(x_test)
rid_r2 = r2_score(y_pred,y_test)
rid_r2


# In[42]:


yt_pred = ridgeReg.predict(xt)
rid_r2_us = r2_score(yt_pred,yt)
rid_r2_us


# In[18]:


lasso = Lasso(alpha = 10)
lasso.fit(x_train,y_train)
train_score_ls =lasso.score(x_train,y_train)
test_score_ls =lasso.score(x_test,y_test)

print("The train score for ls model is {}".format(train_score_ls))
print("The test score for ls model is {}".format(test_score_ls))


# In[43]:


y_pred = lasso.predict(x_test)
lasso_r2 = r2_score(y_pred,y_test)
lasso_r2


# In[44]:


yt_pred = lasso.predict(xt)
lasso_r2_us = r2_score(yt_pred,yt)
lasso_r2_us

XGBR
# In[22]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from xgboost import XGBRegressor
from numpy import absolute

XGBR = XGBRegressor(n_estimators=50, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
score = cross_val_score(XGBR, x, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
scores = absolute(score)
print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()) )


# In[23]:


XGBR.fit(x_train,y_train)


# In[45]:


y_pred = XGBR.predict (x_test)
xgbr_r2 = r2_score(y_pred, y_test)
xgbr_r2


# In[46]:


yt_pred = XGBR.predict(xt)
xgbr_r2_us = r2_score(yt_pred, yt)
xgbr_r2_us

Support Vector Machine
# In[26]:


from sklearn.svm import SVR
import numpy as np
svr = SVR()


# In[27]:


xs = data_temp.iloc[:150000,:-1]
ys = data_temp.iloc[:150000,-1]


# In[28]:


xs_train, xs_test, ys_train, ys_test = train_test_split(xs,ys, test_size = 0.2, random_state = 42)


# In[29]:


ys_train = np.ravel(ys_train, order = 'C')
svr.fit(xs_train, ys_train)


# In[47]:


ys_pred = svr.predict(xs_test)
svm_r2 = r2_score(ys_pred, ys_test)
svm_r2


# In[32]:


xst = data_temp.iloc[150000:205000,:-1]
yst = data_temp.iloc[150000:205000,-1]
yst = np.ravel(yst)


# In[48]:


yst_pred = svr.predict(xst)
svm_r2_us = r2_score(yst_pred,yst)
svm_r2_us

Random Forest Regressor
# In[34]:


from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators = 150, random_state = 42)


# In[35]:


y_train = np.ravel(y_train)
rfr.fit(x_train,y_train)
y_pred = rfr.predict(x_test)


# In[49]:


rfr_r2 = r2_score(y_pred, y_test)
rfr_r2


# In[37]:


xt = data_temp.iloc[1200000:,:-1]
yt = data_temp.iloc[1200000:,-1]


# In[51]:


yt_pred = rfr.predict(xt)
rfr_r2_us = r2_score(yt_pred, yt)
rfr_r2_us


# In[72]:


result = [['Linear regression', linreg_r2, linreg_r2_us], ['Ridge regression', rid_r2, rid_r2_us], ['Lasso regression' , lasso_r2, lasso_r2_us],['XGB regression',xgbr_r2, xgbr_r2_us],['SVM regression', svm_r2,svm_r2_us],['Random Forest',rfr_r2,rfr_r2_us]]
for i in range(len(result)):
    print('The r2_score of {} is : {} and of the unseen data is : {}'.format(result[i][0], result[i][1], result[i][2]))
    print('........................................................................ the deviation between r2_score is {}'.format(result[i][1] - result[i][2]))

The mininmum deviation is in SVM but considerable result can be of linear regression.