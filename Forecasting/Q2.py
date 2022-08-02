import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
from pylab import rcParams
from math import sqrt
from sklearn.metrics import mean_squared_error
import statsmodels.formula.api as smf 
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.graphics.tsaplots as tsa_plots
from statsmodels.tsa.arima_model import ARIMA
import warnings

cc_data = pd.read_excel('C:/Users/ankit/Downloads/CocaCola_Sales_Rawdata.xlsx')
cc_data.head()

print(cc_data.Quarter[1][3:])
print(cc_data.Quarter[1][:2])

year = []
month = []
for i in cc_data['Quarter']:
    year.append('19'+i[3:])
    if i[:2] == 'Q1':
        month.append('jan')
    elif i[:2] == 'Q2':
        month.append('apr')
    elif i[:2] == 'Q3':
        month.append('jul')
    else:
        month.append('oct')

data = cc_data.copy()
data['year'] = year
data['month'] = month
data['t'] = range(1, len(data)+1)
data['t_squared'] = data['t']**2
data.head()

data

data.shape

data.info()

data['year']= pd.to_datetime(data['year'])

plt.style.use('ggplot')
plt.title("Sales Over the Years.",fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(labels ='Sales', fontsize=14)
data['Sales'].plot(figsize=(16,7),c='g');

for i in range(2,10,2):
    data["Sales"].rolling(i).mean().plot(label=str(i),figsize=(16,7))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc='best',fontsize=(14));

tsa_plots.plot_acf(data['Sales'])
tsa_plots.plot_pacf(data['Sales'], lags = 4)
plt.show()

import statsmodels.api as sm

changes = cc_data.Quarter.str.replace(r'(Q\d)_(\d+)', r'19\2-\1')
cc_data['quater'] = pd.to_datetime(changes).dt.strftime('%b-%Y')
cc_data= cc_data.drop(['Quarter'], axis=1)
cc_data.reset_index(inplace=True)
cc_data['quater'] = pd.to_datetime(cc_data['quater'])
cc_data = cc_data.set_index('quater')
cc_data.head()

# graphs to show seasonal_decompose
def seasonal_decompose (y):
    decomposition = sm.tsa.seasonal_decompose(y, model='multiplicative',extrapolate_trend='freq')
    fig = decomposition.plot()
    fig.set_size_inches(14,7)
    plt.show()

seasonal_decompose(cc_data['Sales'])

X = data['Sales'].values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
model = ARIMA(train, order=(5,1,0))
model_fit = model.fit(disp=0)

print(model_fit.summary())

model_fit.forecast(1)

history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs));

error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)

plt.plot(test)
plt.plot(predictions, color='g')
plt.show()

data.head()

data['Quarters'] = 0
data['Year'] = 0
for i in range(42):
    p = data["Quarter"][i]
    data['Quarters'][i]= p[0:2]
    data['Year'][i]= p[3:5]

# Getting dummy variables for Quarters Q1, Q2, Q3, Q4 
Quarters_Dummies = pd.DataFrame(pd.get_dummies(data['Quarters']))
data = pd.concat([data,Quarters_Dummies],axis = 1)
data['log_sales'] =  np.log(data["Sales"])

final_data = data.drop(['year','month'],axis=1)

final_data.head()

Train, Test = final_data.head(32),final_data.tail(10)

import statsmodels.formula.api as smf 

linear_model = smf.ols('Sales~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_linear))**2))
rmse_linear

Exp = smf.ols('log_sales~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['log_sales'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp

Quad = smf.ols('Sales~t+t_squared',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_squared"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_Quad))**2))
rmse_Quad

add_sea = smf.ols('Sales~Q1+Q2+Q3',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Q1','Q2','Q3']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea))**2))
rmse_add_sea

add_sea_Quad = smf.ols('Sales~t+t_squared+Q1+Q2+Q3',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['t','t_squared','Q1','Q2','Q3']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad

Mul_sea = smf.ols('log_sales~Q1+Q2+Q3',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test[['Q1','Q2','Q3']]))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['log_sales'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea

Mul_Add_sea = smf.ols('log_sales~t+Q1+Q2+Q3',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test[['t','Q1','Q2','Q3']]))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['log_sales'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse.sort_values(['RMSE_Values'])

def RMSE(org, pred):
    rmse=np.sqrt(np.mean((np.array(org)-np.array(pred))**2))
    return rmse

final_model = smf.ols('Sales~t+t_squared+Q1+Q2+Q3',data=final_data).fit()
pred_final = pd.Series(final_model.predict(final_data[['Q1','Q2','Q3','t','t_squared']]))
rmse_final_model = RMSE(final_data['Sales'], pred_final)
rmse_final_model

pred_df = pd.DataFrame({'Actual' : final_data.Sales, 'Predicted' : pred_final})

plt.figure(figsize=(14,8))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('Actaul price vs predicted price of coco cola', fontsize=20)
plt.ylabel('Sales', fontsize=16)
plt.plot(final_data['Sales'],label='Actual')
plt.plot(pred_df['Predicted'],label='Predicted')
plt.legend();