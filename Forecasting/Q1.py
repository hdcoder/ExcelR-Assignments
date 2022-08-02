import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.formula.api as smf 
from pylab import rcParams
from math import sqrt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from typing_extensions import final

Airlines_Data = pd.read_excel("Airlines+Data.xlsx")
print(Airlines_Data)
print()

Airlines_Data['Passengers'].plot(figsize=(16,7))
Airlines_Data = Airlines_Data.set_index('Month')
Airlines_Data['SMA_2'] = Airlines_Data['Passengers'].rolling(2, min_periods=1).mean()
Airlines_Data['SMA_4'] = Airlines_Data['Passengers'].rolling(4, min_periods=1).mean()
Airlines_Data['SMA_6'] = Airlines_Data['Passengers'].rolling(6, min_periods=1).mean()
Colors = ['green', 'red', 'orange','blue']
Airlines_Data.plot(color=Colors, linewidth=3, figsize=(12,6))

plt.figure()
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(labels =['No. of Passengers', '2-years SMA', '4-years SMA','6-years SMA'], fontsize=14)
plt.title('The yearly Passengers travelling', fontsize=20)
plt.xlabel('Year', fontsize=16)
plt.ylabel('Passengers', fontsize=16)

Airlines_Data['CMA'] = Airlines_Data['Passengers'].expanding().mean()
Airlines_Data[['Passengers', 'CMA']].plot( linewidth=3, figsize=(12,6))

plt.figure()
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(labels =['Passengers', 'CMA'], fontsize=14)
plt.title('The yearly Passengers travelling', fontsize=20)
plt.xlabel('Year', fontsize=16)
plt.ylabel('Population', fontsize=16)

Airlines_Data['Ema_0.1'] = Airlines_Data['Passengers'].ewm(alpha=0.1,adjust=False).mean()
Airlines_Data['Ema_0.3'] = Airlines_Data['Passengers'].ewm(alpha=0.3,adjust=False).mean()
Colors = ['#B4EEB4', '#00BFFF', '#FF3030']
Airlines_Data[['Passengers', 'Ema_0.1', 'Ema_0.3']].plot(color=Colors, linewidth=3, figsize=(12,6), alpha=0.8)

plt.figure()
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(labels=['No. of passengers', 'EMA - alpha=0.1', 'EMA - alpha=0.3'], fontsize=14)
plt.title('The yearly Passengers travelling.', fontsize=20)
plt.xlabel('Year', fontsize=16)
plt.ylabel('Passengers', fontsize=16)

plot_acf(Airlines_Data['Passengers'])
plt.show()
plot_acf(Airlines_Data['Passengers'],lags=30)
plt.show()

TS_mul = seasonal_decompose(Airlines_Data.Passengers,model="multiplicative")
fig = TS_mul.plot()
plt.show()

X = Airlines_Data['Passengers']
Size = int(len(X)*0.75)
print(Size)
print()

Train , Test = X.iloc[0:Size],X.iloc[Size:len(X)]
Model = ARIMA(Train, order=(5,1,0)).fit(disp=0)
print(Model.summary())
print()

History = [X for X in Train]
print(History[-1])
print()

History = [X for X in Train]
Predictions = list()

for i in range(len(Test)):
    Yhat = History[-1]
    Predictions.append(Yhat)
    Obs = Test[i]
    History.append(Obs)
    print('>Predicted=%.3f, Expected=%.3f' % (Yhat, Obs))
print()

RMSE = sqrt(mean_squared_error(Test, Predictions))
print('RMSE: %.3f' % RMSE)
print()

Final_df = pd.read_excel("Airlines+Data.xlsx")
Final_df['Date'] = pd.to_datetime(Final_df.Month,format="%b-%y")
Final_df['month'] = Final_df.Date.dt.strftime("%b") #month extraction
Final_df['year'] = Final_df.Date.dt.strftime("%y")

plt.figure(figsize=(10,7))
plt.subplot(211)
sns.boxplot(x="month",y="Passengers",data=Final_df,palette='nipy_spectral')
plt.subplot(212)
sns.boxplot(x="year",y="Passengers",data=Final_df,palette='plasma')

Final_df = pd.get_dummies(Final_df, columns = ['month'])

t = np.arange(1,97)
Final_df['t'] = t
Final_df['t_square'] = (t *t)
Log_Passengers = np.log(Final_df['Passengers'])
Final_df['log_Passengers'] = Log_Passengers
print(Final_df)
print()

Train, Test = np.split(Final_df, [int(.75 *len(Final_df))])

Linear_model = smf.ols('Passengers~t',data =Train).fit()
Pred_linear =  pd.Series(Linear_model.predict(pd.DataFrame(Test['t'])))
Rmse_linear = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(Pred_linear))**2))
print(RMSE_linear)
print()

Exp = smf.ols('log_Passengers~t',data=Train).fit()
Pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
Rmse_Exp = np.sqrt(np.mean((np.array(Test['log_Passengers'])-np.array(np.exp(Pred_Exp)))**2))
print(Rmse_Exp)
print()

Quad = smf.ols('Passengers~t+t_square',data=Train).fit()
Pred_Quad = pd.Series(Quad.predict(Test[["t","t_square"]]))
Rmse_Quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(Pred_Quad))**2))
print(Rmse_Quad)
print()

Add_sea = smf.ols('Passengers~month_Jan+month_Feb+month_Mar+month_Apr+month_May+month_Jun+month_Jul+month_Aug+month_Sep+month_Oct+month_Nov+month_Dec',data=Train).fit()
Pred_add_sea = pd.Series(Add_sea.predict(Test[['month_Jan','month_Feb','month_Mar','month_Apr','month_May','month_Jun','month_Jul','month_Aug','month_Sep','month_Oct','month_Nov','month_Dec']]))
Rmse_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(Pred_add_sea))**2))
print(Rmse_add_sea)
print()

Add_sea_Quad = smf.ols('Passengers~t+t_square+month_Jan+month_Feb+month_Mar+month_Apr+month_May+month_Jun+month_Jul+month_Aug+month_Sep+month_Oct+month_Nov+month_Dec',data=Train).fit()
Pred_add_sea_quad = pd.Series(Add_sea_Quad.predict(Test[['t','t_square','month_Jan','month_Feb','month_Mar','month_Apr','month_May','month_Jun','month_Jul','month_Aug','month_Sep','month_Oct','month_Nov','month_Dec']]))
Rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(Pred_add_sea_quad))**2))
print(Rmse_add_sea_quad)
print()

Mul_sea = smf.ols('log_Passengers~month_Jan+month_Feb+month_Mar+month_Apr+month_May+month_Jun+month_Jul+month_Aug+month_Sep+month_Oct+month_Nov+month_Dec',data = Train).fit()
Pred_Mult_sea = pd.Series(Mul_sea.predict(Test[['month_Jan','month_Feb','month_Mar','month_Apr','month_May','month_Jun','month_Jul','month_Aug','month_Sep','month_Oct','month_Nov','month_Dec']]))
Rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['log_Passengers'])-np.array(np.exp(Pred_Mult_sea)))**2))
print(Rmse_Mult_sea)
print()

Mul_Add_sea = smf.ols('log_Passengers~t+month_Jan+month_Feb+month_Mar+month_Apr+month_May+month_Jun+month_Jul+month_Aug+month_Sep+month_Oct+month_Nov+month_Dec',data = Train).fit()
Pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test[['t','month_Jan','month_Feb','month_Mar','month_Apr','month_May','month_Jun','month_Jul','month_Aug','month_Sep','month_Oct','month_Nov','month_Dec']]))
Rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['log_Passengers'])-np.array(np.exp(Pred_Mult_add_sea)))**2))
print(Rmse_Mult_add_sea)
print()

Data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([Rmse_linear,Rmse_Exp,Rmse_Quad,Rmse_add_sea,Rmse_add_sea_quad,Rmse_Mult_sea,Rmse_Mult_add_sea])}
Table_rmse = pd.DataFrame(Data)
Table_rmse.sort_values(['RMSE_Values'])

Model_final = smf.ols('Passengers~t+t_square+month_Jan+month_Feb+month_Mar+month_Apr+month_May+month_Jun+month_Jul+month_Aug+month_Sep+month_Oct+month_Nov+month_Dec',data=Train).fit()

Pred_new  = pd.Series(Model_final.predict(Test))
print(Pred_new)
print()

Predict_data= pd.DataFrame()
Predict_data["forecasted_passengers"] = pd.Series(Pred_new)

Visualize = pd.concat([Train,Predict_data])
print(Visualize)

Visualize[['Passengers','forecasted_passengers']].reset_index(drop=True).plot(figsize=(16,8))