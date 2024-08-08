import streamlit as st
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from prophet.plot import plot_plotly
import plotly.graph_objects as go

st.title('Price Prediction for CryptoCurrency')
Crypto=("BTC","ETH","SOL")
selected_crypto=st.selectbox("Select CryptoCurrency for Prediction:",Crypto)
selected_crypto+="-USD"

@st.cache_data
def load_data(ticker):
    data_ticker=yf.Ticker(ticker)
    data=data_ticker.history(period="max")
    return data

data=load_data(selected_crypto)

st.subheader('Data Description :-')
st.write(data.describe())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Open'], name="stock_open",line=dict(width=5)))
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

st.write('For further work, we will be using Closing value of currency only.')
data=data['Close'].copy()

st.write('Here, we will be using ARIMA model for making predictions. As we know, ARIMA model works only on stationary data, so we will be checking stationarity of data :-')

rolLmean = data.rolling(365).mean()
rolLstd = data.rolling(365).std()

#Plot rolling statistics:
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data, name="Original"))
fig.add_trace(go.Scatter(x=data.index, y=rolLmean, name="Rolling Mean"))
fig.add_trace(go.Scatter(x=data.index, y=rolLstd, name="Rolling Std"))
fig.layout.update(title_text='Plotting Rolling Mean and Standard Deviation to check Stationarity', xaxis_rangeslider_visible=True)
st.plotly_chart(fig)

st.write('We can see that Mean and Standard Deviation of data is not constant, so data is not stationary. To make data stationary we will be doing log-transformation of the data and further try to plot Rolling Mean and Standard Deviation: ')

data=np.log(data)

rolLmean = data.rolling(365).mean()
rolLstd = data.rolling(365).std()
#Plot rolling statistics:
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data, name="Original"))
fig.add_trace(go.Scatter(x=data.index, y=rolLmean, name="Rolling Mean"))
fig.add_trace(go.Scatter(x=data.index, y=rolLstd, name="Rolling Std"))
fig.layout.update(title_text='Plotting Rolling Mean and Standard Deviation to check Stationarity', xaxis_rangeslider_visible=True)
st.plotly_chart(fig)

st.write('We can see that Mean and Standard deviation are somewhat constant now in comparison to previous one. So, we will proceed with using the log-transformed data for all subsequent analyses and modeling.')

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(data,period=54)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data, name="Original"))
fig.add_trace(go.Scatter(x=data.index, y=rolLmean, name="Rolling Mean"))
fig.add_trace(go.Scatter(x=data.index, y=rolLstd, name="Rolling Std"))
fig.layout.update(title_text='Plotting Rolling Mean and Standard Deviation to check Stationarity', xaxis_rangeslider_visible=True)
st.plotly_chart(fig)

from plotly.subplots import make_subplots

fig = make_subplots(
    rows=3, cols=1,subplot_titles=("Trend", "Seasonality", "Residuals"),vertical_spacing=0.165
)

fig.add_trace(
    go.Scatter(x=data.index, y=trend, name="Original"),
    row=1, col=1
)

# Rolling Mean
fig.add_trace(
    go.Scatter(x=data.index, y=seasonal, name="Rolling Mean"),
    row=2, col=1
)

# Rolling Std
fig.add_trace(
    go.Scatter(x=data.index, y=residual, name="Rolling Std"),
    row=3, col=1
)

fig.update_layout(
    title_text='Plotting Trend, Seasonality and Residuals',
    yaxis_title='Value',
    xaxis_rangeslider_visible=False
)
st.plotly_chart(fig)

st.write("We will be taking last one year data as test data. Rest will be used for training our Arima Model.")

train = data.iloc[:len(data)-365]
test = data.iloc[len(data)-365:]

st.write('Dickey-Fuller Test on train data :')

from statsmodels.tsa.stattools import adfuller
dftest=adfuller(train, autolag='AIC')
dfoutput=pd.Series(dftest[0:4], index=['Test Statistic','p-value','Lags Used','No. of Obs'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
dfoutput = pd.DataFrame({
        'Metric': ['Test Statistic', 'p-value', 'Lags Used', 'No. of Obs'] + [f'Critical Value ({key})' for key in dftest[4].keys()],
        'Value': list(dftest[0:4]) + list(dftest[4].values())
    })
st.write(dfoutput)

if(selected_crypto=='BTC-USD'):
  p=0
  d=1
  q=0
elif selected_crypto=='ETH-USD':
  p=0
  d=1
  q=2
else:
  p=0
  d=1
  q=0

st.write(f'Now we will be training ARIMA using training data. But for that we need values of p,d and q as well. To know that we used auto_arima() from pmdarima package. The values it gave to us are p={p}, d={d}, and q={q}')

from statsmodels.tsa.arima.model import ARIMA
model=ARIMA(train, order=(p,d,q))
results=model.fit()

forecast = results.get_forecast(steps=len(test))
forecast_index = test.index
forecast_mean = forecast.predicted_mean
forecast_conf_int = forecast.conf_int()

fig = go.Figure()

# Plot training data
fig.add_trace(go.Scatter(x=train.index, y=np.exp(train), mode='lines', name='Training Data', line=dict(color='blue')))

# Plot test data
fig.add_trace(go.Scatter(x=test.index, y=np.exp(test), mode='lines', name='Test Data', line=dict(color='green')))

# Plot forecasts
fig.add_trace(go.Scatter(x=test.index, y=np.exp(forecast_mean), mode='lines', name='Forecast', line=dict(color='red')))

# Plot confidence intervals
fig.add_trace(go.Scatter(
    x=np.concatenate([forecast_index, forecast_index[::-1]]),
    y=np.concatenate([np.exp(forecast_conf_int.iloc[:, 0]), np.exp(forecast_conf_int.iloc[:, 1])[::-1]]),
    fill='toself',
    fillcolor='rgba(255, 192, 203, 0.3)',
    line=dict(color='rgba(255, 255, 255, 0)'),
    name='Confidence Interval'
))

# Update layout
fig.update_layout(
    title='ARIMA Model Forecast',
    xaxis_title='Date',
    yaxis_title='Value',
    xaxis_rangeslider_visible=True
)

# Display the plot in Streamlit
st.plotly_chart(fig)

from statsmodels.tools.eval_measures import rmse
st.write("Root Mean Squared Error between actual and  predicted values: ",rmse(forecast_mean,test))
st.write("Mean Value of Test Dataset:", test.mean())

from sklearn.metrics import r2_score
r2 = r2_score(test, forecast_mean)
st.write("R-squared:", r2)
