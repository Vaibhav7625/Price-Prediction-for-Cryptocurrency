# Price Prediction for Cryptocurrency 📈

Welcome to the "Price Prediction for Cryptocurrency" project! This repository showcases our efforts to forecast cryptocurrency prices using time series machine learning models.

## Project Overview

This project was developed by Vaibhav Gupta, Navya Sharma, and Anshul Kannaujia. We aimed to predict the prices of three major cryptocurrencies—Bitcoin, Ethereum, and Solana—using advanced time series forecasting techniques.

## Dataset

The dataset was sourced from the `yfinance` package in Python, focusing on the closing values of each cryptocurrency. We used:

- **Training Data:** All historical data except the last year.
- **Testing Data:** Data from the last year.

## Feature Used

We used the closing value of each cryptocurrency as our primary feature for modeling.

## Models

### 1. ARIMA (AutoRegressive Integrated Moving Average)

ARIMA is a widely-used time series forecasting method that combines autoregressive (AR) terms, differencing (I), and moving average (MA) terms to model a time series.

- **Stationarity Check:** We initially checked the stationarity of the data using Rolling Mean, Rolling Standard Deviation, and the Dickey-Fuller (DF) test. The data was found to be non-stationary.
- **Transformation:** We applied a logarithmic transformation to stabilize the variance, making the data somewhat stationary.
- **Differencing:** We performed differencing to get the value of `d` required for the ARIMA model.
- **Hyperparameter Tuning:** For determining the values of `p` and `q`, we used PACF (Partial AutoCorrelation Function) and ACF (AutoCorrelation Function) plots. The `pmdarima` package was used for fine-tuning all parameters (`p`, `d`, and `q`).
- **Trend, Seasonality, and Residuals:** We plotted trend, seasonality, and residuals to gain a better understanding of the data.
- **Forecasting:** After training the ARIMA model, we forecasted values for the testing data and evaluated the performance using RMSE (Root Mean Squared Error) and R² (R-squared).

### 2. Prophet

- **Overview:** Prophet is a forecasting tool developed by Facebook that is designed to handle seasonal effects and holidays.
- **Implementation:** We configured Prophet to capture trends and seasonality in the cryptocurrency prices.

### 3. Neural Prophet

- **Overview:** Neural Prophet extends the Prophet model by incorporating neural network components to capture complex patterns.
- **Implementation:** Neural Prophet was applied to predict future values, and its performance was compared with the other models.

## Results

We calculated RMSE and R² for each model. Due to the high volatility of cryptocurrency prices, none of the models performed exceptionally well. However, ARIMA yielded the best results among the three.

## Web Application

We developed a simple web application using the Streamlit package to provide an interactive interface for users:

- **Selection:** Users can select a cryptocurrency (Bitcoin, Ethereum, or Solana) using a dropdown menu.
- **Model Explanation:** For the selected cryptocurrency, the application explains the dataset, the ARIMA approach, and the results.
- **Results:** The web app showcases the performance of the ARIMA model, including the forecasted values and evaluation metrics.

## Power BI Dashboard

We created a Power BI dashboard to visually represent the data and model predictions. The dashboard includes:

- Various charts to illustrate trends and forecasts.
- Visualization of forecasted values using Power BI's built-in forecasting features.

## Repository Contents

This GitHub repository includes:

- **Web App Code:** Source code for the Streamlit application.
- **Jupyter Notebooks:** Implementation of ARIMA, Prophet, and Neural Prophet models for each cryptocurrency.
- **Power BI Dashboard File:** The `.pbix` file for the Power BI dashboard.
- **Project Report:** Our detailed project report is provided in both PDF and Word formats.
- **Datasets:** CSV files for Bitcoin (BTC), Ethereum (ETH), and Solana (SOL).

## How to Use

1. **Clone the Repository:**
    git clone https://github.com/yourusername/PricePredictionForCryptoCurrency.git

2. **Navigate to the Project Directory:**
    cd PricePredictionForCryptoCurrency

3. **Install Necessary Packages:**
    - pip install numpy
    - pip install pandas
    - pip install matplotlib
    - pip install yfinance
    - pip install statsmodels
    - pip install pmdarima
    - pip install prophet
    - pip install neuralprophet
    - pip install streamlit

4. **Run the Web App:**
    streamlit run app.py

5. **Explore the Jupyter Notebooks:**
   Open the `.ipynb` files in Jupyter Notebook or JupyterLab to see the implementation of different models.

## Acknowledgements
Thank you for exploring our project! We hope you find it useful and informative.

Best regards,  
Vaibhav Gupta, Navya Sharma, and Anshul Kannaujia