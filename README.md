## Developed By : Shriram S
## Register No : 212222240098
## Date:

# EX.NO.09        A project on Time series analysis on Vegetable Price using ARIMA model 


### AIM:
To Create a project on Time series analysis on price change on vegetable price using ARIMA model in  Python and compare with other models.

### ALGORITHM:
1. Explore the dataset of prices. 
2. Check for stationarity of time series time series plot
   ACF plot and PACF plot
   ADF test
   Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions

### PROGRAM:
#### Import the neccessary packages

```py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")
```

#### Load the dataset
```py
data = pd.read_csv('prices.csv')
```
#### Convert 'Date' column to datetime format
```py
data['Price Dates'] = pd.to_datetime(data['Price Dates'], dayfirst=True)
```
#### Set 'Date' column as index
```py
data.set_index('Price Dates', inplace=True)
```
#### ACF and PACF 
```py
def check_stationarity(series):
    result = adfuller(series)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    if result[1] < 0.05:
        print("Series is stationary.")
    else:
        print("Series is not stationary.")

check_stationarity(data['Tomato'])  # Example for Tomato prices

data_diff = data.diff().dropna()  # First-order differencing
check_stationarity(data_diff['Tomato'])  # Recheck after differencing

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(data_diff['Tomato'])
plot_pacf(data_diff['Tomato'])
plt.show()  
```
#### Arima Model
```py
# Train-test split
train, test = data['Tomato'][:250], data['Tomato'][250:]

# Define ARIMA model (e.g., order (1,1,1))
model = ARIMA(train, order=(1, 1, 1))
arima_result = model.fit()

# Forecasting
forecast = arima_result.forecast(steps=len(test))
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test')
plt.plot(test.index, forecast, label='Forecast')
plt.legend()
plt.show()

mse = mean_squared_error(test, forecast)
print(f'Mean Squared Error: {mse}')

final_model = ARIMA(data['Tomato'], order=(1, 1, 1))
final_model_fit = final_model.fit()
future_forecast = final_model_fit.forecast(steps=30)  # Forecasting 30 days ahead

plt.plot(data.index, data['Tomato'], label='Actual')
plt.plot(pd.date_range(data.index[-1], periods=31, freq='D')[1:], future_forecast, label='Forecast')
plt.legend()
plt.show()
```




### OUTPUT:

### ACF and PACF 
![image](https://github.com/user-attachments/assets/f5aad6ca-bfe0-4588-abdb-c611f1571570)
![image](https://github.com/user-attachments/assets/6753bf5b-288f-494d-a3b5-69e479f7db3c)

### ARIMA
![image](https://github.com/user-attachments/assets/31747140-c8cd-4644-af81-e1cb9d213013)

### RESULT:
Thus the project on Time series analysis on Annual percentage change onj India's GDP based on the ARIMA model using python is executed successfully.
