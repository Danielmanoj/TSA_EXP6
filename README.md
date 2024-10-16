# Ex.No: 6               HOLT WINTERS METHOD
### Date: 
### Developed by : MANOJ G
### Register no : 212222240060



### AIM: 
To create and implement Holt Winter's Method Model using python for Astrobiological dataset.

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
```
# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

# Step 2: Load the CSV file and perform initial data exploration
df = pd.read_csv('/content/astrobiological_activity_monitoring.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Step 3: Group the data by date and resample it to a monthly frequency
monthly_activity = df.groupby('Date')['Soil_Microbial_Activity'].sum().resample('M').sum()

# Check the number of observations
print(f'Total monthly observations: {len(monthly_activity)}')

# Ensure there's enough data for seasonal modeling
if len(monthly_activity) >= 24:
    seasonal_periods = 12  # Monthly data
else:
    seasonal_periods = None  # No seasonality

# Step 4: Split the data into training and test sets
train_size = int(len(monthly_activity) * 0.8)  # 80% for training
train, test = monthly_activity[:train_size], monthly_activity[train_size:]

# Step 5: Fit Holt-Winters model on training data
holt_winters_model = ExponentialSmoothing(train, seasonal='add' if seasonal_periods else None, seasonal_periods=seasonal_periods).fit()

# Step 6: Forecast on the test data
test_predictions = holt_winters_model.forecast(steps=len(test))

# Step 7: Fit the model on the entire dataset for final predictions
final_model = ExponentialSmoothing(monthly_activity, seasonal='add' if seasonal_periods else None, seasonal_periods=seasonal_periods).fit()
final_predictions = final_model.forecast(steps=12)  # Forecast for the next 12 months

# Step 8: Plot the original data, test predictions, and final predictions
plt.figure(figsize=(14, 7))
plt.plot(monthly_activity, label='Original Activity', color='blue')
plt.plot(test.index, test_predictions, label='Test Predictions', color='orange')

plt.title('Soil Microbial Activity with Test Predictions')
plt.xlabel('Date')
plt.ylabel('Activity Level')
plt.legend()
plt.show()

# Generate future dates for final predictions
future_dates = pd.date_range(start=monthly_activity.index[-1] + pd.DateOffset(1), periods=12, freq='M')
plt.plot(future_dates, final_predictions, label='Final Predictions', color='red')

plt.title('Soil Microbial Activity with Final Predictions')
plt.xlabel('Date')
plt.ylabel('Activity Level')
plt.legend()
plt.show()

# Step 9: Calculate RMSE for the test predictions
def calculate_rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))

rmse_test = calculate_rmse(test, test_predictions)
print(f'RMSE for Test Predictions: {rmse_test}')

# Optional: Print mean and standard deviation of the dataset
mean_activity = monthly_activity.mean()
std_activity = monthly_activity.std()
print(f'Mean Activity: {mean_activity}, Standard Deviation: {std_activity}')

```

### OUTPUT:


TEST_PREDICTION : 

![image](https://github.com/user-attachments/assets/22f0ec00-9401-4af8-bd9e-dc23392abd60)


FINAL_PREDICTION : 

![image](https://github.com/user-attachments/assets/4e6cdcfd-3b42-4602-b786-774af59af515)


### RESULT:
Thus the program created successfully based on the Holt Winters Method model.
