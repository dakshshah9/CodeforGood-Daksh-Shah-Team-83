import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the solar panel and weather data into a Pandas DataFrame
data = pd.read_csv('solar_data.csv')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data[['temperature', 'humidity', 'cloud_cover']], 
    data['solar_output'], 
    test_size=0.2
)

# Train a linear regression model on the training data
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the mean squared error of the predictions
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)