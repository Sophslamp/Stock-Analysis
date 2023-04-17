import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Collect Data
df = pd.read_csv("stock_data.csv")

# Step 2: Data Preparation
df = df.dropna() # remove any missing data
X = df[["Open", "Close", "Volume"]] # select features
y = df["Adj Close"] # select target variable

# Step 3: Feature Engineering
# create moving averages
ma_5 = X["Close"].rolling(window=5).mean()
ma_10 = X["Close"].rolling(window=10).mean()

# Step 4: Model Selection
model = LinearRegression()

# Step 5: Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Step 6: Model Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("RMSE:", rmse)

# Step 7: Prediction
future_data = pd.read_csv("future_data.csv")
future_pred = model.predict(future_data)
print("Future Predictions:", future_pred)
