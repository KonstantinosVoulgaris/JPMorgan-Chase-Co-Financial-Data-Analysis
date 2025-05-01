import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load Data from CSV file
file_path = r'C:\Users\Giannis\Desktop\Companies\stock_JPM.csv'  # Provide the full path to the file here
df = pd.read_csv(file_path)

# Plot 1: Display the first few rows of the dataset to understand the data structure
print("RESULT 1: FIRST ROWS OF THE DATASET")
print(df.head())

# Plot 2: Data Cleaning: Fill Missing Values
df.ffill(inplace=True)
print("\nRESULT 2: CLEANED DATA WITH FILLED MISSING VALUES")
print(df.isnull().sum())  # Check if there are still missing values

# Convert the 'Date' column to datetime format for analysis
df['Date'] = pd.to_datetime(df['Date'])

# Create a new column for the 'Range' (High - Low)
df['Range'] = df['High'] - df['Low']

# Calculate Returns
df['Return'] = df['Close'].pct_change()

# Calculate 30-day Volatility
df['Volatility'] = df['Return'].rolling(window=30).std()

# Analysis: Calculate Mean Closing Price
mean_close = df['Close'].mean()

# Plot 3: Display Mean Closing Price
print(f"\nRESULT 3: MEAN CLOSING PRICE: {mean_close}")

# Plot 4: Plot Closing Price over Time
print("\nRESULT 4: CLOSING PRICE GRAPH OVER TIME")
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Close'], label='Closing Price', color='blue')
plt.title('JPM Stock Closing Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()

# Plot 5: Plot Return Distribution using Seaborn
print("\nRESULT 5: RETURN DISTRIBUTION GRAPH")
plt.figure(figsize=(10, 6))
sns.histplot(df['Return'], bins=50, kde=True)
plt.title('Distribution of Daily Returns')
plt.xlabel('Daily Return')
plt.ylabel('Frequency')
plt.show()

# Plot 6: Interactive Closing Price Graph with Plotly
print("\nRESULT 6: INTERACTIVE CLOSING PRICE GRAPH")
fig = px.line(df, x='Date', y='Close', title='JPM Stock Closing Price Over Time')
fig.show()

# Plot 7: Heatmap with Seaborn for Feature Correlation
print("\nRESULT 7: FEATURE CORRELATION HEATMAP")
corr_matrix = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Range', 'Return']].corr()
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# If you want to Predict the Closing Price (Stock Price Prediction)
df['Date_ordinal'] = df['Date'].map(pd.Timestamp.toordinal)  # Convert date to numerical value

# Use Linear Regression for prediction
X = df[['Date_ordinal']]
y = df['Close']

# Split data into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Create and Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Make Predictions
predictions = model.predict(X_test)

# Plot 8: Plot Predictions vs Actual Values
print("\nRESULT 8: PREDICTIONS VS ACTUAL VALUES GRAPH")
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Close'], label='Actual')
plt.plot(df['Date'].iloc[-len(predictions):], predictions, label='Predicted', linestyle='--')
plt.legend()
plt.title('Stock Price Prediction')
plt.show()

# Plot 9: Check for Seasonality (per month)
print("\nRESULT 9: AVERAGE MONTHLY RETURNS")
df['Month'] = df['Date'].dt.month
monthly_returns = df.groupby('Month')['Return'].mean()
plt.figure(figsize=(10, 6))
monthly_returns.plot(kind='bar', color='skyblue')
plt.title('Average Monthly Returns for JPM Stock')
plt.xlabel('Month')
plt.ylabel('Average Return')
plt.grid(True)
plt.show()

# Plot 10: Volume – Performance Correlation
print("\nRESULT 10: RETURN-VOLUME CORRELATION")
correlation = df[['Return', 'Volume']].corr().iloc[0, 1]
print(f"Correlation between Return and Volume: {correlation:.4f}")
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Volume', y='Return', data=df, alpha=0.3)
plt.title('Scatterplot of Volume vs Return')
plt.xlabel('Volume')
plt.ylabel('Daily Return')
plt.grid(True)
plt.show()

# Plot 11: Identifying High Volatility Points
print("\nRESULT 11: HIGH VOLATILITY PERIODS")
high_vol = df[df['Volatility'] > df['Volatility'].quantile(0.95)]
print(f"Dates with High Volatility (top 5%):\n{high_vol[['Date', 'Volatility']].head()}")
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Volatility'], label='30-day Volatility')
plt.axhline(df['Volatility'].quantile(0.95), color='red', linestyle='--', label='95th Percentile Threshold')
plt.title('JPM 30-Day Rolling Volatility Over Time')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.legend()
plt.grid(True)
plt.show()

# Plot 12: Calculate 50-day and 200-day Simple Moving Averages
df['SMA50'] = df['Close'].rolling(window=50).mean()
df['SMA200'] = df['Close'].rolling(window=200).mean()
print("\nRESULT 12: MOVING AVERAGES (SMA50 & SMA200)")
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Close'], label='Close Price', alpha=0.6)
plt.plot(df['Date'], df['SMA50'], label='SMA50', color='orange')
plt.plot(df['Date'], df['SMA200'], label='SMA200', color='red')
plt.title('JPM Stock - SMA50 & SMA200')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()

# Plot 13: RSI Function
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Calculate RSI
df['RSI'] = calculate_rsi(df['Close'])

# Plot 14: RSI
print("\nRESULT 13: RSI (RELATIVE STRENGTH INDEX)")
plt.figure(figsize=(12, 4))
plt.plot(df['Date'], df['RSI'], label='RSI', color='purple')
plt.axhline(70, color='red', linestyle='--', label='Overbought (70)')
plt.axhline(30, color='green', linestyle='--', label='Oversold (30)')
plt.title('JPM Stock - RSI (14-day)')
plt.xlabel('Date')
plt.ylabel('RSI')
plt.legend()
plt.grid(True)
plt.show()

# Plot 15: Calculate MACD and Signal Line
exp1 = df['Close'].ewm(span=12, adjust=False).mean()
exp2 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = exp1 - exp2
df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

# Plot MACD
print("\nRESULT 14: MACD (MOVING AVERAGE CONVERGENCE DIVERGENCE)")
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['MACD'], label='MACD', color='blue')
plt.plot(df['Date'], df['Signal'], label='Signal Line', color='orange')
plt.axhline(0, color='gray', linestyle='--')
plt.title('JPM Stock - MACD')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
