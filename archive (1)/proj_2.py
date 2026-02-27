"""
================================================================================
  STOCK PRICE PREDICTION OF TATA STEEL USING LSTM (Deep Learning)
  Capstone Project - BCA/BSc/MCA Final Year
================================================================================
  Author      : [Your Name]
  Institution : [Your College/University Name]
  Guide       : [Your Project Guide Name]
  Date        : 2025
  Description : This project predicts Tata Steel's stock prices using a 
                Stacked LSTM model. It includes data download, preprocessing,
                model training, evaluation, visualization, and forecasting.
================================================================================
"""

# ==============================================================================
# SECTION 1: IMPORT LIBRARIES
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")

# Data
import yfinance as yf

# Preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Technical Indicators
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 70)
print("  Stock Price Prediction of Tata Steel Using LSTM")
print("=" * 70)
print(f"  TensorFlow Version : {tf.__version__}")
print("=" * 70)

# ==============================================================================
# SECTION 2: DATA COLLECTION
# ==============================================================================

print("\n[INFO] Downloading historical data for TATASTEEL.NS from yfinance...")

ticker = "TATASTEEL.NS"
start_date = "2015-01-01"
end_date   = pd.Timestamp.today().strftime("%Y-%m-%d")

df = yf.download(ticker, start=start_date, end=end_date, progress=False)

if df.empty:
    raise ValueError("No data downloaded. Check your internet connection or ticker symbol.")

# Use only the Close price
df = df[["Close"]].copy()
df.dropna(inplace=True)

print(f"  Data Shape   : {df.shape}")
print(f"  Date Range   : {df.index.min().date()} → {df.index.max().date()}")
print(f"  Sample Data  :\n{df.tail(5)}")

# ==============================================================================
# SECTION 3: TECHNICAL INDICATORS (Moving Averages + RSI)
# ==============================================================================

print("\n[INFO] Computing Technical Indicators (MA50, MA200, RSI)...")

# Moving Averages
df["MA50"]  = df["Close"].rolling(window=50).mean()
df["MA200"] = df["Close"].rolling(window=200).mean()

# RSI - Relative Strength Index (14-day)
def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs  = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

df["RSI"] = compute_rsi(df["Close"])
print("  Technical indicators computed successfully.")

# ==============================================================================
# SECTION 4: DATA PREPROCESSING
# ==============================================================================

print("\n[INFO] Preprocessing data...")

# Use only Close for LSTM
close_prices = df["Close"].values.reshape(-1, 1)

# Train/Test Split: 80% train, 20% test
train_size  = int(len(close_prices) * 0.80)
train_data  = close_prices[:train_size]
test_data   = close_prices[train_size:]

print(f"  Total Samples : {len(close_prices)}")
print(f"  Train Samples : {len(train_data)}")
print(f"  Test  Samples : {len(test_data)}")

# Normalize using MinMaxScaler (fit ONLY on training data to prevent data leakage)
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_data)
test_scaled  = scaler.transform(test_data)

# Create time-series sequences with 60-day lookback window
LOOKBACK = 60

def create_sequences(data, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_scaled, LOOKBACK)
X_test,  y_test  = create_sequences(test_scaled,  LOOKBACK)

# Reshape for LSTM: [samples, time_steps, features]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test  = X_test.reshape((X_test.shape[0],  X_test.shape[1],  1))

print(f"  X_train Shape : {X_train.shape}")
print(f"  X_test  Shape : {X_test.shape}")

# ==============================================================================
# SECTION 5: BUILD STACKED LSTM MODEL
# ==============================================================================

print("\n[INFO] Building Stacked LSTM Model...")

model = Sequential([
    # First LSTM Layer — return sequences for stacking
    LSTM(units=64, return_sequences=True, input_shape=(LOOKBACK, 1)),
    Dropout(0.2),

    # Second LSTM Layer
    LSTM(units=64, return_sequences=False),
    Dropout(0.2),

    # Output Layer
    Dense(units=1)
])

model.compile(optimizer="adam", loss="mean_squared_error")
model.summary()

# ==============================================================================
# SECTION 6: TRAIN MODEL
# ==============================================================================

print("\n[INFO] Training LSTM Model (this may take a few minutes)...")

early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs          = 50,
    batch_size      = 32,
    validation_split= 0.10,
    callbacks       = [early_stop],
    verbose         = 1
)

print("\n  Model training complete.")

# ==============================================================================
# SECTION 7: MODEL EVALUATION
# ==============================================================================

print("\n[INFO] Evaluating Model Performance...")

# Predict on test set
y_pred_scaled = model.predict(X_test)

# Inverse transform to original price scale
y_pred = scaler.inverse_transform(y_pred_scaled)
y_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Metrics
mse  = mean_squared_error(y_actual, y_pred)
rmse = np.sqrt(mse)

print(f"  Mean Squared Error (MSE)  : {mse:.4f}")
print(f"  Root Mean Squared Error   : {rmse:.4f}")
print(f"  RMSE (in ₹)               : ₹{rmse:.2f}")

# ==============================================================================
# SECTION 8: FUTURE 30-DAY FORECAST
# ==============================================================================

print("\n[INFO] Forecasting next 30 business days...")

# Last 60 days of scaled data as seed
last_60_scaled = scaler.transform(close_prices[-LOOKBACK:])
forecast_input = last_60_scaled.reshape(1, LOOKBACK, 1)

future_predictions = []

for _ in range(30):
    next_price_scaled = model.predict(forecast_input, verbose=0)
    future_predictions.append(next_price_scaled[0, 0])
    # Append and slide window
    forecast_input = np.append(
        forecast_input[:, 1:, :],
        next_price_scaled.reshape(1, 1, 1),
        axis=1
    )

# Inverse transform forecast
future_prices = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Generate future business dates
last_date     = df.index[-1]
future_dates  = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=30)

future_df = pd.DataFrame({"Date": future_dates, "Forecast": future_prices.flatten()})
future_df.set_index("Date", inplace=True)

print("  30-Day Forecast (first 10 days):")
print(future_df.head(10).to_string())

# ==============================================================================
# SECTION 9: MATPLOTLIB VISUALIZATIONS
# ==============================================================================

print("\n[INFO] Generating plots...")

# --- Prepare date arrays for test predictions ---
test_dates = df.index[train_size + LOOKBACK:]

# Ensure lengths match
min_len = min(len(test_dates), len(y_actual), len(y_pred))
test_dates = test_dates[:min_len]
y_actual   = y_actual[:min_len]
y_pred     = y_pred[:min_len]

# ── Plot 1: Training Loss ──────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(history.history["loss"],     label="Training Loss",   color="steelblue")
ax1.plot(history.history["val_loss"], label="Validation Loss", color="coral")
ax1.set_title("LSTM Model — Training & Validation Loss", fontsize=14, fontweight="bold")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss (MSE)")
ax1.legend()
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot1_training_loss.png", dpi=150)
plt.show()

# ── Plot 2: Actual vs Predicted (Test Period) ──────────────────────────────
fig2, ax2 = plt.subplots(figsize=(14, 5))
ax2.plot(test_dates, y_actual, label="Actual Price",    color="steelblue", linewidth=1.5)
ax2.plot(test_dates, y_pred,   label="Predicted Price", color="orange",    linewidth=1.5, linestyle="--")
ax2.set_title("Tata Steel — Actual vs Predicted Stock Prices (Test Set)", fontsize=14, fontweight="bold")
ax2.set_xlabel("Date")
ax2.set_ylabel("Price (₹)")
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.xticks(rotation=45)
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot2_actual_vs_predicted.png", dpi=150)
plt.show()

# ── Plot 3: 30-Day Future Forecast ────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(14, 5))
# Last 120 days of actual for context
ax3.plot(df.index[-120:], df["Close"].values[-120:], label="Historical Close",
         color="steelblue", linewidth=1.5)
ax3.plot(future_df.index, future_df["Forecast"],
         label="30-Day Forecast",  color="green",
         linewidth=2, linestyle="--", marker="o", markersize=4)
ax3.axvline(x=df.index[-1], color="gray", linestyle=":", linewidth=1.5, label="Forecast Start")
ax3.set_title("Tata Steel — 30 Business Day Price Forecast",
              fontsize=14, fontweight="bold")
ax3.set_xlabel("Date")
ax3.set_ylabel("Price (₹)")
ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.xticks(rotation=45)
ax3.legend()
ax3.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot3_future_forecast.png", dpi=150)
plt.show()

# ── Plot 4: Moving Averages + RSI ─────────────────────────────────────────
fig4, (ax4a, ax4b) = plt.subplots(
    2, 1, figsize=(14, 8), sharex=True,
    gridspec_kw={"height_ratios": [3, 1]}
)

ax4a.plot(df.index, df["Close"], label="Close Price", color="steelblue", linewidth=1, alpha=0.8)
ax4a.plot(df.index, df["MA50"],  label="MA50",        color="orange",    linewidth=1.5)
ax4a.plot(df.index, df["MA200"], label="MA200",       color="red",       linewidth=1.5)
ax4a.set_title("Tata Steel — Close Price with Moving Averages",
               fontsize=14, fontweight="bold")
ax4a.set_ylabel("Price (₹)")
ax4a.legend()
ax4a.grid(True, alpha=0.3)

ax4b.plot(df.index, df["RSI"], label="RSI (14)", color="purple", linewidth=1)
ax4b.axhline(70, color="red",   linestyle="--", linewidth=1, alpha=0.7, label="Overbought (70)")
ax4b.axhline(30, color="green", linestyle="--", linewidth=1, alpha=0.7, label="Oversold (30)")
ax4b.set_ylabel("RSI")
ax4b.set_xlabel("Date")
ax4b.legend(loc="upper left")
ax4b.xaxis.set_major_locator(mdates.YearLocator())
ax4b.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax4b.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plot4_ma_rsi.png", dpi=150)
plt.show()

# ==============================================================================
# EXTRA: TABLES FOR REPORT (NO LOGIC CHANGE)
# ==============================================================================

# Last 10 test predictions vs actual as table
print("\n[INFO] Last 10 Actual vs Predicted (Test Set):")
last_10 = pd.DataFrame({
    "Date": test_dates[-10:],
    "Actual_Close": y_actual[-10:, 0],
    "Predicted_Close": y_pred[-10:, 0]
})
last_10.set_index("Date", inplace=True)
print(last_10.round(2).to_string())

# 30-day forecast full table (already printed first 10 above)
print("\n[INFO] Full 30-Day Forecast Table:")
print(future_df.round(2).to_string())

# Also save tables to CSV for project report
last_10.to_csv("last_10_actual_vs_predicted.csv")
future_df.to_csv("future_30day_forecast.csv")

# ==============================================================================
# SECTION 10: INTERACTIVE PLOTLY DASHBOARD
# ==============================================================================

print("\n[INFO] Creating Interactive Plotly Dashboard...")

fig_plotly = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    row_heights=[0.55, 0.25, 0.20],
    subplot_titles=(
        "Tata Steel Close Price — Actual vs Predicted + Forecast",
        "Moving Averages (MA50, MA200)",
        "RSI (14-Day)"
    )
)

# Row 1: Actual + Predicted + Forecast
fig_plotly.add_trace(go.Scatter(
    x=df.index, y=df["Close"].values.flatten(),
    name="Close Price", line=dict(color="#2196F3", width=1.5)
), row=1, col=1)

fig_plotly.add_trace(go.Scatter(
    x=test_dates, y=y_pred.flatten(),
    name="LSTM Predicted", line=dict(color="#FF9800", width=2, dash="dash")
), row=1, col=1)

fig_plotly.add_trace(go.Scatter(
    x=future_df.index, y=future_df["Forecast"],
    name="30-Day Forecast", line=dict(color="#4CAF50", width=2.5, dash="dot"),
    marker=dict(size=5)
), row=1, col=1)

# Row 2: Moving Averages
fig_plotly.add_trace(go.Scatter(
    x=df.index, y=df["MA50"].values.flatten(),
    name="MA50",  line=dict(color="#FF5722", width=1.5)
), row=2, col=1)
fig_plotly.add_trace(go.Scatter(
    x=df.index, y=df["MA200"].values.flatten(),
    name="MA200", line=dict(color="#9C27B0", width=1.5)
), row=2, col=1)

# Row 3: RSI
fig_plotly.add_trace(go.Scatter(
    x=df.index, y=df["RSI"].values.flatten(),
    name="RSI", line=dict(color="#607D8B", width=1.5), fill="tozeroy",
    fillcolor="rgba(96,125,139,0.1)"
), row=3, col=1)

fig_plotly.add_hline(y=70, line=dict(color="red",   dash="dash", width=1), row=3, col=1)
fig_plotly.add_hline(y=30, line=dict(color="green", dash="dash", width=1), row=3, col=1)

# Layout
fig_plotly.update_layout(
    title=dict(text="<b>Tata Steel Stock Price Prediction — LSTM Dashboard</b>",
               font=dict(size=20), x=0.5),
    height=850,
    template="plotly_white",
    hovermode="x unified",
    legend=dict(orientation="h", x=0, y=1.08)
)

fig_plotly.update_yaxes(title_text="Price (₹)", row=1, col=1)
fig_plotly.update_yaxes(title_text="Price (₹)", row=2, col=1)
fig_plotly.update_yaxes(title_text="RSI",       row=3, col=1, range=[0, 100])

fig_plotly.write_html("dashboard_interactive.html")
fig_plotly.show()

print("  Interactive dashboard saved as dashboard_interactive.html")

# ==============================================================================
# SECTION 11: SUMMARY REPORT
# ==============================================================================

print("\n" + "=" * 70)
print("  PROJECT SUMMARY")
print("=" * 70)
print(f"  Stock Ticker         : {ticker}")
print(f"  Data Period          : {df.index.min().date()} to {df.index.max().date()}")
print(f"  Total Data Points    : {len(close_prices)}")
print(f"  Training Samples     : {len(X_train)}")
print(f"  Testing Samples      : {len(X_test)}")
print(f"  Lookback Window      : {LOOKBACK} days")
print(f"  LSTM Architecture    : 2 LSTM Layers (64 units each) + Dense(1)")
print(f"  Optimizer            : Adam | Loss: MSE")
print(f"  Epochs Trained       : {len(history.history['loss'])}")
print(f"  Final Train Loss     : {history.history['loss'][-1]:.6f}")
print(f"  Final Val   Loss     : {history.history['val_loss'][-1]:.6f}")
print(f"  MSE (Test)           : {mse:.4f}")
print(f"  RMSE (Test)          : ₹{rmse:.2f}")
print(f"\n  Forecasted Price     :")
print(f"    Day  1  : ₹{future_prices[0,0]:.2f}")
print(f"    Day  15 : ₹{future_prices[14,0]:.2f}")
print(f"    Day  30 : ₹{future_prices[29,0]:.2f}")
print("=" * 70)
print("  All plots and tables saved. Project complete.")
print("=" * 70)
