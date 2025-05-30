
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import itertools

st.set_page_config(page_title="Gold Price Forecast", layout="wide")
st.title("📈 Gold Price Forecast with Linear Regression & ARIMA")

uploaded_file = st.file_uploader("Upload your Gold CSV file", type=["csv"])
if uploaded_file is not None:
    gold_df = pd.read_csv(uploaded_file)
    gold_df['date'] = pd.to_datetime(gold_df['date'])
    gold_df.set_index('date', inplace=True)
    gold_df.sort_index(inplace=True)
    gold_df = gold_df.dropna()

    st.subheader("📊 Raw Data")
    st.write(gold_df.tail())

    # Plot gold price
    st.subheader("🟡 Gold Price Over Time")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(gold_df.index, gold_df['price'], color='gold', label='Gold Price')
    ax.set_title('Daily Gold Price Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    # Linear Regression
    gold_df['time_ordinal'] = gold_df.index.map(pd.Timestamp.toordinal)
    X = gold_df[['time_ordinal']]
    y = gold_df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    mae_lr = mean_absolute_error(y_test, y_pred_lr)

    st.subheader("🔵 Linear Regression Forecast")
    fig, ax = plt.subplots()
    ax.plot(y_test.index, y_test, label='Actual')
    ax.plot(y_test.index, y_pred_lr, label='Linear Regression Forecast', color='blue')
    ax.legend()
    st.pyplot(fig)
    st.markdown(f"**Linear Regression RMSE:** {rmse_lr:.2f}, **MAE:** {mae_lr:.2f}")

    # ARIMA Model Selection
    st.subheader("🔮 ARIMA Forecast")
    p = d = q = range(0, 3)
    pdq = list(itertools.product(p, d, q))
    best_aic = np.inf
    best_order = None
    best_mdl = None
    train_arima = gold_df['price'][:int(len(gold_df) * 0.8)]

    for order in pdq:
        try:
            mdl = sm.tsa.ARIMA(train_arima, order=order).fit()
            if mdl.aic < best_aic:
                best_aic = mdl.aic
                best_order = order
                best_mdl = mdl
        except Exception:
            continue

    st.write(f"**Best ARIMA order:** {best_order} (AIC: {best_aic:.2f})")
    arima_model = sm.tsa.ARIMA(train_arima, order=best_order).fit()
    n_test = len(gold_df) - len(train_arima)
    arima_forecast = arima_model.forecast(steps=n_test)
    actual = gold_df['price'][len(train_arima):]
    rmse_arima = np.sqrt(mean_squared_error(actual, arima_forecast))
    mae_arima = mean_absolute_error(actual, arima_forecast)

    fig, ax = plt.subplots()
    ax.plot(actual.index, actual, label='Actual Price')
    ax.plot(actual.index, arima_forecast, label='ARIMA Forecast', color='green')
    ax.legend()
    st.pyplot(fig)
    st.markdown(f"**ARIMA RMSE:** {rmse_arima:.2f}, **MAE:** {mae_arima:.2f}")
else:
    st.info("📤 Please upload a CSV file with 'date' and 'price' columns.")
