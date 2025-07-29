import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# 📌 Function to calculate MAPE (Mean Absolute Percentage Error)
def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_indices = y_true != 0
    mape = np.mean(np.abs((y_true[non_zero_indices] - y_pred[non_zero_indices]) / y_true[non_zero_indices])) * 100
    return mape

# 📌 Function to compute model accuracy
def calculate_accuracy(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mape_max = calculate_mape(y_true[:, 0], y_pred[:, 0])
    mape_min = calculate_mape(y_true[:, 1], y_pred[:, 1])
    
    accuracy_max = 100 - mape_max
    accuracy_min = 100 - mape_min
    
    return mae, accuracy_max, accuracy_min

# 📌 Function to load weather data for a given city
def load_data(city):
    file_map = {
        "Mysore": "weather_data/weather_mysore.csv",
        "Bangalore": "weather_data/weather_bangalore.csv",
        "Belgavi": "weather_data/weather_belgavi.csv",
        "Chikmagalur": "weather_data/weather_chikmagalur.csv",
        "Chickballapur": "weather_data/weather_chickballapur.csv"
    }
    file_path = file_map.get(city)
    
    if file_path:
        try:
            data = pd.read_csv(file_path)
            return data
        except FileNotFoundError:
            st.error("⚠️ Data file not found. Please check the file path.")
    return None

# 📌 Function to preprocess the dataset
def preprocess_data(data):
    # Convert Date column to datetime format
    data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')

    # Handle missing values
    data.replace(-999, np.nan, inplace=True)
    data.dropna(inplace=True)

    # Sort data by date
    data = data.sort_values(by='Date')

    # Feature Engineering
    data['prev_max_temp'] = data['MaxTemp'].shift(1)
    data['prev_min_temp'] = data['MinTemp'].shift(1)
    data['prev_precipitation'] = data['PrecipitationAmount'].shift(1)

    return data.dropna()

# 📌 Function to train and evaluate the model
def train_model(data):
    features = ['prev_max_temp', 'prev_min_temp', 'prev_precipitation']
    target = ['MaxTemp', 'MinTemp', 'PrecipitationAmount']

    # Train-test split (80-20)
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]

    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]

    # Train RandomForest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae, acc_max, acc_min = calculate_accuracy(y_test.values[:, :2], predictions[:, :2])

    return model, mae, acc_max, acc_min

# 📌 Function to generate weather forecasts
def generate_forecast(model, last_known, forecast_days=365):
    forecasts = []
    forecast_dates = pd.date_range(start=last_known['Date'] + pd.Timedelta(days=1), periods=forecast_days)

    for forecast_date in forecast_dates:
        features = np.array([[last_known['prev_max_temp'], last_known['prev_min_temp'], last_known['prev_precipitation']]])
        prediction = model.predict(features)[0]

        # Compute average temperature
        avg_temp = int((prediction[0] + prediction[1]) / 2)

        # Determine weather condition
        precipitation = prediction[2]
        if precipitation < 1:
            weather_condition = "HOT Weather"
        elif 1.01 <= precipitation <= 2.5:
            weather_condition = "Sunny"
        elif 2.51 <= precipitation <= 5:
            weather_condition = "Cloudy"
        elif 5.01 <= precipitation <= 8:
            weather_condition = "Moderate Rain"
        else:
            weather_condition = "Heavy Rain"

        # Format values
        forecasts.append({
            "Date": forecast_date.strftime('%Y-%m-%d'),
            "MaxTemp": f"{int(prediction[0])} °C",
            "MinTemp": f"{int(prediction[1])} °C",
            "PrecipitationAmount": f"{precipitation:.2f} mm",
            "AverageTemp": f"{avg_temp} °C",
            "Weather": weather_condition
        })

        # Update last known values
        last_known['prev_max_temp'], last_known['prev_min_temp'], last_known['prev_precipitation'] = prediction

    return pd.DataFrame(forecasts)

# 📌 Main function to run the Streamlit app
def main():
    st.title("🌤️ Weather Forecasting App")

    # City selection dropdown
    city = st.selectbox("📍 Select a City", ["Select City", "Mysore", "Bangalore", "Belgavi", "Chikmagalur", "Chickballapur"])

    if city != "Select City":
        data = load_data(city)

        if data is not None:
            data = preprocess_data(data)

            # Train the model and get evaluation metrics
            model, mae, acc_max, acc_min = train_model(data)

            # Display Model Performance
            st.subheader("📊 Model Performance")
            st.write(f"✅ **Mean Absolute Error:** {mae:.2f}")
            st.write(f"✅ **Max Temp Accuracy:** {acc_max:.2f}%")
            st.write(f"✅ **Min Temp Accuracy:** {acc_min:.2f}%")

            # Generate forecast for the next 365 days
            last_known = data.iloc[-1].copy()
            forecast_df = generate_forecast(model, last_known, forecast_days=365)

            # 📅 Date selection for weather forecast
            selected_date = st.date_input("📆 Select a Date for Forecast", pd.to_datetime(forecast_df["Date"].min()).date())


            if str(selected_date) in forecast_df["Date"].values:
                forecast_row = forecast_df[forecast_df["Date"] == str(selected_date)]
                st.subheader(f"🌡️ Weather Forecast for {selected_date}")

                st.write(f"🌞 **Max Temperature:** {forecast_row['MaxTemp'].values[0]}")
                st.write(f"❄️ **Min Temperature:** {forecast_row['MinTemp'].values[0]}")
                st.write(f"🌧️ **Precipitation:** {forecast_row['PrecipitationAmount'].values[0]}")
                st.write(f"🌍 **Weather Condition:** {forecast_row['Weather'].values[0]}")
            else:
                st.warning("⚠️ Selected date is out of range. Please choose a valid date.")

          

if __name__ == "__main__":
    main()
