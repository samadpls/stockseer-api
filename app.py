import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from fastapi import FastAPI, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from keras.models import Sequential
from keras.layers import Dense, LSTM

import matplotlib.pyplot as plt
import numpy as np

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def download_and_preprocess_data(company, time_diff_unit):
    """
    Downloads and preprocesses stock data for a given company based on the time difference unit.

    Args:
        company (str): The stock ticker symbol.
        time_diff_unit (str): The time difference unit (days, hours, minutes).

    Returns:
        pandas.DataFrame: The preprocessed DataFrame containing closing prices, or an empty DataFrame if no data is found.

    Raises:
        ValueError: If time difference unit is invalid (not 'days', 'hours', or 'minutes').
    """

    if time_diff_unit.lower() not in ('days', 'hours', 'minutes'):
        raise ValueError(
            "Invalid time difference unit. Use 'days', 'hours', or 'minutes'.")

    end = datetime.now()

    # Set default maximum download period based on time_diff_unit
    if time_diff_unit.lower() == 'days':
        max_period = 365*10
        start = end - timedelta(days=max_period)
    elif time_diff_unit.lower() == 'hours':
        max_period = 60
    elif time_diff_unit.lower() == 'minutes':
        max_period = 7
    else:
        raise ValueError(
            "Unexpected error: invalid time_diff_unit after validation.")

    start = end - timedelta(days=max_period)
    # Download data for the maximum period

    print(
        f"Downloading data for {company} (up to day {max_period} with interval {time_diff_unit})...")
    if time_diff_unit.lower() == 'minutes':
        interval = '1m'
    elif time_diff_unit.lower() == 'hours':
        interval = '1h'
    else:
        interval = '1d'
    data = yf.download(company, start, end, interval=interval)

    if data.empty:
        print("No data found.")
        return pd.DataFrame()

    data["company_name"] = company
    print(data)
    return data.filter(["Close"])


# train a stock price prediction model


def train_model(data, model_type="RandomForestRegressor", training_split=0.95):

    dataset = data.values

    training_data_len = int(len(dataset) * training_split)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    train_data = scaled_data[:training_data_len, :]

    # Split the data into x_train and y_train data sets
    x_train = []
    y_train = []
    window_size = int(training_data_len * 0.05)
    for i in range(window_size, len(train_data)):
        x_train.append(train_data[i-window_size:i, 0])
        y_train.append(train_data[i, 0])

    # Convert the x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    if model_type == "LSTM":
        model = Sequential()
        model.add(LSTM(128, return_sequences=True,
                  input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, batch_size=1, epochs=1)
    else:
        # Scikit-learn models
        if model_type == "LinearRegression":
            model = LinearRegression()
        elif model_type == "KNeighborsRegressor":
            model = KNeighborsRegressor()
        elif model_type == "XGBRegressor":
            model = XGBRegressor()
        elif model_type == "RandomForestRegressor":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == "ExtraTreesRegressor":
            model = ExtraTreesRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Invalid model type: {model_type}")
        model.fit(x_train, y_train)

    return model, scaled_data, scaler, training_data_len


# predictions
def make_predictions(model, scaled_data, scaler, training_data_len):

    window_size = int(training_data_len * 0.05)
    test_data = scaled_data[training_data_len - window_size:, :]
    x_test = []
    for i in range(window_size, len(test_data)):
        x_test.append(test_data[i-window_size:i, 0])

    # Convert the data to a numpy array
    x_test = np.array(x_test)

    # Get the models predicted price values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
    return predictions


@app.post("/api/predict")
async def predict_stock_price(
    company: str = Query(
        default="GOOG", description="The stock ticker symbol"),
    time_diff_value: str = Query(
        default="days", description="Time difference unit (days, hours, minutes)"
    ),
    model_type: str = Query(
        default="RandomForestRegressor",
        description="Model type ( RandomForestRegressor, ExtraTreesRegressor,"
        'XGBRegressor, LinearRegression, KNeighborsRegressor, or LSTM)',
    ),
):
    """
    Predicts the closing stock price for a given company based on user-specified parameters.

    Args:
        company (str, optional): The stock ticker symbol. Defaults to "GOOG".
        time_diff_unit (str, optional): The time difference unit (days, hours, minutes).
        Defaults to "days".
        model_type (str, optional): The model type to use for prediction
        Model type (RandomForestRegressor, ExtraTreesRegressor, XGBRegressor,
        LinearRegression, KNeighborsRegressor, or LSTM implementation)

    Returns:
        dict: A dictionary containing the predicted closing price and
        additional information.
    """

    try:
        print("company-->", company)
        data = download_and_preprocess_data(
            company, time_diff_value)
        model, scaled_data, scaler, training_data_len = train_model(
            data, model_type)
        predictions = make_predictions(
            model, scaled_data, scaler, training_data_len)
        # Plot the data
        training_data_len = int(len(data) * 0.95)
        train = data[:training_data_len]
        valid = data[training_data_len:]
        valid['Predictions'] = predictions
        valid['Predictions'] = valid['Predictions'].apply(
            lambda x: x.item() if isinstance(x, np.generic) else x)

        plt.figure(figsize=(16, 6))
        plt.title(f"{model_type} Model")
        plt.ylabel("Close Price USD ($)", fontsize=18)
        plt.plot(train["Close"])
        plt.plot(valid[["Close", "Predictions"]])
        plt.legend(["Train", "Val", "Predictions"], loc="lower right")
        image_path = "prediction_plot.png"
        plt.savefig(image_path)
        plt.close()

        return {
            "company": company,
            "validation_table": valid,
            "plot_image": f"/image/{image_path}",
        }
    except Exception as e:
        return {"message": f"Error occurred: {str(e)}"}


@app.get("/image/{filename}")
async def serve_image(filename: str):
    return FileResponse(f'{filename}')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
