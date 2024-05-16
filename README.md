
## StockSeer-API  `(Stock Price Prediction API)`
<img src='https://github.com/samadpls/stockseer-api/assets/94792103/dc4f0585-3eaa-4837-a50f-c64375674f41' width=250px align='right'>

![Supported python versions](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) 
[![License](https://img.shields.io/badge/License-MIT%202.0-blue.svg)](LICENSE)

This project is a FastAPI application that predicts the closing stock price for a given company based on user-specified parameters. It utilizes various machine learning models for prediction, including:

* RandomForestRegressor 🌳
* ExtraTreesRegressor 🌲
* XGBRegressor 🚀
* LinearRegression ➖
* KNeighborsRegressor 🤝
* LSTM implementation 🔄

## Data Source:

This application utilizes the Yahoo Finance API to retrieve historical stock data for training and prediction purposes.

## Features✨

* Download and preprocess historical stock data
* Train a stock price prediction model of your choice
* Make predictions on future closing stock prices


## Working Prototype 
**Example: Predicting <img src='https://github.com/samadpls/stockseer-api/assets/94792103/be02a515-e6d3-402e-a6bd-2f7f21065fa6' width=15px>oogle
 Stock Price**

| Input | Response| <img src='https://github.com/samadpls/stockseer-api/assets/94792103/be02a515-e6d3-402e-a6bd-2f7f21065fa6' width=20px> raph |
|------|---------|---------|
|![image](https://github.com/samadpls/stockseer-api/assets/94792103/b6860128-41fb-463d-908d-433b11f3d826)|![image](https://github.com/samadpls/stockseer-api/assets/94792103/5ef2c728-22e3-4e9c-9049-b8b8ceddc276)| <img src='prediction_plot.png' width=650px>|

## Installation

1. Ensure you have Python installed.
2. Create a new virtual environment (recommended).
3. Activate the virtual environment.
4. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:

```bash
uvicorn app:app --reload
```

2. Access the API documentation in your web browser: http://127.0.0.1:8000/docs

The documentation provides instructions on interacting with the API to make predictions.


## Credits

* **Maira Usman:**  Developed the GUI for this project. You can find the code here: Link to [StockSeer-Frontend](https://github.com/Myrausman/StockSeer-Frontend).

## Disclaimer

**Important:** Stock price prediction is inherently uncertain. This application should not be used for making financial decisions. 
