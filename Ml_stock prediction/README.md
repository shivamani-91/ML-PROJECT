# AI Stock Price Predictor – LSTM & GRU

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Deep%20Learning-Keras%20%2F%20TensorFlow-orange.svg)](https://keras.io/)
[![Backend](https://img.shields.io/badge/Backend-Flask%20REST%20API-lightgrey.svg)](https://flask.palletsprojects.com/)
[![Frontend](https://img.shields.io/badge/Frontend-Responsive%20Plotly%20Dashboard-blueviolet.svg)](https://plotly.com/javascript/)
[![License](https://img.shields.io/badge/Academic%20Project-B.Tech%20CSE-green.svg)]()

A complete, production-grade **Machine Learning Stock Price Prediction Web Application** built for academic evaluation, research demonstration, and viva voce presentation. The system implements authentic deep learning time-series architectures—**Long Short-Term Memory (LSTM)** and **Gated Recurrent Unit (GRU)**—trained dynamically on historical stock-market datasets.

---

## 📌 Disclaimer
> **Academic & Educational Use Only**: This software is developed solely as an educational machine-learning engineering project. Stock market predictions are based on historical pattern recognition and must **NOT** be used as financial advice or guarantees of future market values.

---

## 📑 Table of Contents
1. [Project Overview](#-project-overview)
2. [Architectural Highlights](#-architectural-highlights)
3. [Deep Learning Theory: LSTM vs GRU](#-deep-learning-theory-lstm-vs-gru)
4. [Machine Learning Pipeline](#-machine-learning-pipeline)
5. [Evaluation Metrics & Formulas](#-evaluation-metrics--formulas)
6. [Folder Structure](#-folder-structure)
7. [REST API Endpoints](#-rest-api-endpoints)
8. [Setup & Running in VS Code](#-setup--running-in-vs-code)
9. [Viva & Oral Defense Q&A](#-viva--oral-defense-qa)

---

## 🎯 Project Overview

Stock market time-series forecasting presents substantial challenges due to non-linear dependencies, regime shifts, and market noise. Traditional econometric models (such as ARIMA or Holt-Winters) presume linear relationships and stationary variances.

This project implements:
1. **Dynamic Data Ingestion**: Automated retrieval of real historical OHLCV data using `yfinance` across major tickers (AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META) or user-uploaded CSV datasets.
2. **Offline Demo Fallback**: Built-in realistic stock simulation dataset ensuring 100% presentation reliability even without an internet connection or if Yahoo Finance experiences rate limits.
3. **True Neural Network Training**: Real TensorFlow / Keras neural networks trained on-demand with live epoch-by-epoch loss updates, rather than static mock values.
4. **Model Benchmark & Comparison**: Side-by-side calculation of MSE, RMSE, MAE, MAPE, and Directional Trend Accuracy with automatic selection of the superior architecture.
5. **Interactive Plotly Visualizations**: Responsive zoom, pan, hover, and overlay comparisons of actual vs predicted closing prices.
6. **Recursive Multi-Step Forecasting**: Forward projection of future stock prices over 1, 5, 7, and 30 business trading days.

---

## 🧠 Deep Learning Theory: LSTM vs GRU

### 1. Long Short-Term Memory (LSTM)
LSTM addresses the vanishing and exploding gradient problem in standard recurrent neural networks by introducing dedicated gating mechanisms and a distinct cell state vector $C_t$:

* **Forget Gate**: Decides what information to discard from the previous cell state:
  $$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$
* **Input Gate & Candidate State**: Determines which new values update the cell state:
  $$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
  $$\tilde{C}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$
* **Cell State Update**:
  $$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$
* **Output Gate & Hidden State**:
  $$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
  $$h_t = o_t \odot \tanh(C_t)$$

### 2. Gated Recurrent Unit (GRU)
GRU simplifies the LSTM architecture by combining the cell state and hidden state, and utilizing two gates:

* **Reset Gate**: Determines how to combine new input with previous memory:
  $$r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)$$
* **Update Gate**: Acts similarly to LSTM's forget and input gates:
  $$z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)$$
* **Candidate Hidden State**:
  $$\tilde{h}_t = \tanh(W_h \cdot [r_t \odot h_{t-1}, x_t] + b_h)$$
* **Final Hidden State**:
  $$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

**Key Trade-off**: GRUs have fewer parameters, train faster, and perform comparably on small-to-medium datasets, whereas LSTMs offer greater expressive capacity on larger sequential datasets.

---

## 🔄 Machine Learning Pipeline

```text
       Yahoo Finance (yfinance) OR Custom CSV Upload
                            ↓
             Raw Historical OHLCV Dataset
                            ↓
         Data Cleaning & Missing Value Handling
               (Forward Fill + Backward Fill)
                            ↓
              Chronological Sorting by Date
                            ↓
             Target Selection: 'Close' Price
                            ↓
         Normalization using MinMaxScaler [0, 1]
         (Fitted strictly on Train partition)
                            ↓
         Time-Series Sliding Window Sequence Creation
                  (20, 30, 60, or 90 Days)
                            ↓
         Chronological Train / Test Split (80% / 20%)
                            ↓
            ┌───────────────────────────────┐
            ↓                               ↓
       LSTM Model                       GRU Model
    (2 Layers + Dropout)            (2 Layers + Dropout)
            ↓                               ↓
    Test Set Prediction             Test Set Prediction
            └───────────────┬───────────────┘
                            ↓
                Objective Model Evaluation
                 (MSE, RMSE, MAE, MAPE, Acc)
                            ↓
                 Automated Model Comparison
                  (Best Architecture Badge)
                            ↓
             Recursive Multi-Step Future Forecast
                    (1, 5, 7, 30 Days)
```

---

## 📊 Evaluation Metrics & Formulas

| Metric | Mathematical Formula | Purpose in Stock Forecasting |
| :--- | :--- | :--- |
| **MSE** (Mean Squared Error) | $\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$ | Quantifies average squared variance. |
| **RMSE** (Root Mean Squared Error) | $\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$ | Standard error in actual dollar currency ($); heavily penalizes outlier errors. |
| **MAE** (Mean Absolute Error) | $\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} \|y_i - \hat{y}_i\|$ | Average absolute price deviation ($). |
| **MAPE** | $\text{MAPE} = \frac{100\%}{n} \sum_{i=1}^{n} \left\|\frac{y_i - \hat{y}_i}{y_i}\right\|$ | Percentage error scale-independent. |

---

## 📁 Folder Structure

```
Ml_stock prediction/
│
├── backend/
│   ├── app.py                      # Flask REST API server & static frontend serving
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── data_processor.py       # Cleaning, MinMaxScaler, sequence generator
│   ├── training/
│   │   ├── __init__.py
│   │   ├── lstm_model.py           # Keras LSTM neural network
│   │   ├── gru_model.py            # Keras GRU neural network
│   │   └── trainer.py              # Asynchronous trainer & epoch callback
│   ├── prediction/
│   │   ├── __init__.py
│   │   └── predictor.py            # Model evaluation & recursive future forecasting
│   ├── dataset/
│   │   ├── __init__.py
│   │   ├── stock_fetcher.py        # yfinance downloader & CSV parser
│   │   └── demo_stock_data.csv     # Bundled offline fallback dataset
│   └── models/
│       ├── lstm_model.keras        # Saved LSTM weights
│       └── gru_model.keras         # Saved GRU weights
│
├── frontend/
│   ├── index.html                  # Responsive UI dashboard
│   ├── styles.css                  # Dark/Light terminal styling & animations
│   └── script.js                   # API integration & interactive Plotly charts
│
├── run.py                          # One-click launcher script
├── requirements.txt                # Python dependencies
└── README.md                       # Comprehensive documentation & viva defense
```

---

## 🌐 REST API Endpoints

| HTTP Method | Endpoint | Parameters / Body | Description |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/stock-data` | `?ticker=AAPL&start=...&end=...` | Fetches historical OHLCV data from Yahoo Finance or demo fallback |
| `POST` | `/api/upload-dataset` | Multipart Form: `file` | Ingests and validates user-uploaded CSV dataset |
| `POST` | `/api/preprocess` | `{"sequence_length": 60}` | Updates sequence window length and regenerates train/test split |
| `POST` | `/api/train/lstm` | Hyperparameters JSON | Initiates asynchronous LSTM model training |
| `POST` | `/api/train/gru` | Hyperparameters JSON | Initiates asynchronous GRU model training |
| `POST` | `/api/train/both` | Hyperparameters JSON | Sequentially trains LSTM followed by GRU |
| `GET` | `/api/train/status` | None | Returns live training state (epoch, loss, val_loss, percent) |
| `GET` | `/api/results` | None | Returns calculated MSE, RMSE, MAE, and test set predictions |
| `POST` | `/api/predict-future` | `{"days": 30, "model": "lstm"}` | Generates recursive multi-step future price forecasts |
| `GET` | `/api/training-history`| None | Returns epoch loss arrays for convergence curves |

---

## 🚀 Setup & Running in VS Code

### Step 1: Clone or Open Directory in VS Code
Open VS Code, select **File > Open Folder**, and navigate to:
```bash
Ml_stock prediction
```

### Step 2: Install Dependencies
Open the VS Code integrated terminal (`Ctrl + ~`) and run:
```bash
pip install -r requirements.txt
```

### Step 3: Launch Application
Run the one-click launcher script:
```bash
python run.py
```
This automatically:
- Validates deep learning packages.
- Starts the Flask REST API server on `http://127.0.0.1:5000`.
- Automatically opens your default web browser to the dashboard!

---

## 🎓 Viva & Oral Defense Q&A

### Q1: Why use LSTM and GRU instead of traditional linear regression or ARIMA?
> **Answer**: Financial time-series data exhibits strong non-linearities, heteroscedasticity (varying volatility), and multi-timescale temporal dependencies. Linear regression presumes independent and identically distributed (i.i.d.) variables, while ARIMA requires stationary series and only captures linear lags. LSTMs and GRUs utilize non-linear gating mechanisms capable of preserving context across extended sequences without suffering from the vanishing gradient problem.

### Q2: Why is the MinMaxScaler fitted ONLY on the training dataset?
> **Answer**: Fitting the scaler on the entire dataset would introduce **Data Leakage** (lookahead bias). In real-world deployment, future test prices are unknown. Transforming test data with parameters ($\mu, \sigma$ or $\min, \max$) calculated exclusively on prior training history preserves genuine out-of-sample evaluation validity.

### Q3: What is the purpose of the sliding window sequence generator?
> **Answer**: Neural networks require fixed-dimension input matrices. The sliding window converts continuous time-series into supervised training pairs $(X, y)$, where input $X$ contains prices from $[t - N, t - 1]$ and target $y$ is the closing price at time $t$.

### Q4: How does recursive future price forecasting work?
> **Answer**: To forecast $k$ days ahead, the trained model predicts day $t+1$ using the last known $N$ days. The predicted value is appended to the sequence, the oldest observation is dropped, and the model predicts $t+2$. This recursive roll repeats until the chosen horizon (1, 5, 7, 30 days) is completed.

### Q5: Why is RMSE preferred over MSE as the primary comparative metric?
> **Answer**: While MSE squares errors, RMSE takes the square root, returning the error magnitude to the original scale of the stock price ($/₹). This makes the performance directly interpretable by financial analysts and traders.

---
**Academic Project Completed by:** G Yuvan Shivamani &bull; Roll Number: 2520030310 &bull; B.Tech Computer Science &bull; Machine Learning
