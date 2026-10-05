# 🌍 EcoCast AI

### Deep Learning-Based PM2.5 Air Quality Forecasting

EcoCast AI is a deep learning-based air quality forecasting application that uses historical PM2.5 data and an LSTM (Long Short-Term Memory) neural network to predict PM2.5 levels for the next 24 hours.

The project provides an interactive Streamlit dashboard for exploring historical air quality data, viewing forecasts, and understanding model performance.

---

## 🚀 Live Demo

🔗 **Live Application:**  
[Open EcoCast AI](https://ecocastai-onent9opyvxsozjxtiqpcm.streamlit.app/)


---

## 🎯 Problem Statement

Air pollution changes over time and can reach unhealthy levels unexpectedly.

Traditional monitoring systems mainly show current or historical pollution levels. EcoCast AI aims to go one step further by using historical PM2.5 patterns to forecast future pollution levels.

### Objective

To develop a deep learning system that:

- Analyzes historical PM2.5 data
- Identifies temporal patterns
- Uses an LSTM model for time-series forecasting
- Predicts PM2.5 levels for the next 24 hours
- Provides an interactive visualization dashboard

---

## 📊 Dataset

The project uses an air-quality dataset containing historical PM2.5 measurements.

### Dataset Information

| Feature | Description |
|---|---|
| Timestamp | Date and time of observation |
| Year | Year of observation |
| Month | Month of observation |
| Day | Day of observation |
| Hour | Hour of observation |
| PM2.5 | PM2.5 concentration |

### Dataset Size

- Original records: **36,192**
- Original features: **6**
- Final continuous hourly records: **40,084**

---

## 🔄 Project Workflow

```text
Historical PM2.5 Data
        ↓
Data Cleaning
        ↓
Timestamp Processing
        ↓
Missing Time Gap Detection
        ↓
Hourly Timeline Creation
        ↓
Interpolation
        ↓
Time-Series Sequence Creation
        ↓
Train/Test Split
        ↓
Data Scaling
        ↓
LSTM Model Training
        ↓
Model Evaluation
        ↓
24-Hour Forecast
        ↓
Streamlit Dashboard