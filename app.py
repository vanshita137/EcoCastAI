import streamlit as st
import pandas as pd


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="EcoCast AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Main title */
.main-title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 0px;
}

.subtitle {
    font-size: 18px;
    color: #666666;
    margin-bottom: 25px;
}

/* Cards */
.info-card {
    padding: 20px;
    border-radius: 15px;
    background-color: white;
    border: 1px solid #e5e7eb;
    margin-bottom: 15px;
}

/* Risk boxes */
.success-box {
    padding: 18px;
    border-radius: 12px;
    background-color: #d1e7dd;
    border: 1px solid #a3cfbb;
    margin-bottom: 20px;
}

.warning-box {
    padding: 18px;
    border-radius: 12px;
    background-color: #fff3cd;
    border: 1px solid #ffe69c;
    margin-bottom: 20px;
}

.danger-box {
    padding: 18px;
    border-radius: 12px;
    background-color: #f8d7da;
    border: 1px solid #f1aeb5;
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    air_quality = pd.read_csv(
        "data/final_air_quality.csv",
        parse_dates=["Timestamp"]
    )

    forecast = pd.read_csv(
        "data/forecast_24_hours.csv",
        parse_dates=["Timestamp"]
    )

    return air_quality, forecast


df, forecast = load_data()


# =========================================================
# AIR QUALITY FUNCTIONS
# =========================================================

def air_quality_status(value):

    if value <= 30:
        return "Good", "🟢"

    elif value <= 60:
        return "Moderate", "🟡"

    elif value <= 90:
        return "Unhealthy", "🟠"

    else:
        return "Poor", "🔴"


def risk_level(value):

    if value <= 30:
        return "Low Risk", "🟢"

    elif value <= 60:
        return "Moderate Risk", "🟡"

    elif value <= 90:
        return "High Risk", "🟠"

    else:
        return "Very High Risk", "🔴"


# =========================================================
# CALCULATE VALUES
# =========================================================

latest_pm25 = df["PM2.5"].iloc[-1]

historical_average = df["PM2.5"].mean()

forecast_average = forecast["Predicted_PM2.5"].mean()

forecast_min = forecast["Predicted_PM2.5"].min()

forecast_max = forecast["Predicted_PM2.5"].max()

current_status, current_icon = air_quality_status(
    latest_pm25
)

current_risk, current_risk_icon = risk_level(
    latest_pm25
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🌍 EcoCast AI")

    st.caption("Air Quality Intelligence")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Overview",
            "📈 Historical Analysis",
            "🔮 24-Hour Forecast",
            "🤖 Model Performance"
        ]
    )

    st.divider()

    st.info(
        "EcoCast AI uses an LSTM deep learning model "
        "to forecast PM2.5 levels."
    )

    st.caption(
        "Deep Learning • Time Series • Forecasting"
    )


# =========================================================
# OVERVIEW PAGE
# =========================================================

if page == "🏠 Overview":

    st.markdown(
        '<div class="main-title">🌍 EcoCast AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Deep Learning-Based PM2.5 Air Quality Forecasting'
        '</div>',
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # Current Air Quality Status
    # -----------------------------------------------------

    if latest_pm25 <= 60:

        st.markdown(
            f"""
            <div class="success-box">
                <h3>{current_icon} Current Air Quality: {current_status}</h3>
                <p>
                Current PM2.5 concentration:
                <b>{latest_pm25:.2f}</b>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    elif latest_pm25 <= 90:

        st.markdown(
            f"""
            <div class="warning-box">
                <h3>{current_icon} Air Quality Alert: {current_status}</h3>
                <p>
                Current PM2.5 concentration:
                <b>{latest_pm25:.2f}</b>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="danger-box">
                <h3>{current_icon} Air Quality Alert: {current_status}</h3>
                <p>
                Current PM2.5 concentration:
                <b>{latest_pm25:.2f}</b>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


    # -----------------------------------------------------
    # Main Metrics
    # -----------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Current PM2.5",
            f"{latest_pm25:.2f}"
        )

    with col2:

        st.metric(
            "Historical Average",
            f"{historical_average:.2f}"
        )

    with col3:

        st.metric(
            "Forecast Average",
            f"{forecast_average:.2f}"
        )

    with col4:

        st.metric(
            "Forecast Maximum",
            f"{forecast_max:.2f}"
        )


    st.divider()


    # -----------------------------------------------------
    # PM2.5 Level Indicator
    # -----------------------------------------------------

    st.subheader("🌡️ Current PM2.5 Level")

    progress_value = min(
        max(int(latest_pm25), 0),
        100
    )

    st.progress(progress_value)

    st.caption(
        f"Current concentration: {latest_pm25:.2f}"
    )


    # -----------------------------------------------------
    # Risk Level
    # -----------------------------------------------------

    st.subheader("⚠️ Current Risk Level")

    st.markdown(
        f"""
        <div class="info-card">
            <h2>{current_risk_icon} {current_risk}</h2>
            <p>
            Based on the current PM2.5 concentration.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # Recent Historical Trend
    # -----------------------------------------------------

    st.subheader("📈 Recent PM2.5 Trend")

    recent_data = df.tail(168)

    recent_chart = recent_data[
        ["Timestamp", "PM2.5"]
    ].set_index("Timestamp")

    st.line_chart(
        recent_chart,
        height=350
    )


    # -----------------------------------------------------
    # Quick Forecast
    # -----------------------------------------------------

    st.subheader("🔮 Next 24 Hours Forecast")

    forecast_chart = forecast[
        ["Timestamp", "Predicted_PM2.5"]
    ].set_index("Timestamp")

    st.line_chart(
        forecast_chart,
        height=350
    )


# =========================================================
# HISTORICAL ANALYSIS PAGE
# =========================================================

elif page == "📈 Historical Analysis":

    st.title("📈 Historical Air Quality Analysis")

    st.write(
        "Explore PM2.5 concentration throughout the "
        "historical dataset."
    )


    # -----------------------------------------------------
    # Date Selection
    # -----------------------------------------------------

    min_date = df["Timestamp"].min().date()

    max_date = df["Timestamp"].max().date()

    selected_dates = st.date_input(
        "Select date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )


    if len(selected_dates) == 2:

        start_date = pd.Timestamp(
            selected_dates[0]
        )

        end_date = pd.Timestamp(
            selected_dates[1]
        ) + pd.Timedelta(days=1)

        filtered_df = df[
            (df["Timestamp"] >= start_date)
            &
            (df["Timestamp"] < end_date)
        ]

    else:

        filtered_df = df


    # -----------------------------------------------------
    # Records
    # -----------------------------------------------------

    st.metric(
        "Records in Selected Period",
        len(filtered_df)
    )


    # -----------------------------------------------------
    # Historical Chart
    # -----------------------------------------------------

    chart_data = filtered_df[
        ["Timestamp", "PM2.5"]
    ].set_index("Timestamp")

    st.line_chart(
        chart_data,
        height=450
    )


    # -----------------------------------------------------
    # Statistics
    # -----------------------------------------------------

    st.subheader("📊 Statistical Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Minimum",
            f"{filtered_df['PM2.5'].min():.2f}"
        )

    with col2:

        st.metric(
            "Maximum",
            f"{filtered_df['PM2.5'].max():.2f}"
        )

    with col3:

        st.metric(
            "Average",
            f"{filtered_df['PM2.5'].mean():.2f}"
        )

    with col4:

        st.metric(
            "Median",
            f"{filtered_df['PM2.5'].median():.2f}"
        )


# =========================================================
# 24-HOUR FORECAST PAGE
# =========================================================

elif page == "🔮 24-Hour Forecast":

    st.title("🔮 Next 24 Hours PM2.5 Forecast")

    st.write(
        "Predictions generated using the trained "
        "LSTM deep learning model."
    )


    # -----------------------------------------------------
    # Forecast Chart
    # -----------------------------------------------------

    forecast_chart = forecast[
        ["Timestamp", "Predicted_PM2.5"]
    ].set_index("Timestamp")

    st.line_chart(
        forecast_chart,
        height=450
    )


    # -----------------------------------------------------
    # Forecast Metrics
    # -----------------------------------------------------

    st.subheader("📊 Forecast Overview")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Minimum",
            f"{forecast_min:.2f}"
        )

    with col2:

        st.metric(
            "Average",
            f"{forecast_average:.2f}"
        )

    with col3:

        st.metric(
            "Maximum",
            f"{forecast_max:.2f}"
        )


    # -----------------------------------------------------
    # Forecast Alert
    # -----------------------------------------------------

    if forecast_max > 90:

        st.error(
            "🔴 Very high PM2.5 levels are predicted "
            "during the next 24 hours."
        )

    elif forecast_max > 60:

        st.warning(
            "🟠 Elevated PM2.5 levels are predicted "
            "during the next 24 hours."
        )

    else:

        st.success(
            "🟢 No highly elevated PM2.5 level detected "
            "in the forecast."
        )


    # -----------------------------------------------------
    # Hour-by-Hour Risk Analysis
    # -----------------------------------------------------

    st.subheader("⚠️ Hour-by-Hour Risk Analysis")

    risk_data = []

    for _, row in forecast.iterrows():

        value = row["Predicted_PM2.5"]

        risk, icon = risk_level(value)

        risk_data.append({
            "Time": row["Timestamp"].strftime(
                "%Y-%m-%d %H:%M"
            ),
            "Predicted PM2.5": round(value, 2),
            "Risk Level": f"{icon} {risk}"
        })


    risk_df = pd.DataFrame(risk_data)

    st.dataframe(
        risk_df,
        use_container_width=True,
        hide_index=True
    )


    # -----------------------------------------------------
    # Forecast Table
    # -----------------------------------------------------

    st.subheader("🕐 Hourly Predictions")

    display_forecast = forecast.copy()

    display_forecast["Timestamp"] = (
        display_forecast["Timestamp"]
        .dt.strftime("%Y-%m-%d %H:%M")
    )

    display_forecast["Predicted_PM2.5"] = (
        display_forecast["Predicted_PM2.5"]
        .round(2)
    )

    st.dataframe(
        display_forecast,
        use_container_width=True,
        hide_index=True
    )


    # -----------------------------------------------------
    # CSV Download
    # -----------------------------------------------------

    csv = forecast.to_csv(
        index=False
    )

    st.download_button(
        label="⬇️ Download Forecast CSV",
        data=csv,
        file_name="ecocast_24_hour_forecast.csv",
        mime="text/csv"
    )


    # -----------------------------------------------------
    # Text Report
    # -----------------------------------------------------

    st.subheader("📄 Generate Report")

    report = f"""
EcoCast AI - Air Quality Forecast Report
=========================================

Current PM2.5: {latest_pm25:.2f}

Current Air Quality: {current_status}

Current Risk Level: {current_risk}

Historical Average PM2.5: {historical_average:.2f}


NEXT 24 HOURS FORECAST
----------------------

Minimum PM2.5: {forecast_min:.2f}

Average PM2.5: {forecast_average:.2f}

Maximum PM2.5: {forecast_max:.2f}


MODEL PERFORMANCE
-----------------

MAE: 5.79

RMSE: 6.91

Model: LSTM

Input Window: Previous 24 Hours

Forecast Horizon: Next 24 Hours


=========================================
Generated by EcoCast AI
"""

    st.download_button(
        label="📄 Download Air Quality Report",
        data=report,
        file_name="EcoCast_AI_Report.txt",
        mime="text/plain"
    )


# =========================================================
# MODEL PERFORMANCE PAGE
# =========================================================

elif page == "🤖 Model Performance":

    st.title("🤖 LSTM Model Performance")

    st.write(
        "Performance of the trained model on unseen "
        "test data."
    )


    # -----------------------------------------------------
    # Performance Metrics
    # -----------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "MAE",
            "5.79"
        )

        st.caption(
            "Mean Absolute Error"
        )

    with col2:

        st.metric(
            "RMSE",
            "6.91"
        )

        st.caption(
            "Root Mean Squared Error"
        )


    st.divider()


    # -----------------------------------------------------
    # Model Architecture
    # -----------------------------------------------------

    st.subheader("🧠 Model Architecture")

    architecture = pd.DataFrame({

        "Layer": [
            "Input",
            "LSTM",
            "Dropout",
            "LSTM",
            "Dropout",
            "Dense",
            "Output"
        ],

        "Configuration": [
            "24 time steps × 1 feature",
            "64 units",
            "0.2",
            "32 units",
            "0.2",
            "16 units + ReLU",
            "1 unit"
        ]

    })


    st.dataframe(
        architecture,
        use_container_width=True,
        hide_index=True
    )


    # -----------------------------------------------------
    # Training Configuration
    # -----------------------------------------------------

    st.subheader("⚙️ Training Configuration")

    st.write("""
    **Model:** LSTM

    **Input:** Previous 24 hours of PM2.5

    **Prediction:** Next hour PM2.5

    **Forecasting:** Recursive 24-hour prediction

    **Optimizer:** Adam

    **Loss Function:** Mean Squared Error

    **Batch Size:** 32

    **Maximum Epochs:** 30

    **Early Stopping:** Enabled
    """)


    # -----------------------------------------------------
    # Explanation
    # -----------------------------------------------------

    st.subheader("💡 How EcoCast AI Works")

    st.info("""
    EcoCast AI takes the previous 24 hours of PM2.5
    measurements as input.

    The LSTM model learns temporal patterns from this
    historical sequence and predicts the next PM2.5 value.

    The prediction is then fed back into the model to
    generate a recursive forecast for the next 24 hours.
    """)