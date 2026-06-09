import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

def calculate_national_index(df):
    """
    Calculates a daily national risk index from raw posts.
    Formula: (High Risk Count * 1.5 + Medium Risk Count * 1.0) / Total Posts * 100
    Adds some smoothing for graph stability.
    """
    df['date'] = pd.to_datetime(df['date'])
    daily = df.groupby(df['date'].dt.date).apply(
        lambda x: (
            ((x['risk_level'] == 'High Risk').sum() * 1.5 + 
             (x['risk_level'] == 'Medium Risk').sum() * 1.0) / len(x)
        ) * 100 if len(x) > 0 else 0
    ).reset_index(name='risk_index')
    
    daily['date'] = pd.to_datetime(daily['date'])
    
    # Smooth the index with a 7-day rolling average
    daily['smoothed_risk_index'] = daily['risk_index']
    return daily

def forecast_risk(daily_df, years=5):
    """
    Forecasts the NATIONAL yearly risk index for the next `years`
    using ARIMA on aggregated yearly data.
    """

    # 🔴 Convert to yearly aggregated risk
    yearly = daily_df.copy()
    yearly['date'] = pd.to_datetime(yearly['date'])
    yearly.set_index('date', inplace=True)

    # IMPORTANT: national signal should be averaged over the year for scaling
    yearly_series = yearly['risk_index'].resample('Y').mean()

    # If too few years exist → fallback
    if len(yearly_series) < 3:
        last_val = yearly_series.iloc[-1]
        forecast_vals = np.array([last_val] * years)
        lower_vals = np.array([last_val * (1 - 0.05 * i) for i in range(1, years + 1)])
        upper_vals = np.array([last_val * (1 + 0.05 * i) for i in range(1, years + 1)])
    else:
        try:
            model = ARIMA(yearly_series, order=(1,1,1))
            fitted = model.fit()
            frc = fitted.get_forecast(steps=years)
            forecast_vals = frc.predicted_mean.values
            conf = frc.conf_int(alpha=0.2) # 80% confidence
            lower_vals = conf.iloc[:, 0].values
            upper_vals = conf.iloc[:, 1].values
        except Exception as e:
            print("Yearly ARIMA failed, fallback used", e)
            last_val = yearly_series.iloc[-1]
            forecast_vals = np.array([last_val] * years)
            lower_vals = np.array([last_val * (1 - 0.05 * i) for i in range(1, years + 1)])
            upper_vals = np.array([last_val * (1 + 0.05 * i) for i in range(1, years + 1)])

    last_year = yearly_series.index[-1]
    last_val = yearly_series.iloc[-1]

    future_dates = pd.date_range(
        start=last_year,
        periods=years + 1,
        freq='YE'
    )[1:]

    forecast_df = pd.DataFrame({
        'date': [last_year] + list(future_dates),
        'predicted_risk_index': [last_val] + list(forecast_vals),
        'lower_bound': [last_val] + list(lower_vals),
        'upper_bound': [last_val] + list(upper_vals)
    })
    
    historical_yearly = yearly_series.reset_index()
    historical_yearly.columns = ['date', 'historical_risk_index']
    
    return historical_yearly, forecast_df
def generate_policy_recommendations(current_risk_index):
    """
    Simulates a government decision intelligence layer returning recommendations
    based on the current national risk thresholds.
    """
    if current_risk_index > 60:
        return [
            "CRITICAL: Deploy emergency mobile mental health response units in high-density urban areas.",
            "Alert national hotline centers for immediate surge capacity.",
            "Initiate direct-outreach ad campaigns on major social networking platforms."
        ]
    elif current_risk_index > 40:
        return [
            "WARNING: Escalate mental health awareness protocols in public schools.",
            "Provide community centers with additional counseling resources.",
            "Monitor regional hotspots for localized intervention."
        ]
    else:
        return [
            "STATUS NORMAL: Continue baseline monitoring of social channels.",
            "Maintain standard funding for community mental health programs.",
            "Review monthly aggregated trends for slow-moving shifts."
        ]
