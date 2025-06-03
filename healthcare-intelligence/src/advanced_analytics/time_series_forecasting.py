"""
Healthcare Time Series Forecasting
Prophet-based cost prediction and seasonal fraud pattern analysis
Advanced analytics module for fraud detection platform
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import base model class for consistency
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.base_healthcare_model import BaseHealthcareModel

# Try to import Prophet, with fallback if not available
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    print("Warning: Prophet not installed. Time series forecasting will use ARIMA fallback.")
    PROPHET_AVAILABLE = False
    
# Alternative time series imports
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

class HealthcareCostForecaster(BaseHealthcareModel):
    """
    Time series forecasting for healthcare costs and fraud patterns
    Supports both Prophet and ARIMA models with healthcare-specific seasonality
    """
    
    def __init__(self):
        super().__init__("CostForecasting", "Advanced_Analytics")
        self.prophet_model = None
        self.arima_model = None
        self.forecast_data = None
        self.seasonal_patterns = {}
        self.business_calendar = self._create_healthcare_calendar()
        
    def _create_healthcare_calendar(self) -> pd.DataFrame:
        """
        Create healthcare-specific business calendar with billing cycles
        
        Returns:
            DataFrame with healthcare business calendar
        """
        # Healthcare billing and regulatory calendar
        calendar_events = {
            'year_end_billing': ['2023-12-31', '2024-12-31', '2025-12-31'],
            'quarter_end': ['2023-03-31', '2023-06-30', '2023-09-30', '2023-12-31',
                           '2024-03-31', '2024-06-30', '2024-09-30', '2024-12-31'],
            'medicare_payment_cycles': ['2023-01-15', '2023-02-15', '2023-03-15',
                                      '2024-01-15', '2024-02-15', '2024-03-15'],
            'compliance_reporting': ['2023-04-15', '2023-07-15', '2023-10-15',
                                   '2024-04-15', '2024-07-15', '2024-10-15']
        }
        
        return calendar_events
    
    def prepare_time_series_data(self, claims_df: pd.DataFrame, freq: str = 'D') -> pd.DataFrame:
        """
        Prepare claims data for time series analysis
        
        Args:
            claims_df: Claims data with dates and amounts
            freq: Frequency for aggregation ('D' for daily, 'W' for weekly, 'M' for monthly)
            
        Returns:
            DataFrame with time series data
        """
        self.logger.info(f"Preparing time series data with frequency: {freq}")
        
        try:
            # Ensure claim_date is datetime
            if 'claim_date' not in claims_df.columns:
                # Generate synthetic dates if not present
                start_date = datetime(2023, 1, 1)
                claims_df['claim_date'] = [
                    start_date + timedelta(days=np.random.randint(0, 365))
                    for _ in range(len(claims_df))
                ]
            
            claims_df['claim_date'] = pd.to_datetime(claims_df['claim_date'])
            
            # Aggregate by time period
            if freq == 'D':
                time_series = claims_df.groupby(claims_df['claim_date'].dt.date).agg({
                    'claim_amount': ['sum', 'count', 'mean'],
                    'provider_id': 'nunique'
                }).round(2)
            elif freq == 'W':
                time_series = claims_df.groupby(pd.Grouper(key='claim_date', freq='W')).agg({
                    'claim_amount': ['sum', 'count', 'mean'],
                    'provider_id': 'nunique'
                }).round(2)
            elif freq == 'M':
                time_series = claims_df.groupby(pd.Grouper(key='claim_date', freq='M')).agg({
                    'claim_amount': ['sum', 'count', 'mean'],
                    'provider_id': 'nunique'
                }).round(2)
            
            # Flatten column names
            time_series.columns = [
                'total_amount', 'claim_count', 'avg_amount', 'unique_providers'
            ]
            
            # Reset index to make date a column
            time_series = time_series.reset_index()
            time_series.columns = ['ds', 'y', 'claim_count', 'avg_amount', 'unique_providers']
            
            # Remove any rows with zero amounts
            time_series = time_series[time_series['y'] > 0]
            
            # Sort by date
            time_series = time_series.sort_values('ds').reset_index(drop=True)
            
            self.logger.info(f"Prepared time series data: {len(time_series)} periods")
            
            return time_series
            
        except Exception as e:
            self.logger.error(f"Error preparing time series data: {str(e)}")
            raise
    
    def analyze_seasonal_patterns(self, time_series_df: pd.DataFrame) -> dict:
        """
        Analyze seasonal patterns in healthcare claims data
        
        Args:
            time_series_df: Time series data
            
        Returns:
            Dictionary with seasonal analysis results
        """
        self.logger.info("Analyzing seasonal patterns in healthcare data")
        
        try:
            # Set date as index for seasonal decomposition
            ts_indexed = time_series_df.set_index('ds')
            
            # Seasonal decomposition
            if len(ts_indexed) >= 24:  # Need sufficient data points
                decomposition = seasonal_decompose(
                    ts_indexed['y'], 
                    model='additive', 
                    period=12 if len(ts_indexed) >= 24 else len(ts_indexed)//2
                )
                
                seasonal_strength = np.var(decomposition.seasonal) / np.var(ts_indexed['y'])
                trend_strength = np.var(decomposition.trend.dropna()) / np.var(ts_indexed['y'])
            else:
                seasonal_strength = 0
                trend_strength = 0
            
            # Monthly patterns (if enough data)
            monthly_patterns = {}
            if len(time_series_df) >= 12:
                time_series_df['month'] = pd.to_datetime(time_series_df['ds']).dt.month
                monthly_avg = time_series_df.groupby('month')['y'].mean()
                monthly_patterns = monthly_avg.to_dict()
            
            # Weekly patterns (if enough data)
            weekly_patterns = {}
            if len(time_series_df) >= 7:
                time_series_df['dayofweek'] = pd.to_datetime(time_series_df['ds']).dt.dayofweek
                weekly_avg = time_series_df.groupby('dayofweek')['y'].mean()
                weekly_patterns = weekly_avg.to_dict()
            
            # Identify peak periods
            peak_periods = time_series_df.nlargest(5, 'y')[['ds', 'y']].to_dict('records')
            
            seasonal_analysis = {
                'seasonal_strength': round(seasonal_strength, 3),
                'trend_strength': round(trend_strength, 3),
                'monthly_patterns': monthly_patterns,
                'weekly_patterns': weekly_patterns,
                'peak_periods': peak_periods,
                'total_variation': round(np.var(time_series_df['y']), 2),
                'period_count': len(time_series_df)
            }
            
            self.seasonal_patterns = seasonal_analysis
            
            self.logger.info(f"Seasonal analysis complete. Seasonal strength: {seasonal_strength:.3f}")
            
            return seasonal_analysis
            
        except Exception as e:
            self.logger.error(f"Error in seasonal analysis: {str(e)}")
            raise
    
    def create_prophet_forecast(self, time_series_df: pd.DataFrame, periods: int = 30) -> dict:
        """
        Create Prophet-based forecast for healthcare costs
        
        Args:
            time_series_df: Time series data with 'ds' and 'y' columns
            periods: Number of periods to forecast
            
        Returns:
            Dictionary with forecast results
        """
        if not PROPHET_AVAILABLE:
            return self.create_arima_forecast(time_series_df, periods)
        
        self.logger.info(f"Creating Prophet forecast for {periods} periods")
        
        try:
            # Initialize Prophet with healthcare-specific parameters
            self.prophet_model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05,  # Conservative for healthcare
                seasonality_prior_scale=10.0,   # Allow seasonal patterns
                interval_width=0.80             # 80% confidence intervals
            )
            
            # Add healthcare-specific seasonalities
            if len(time_series_df) >= 30:
                self.prophet_model.add_seasonality(
                    name='monthly',
                    period=30.5,
                    fourier_order=5
                )
            
            if len(time_series_df) >= 90:
                self.prophet_model.add_seasonality(
                    name='quarterly',
                    period=91.25,
                    fourier_order=3
                )
            
            # Fit the model
            self.prophet_model.fit(time_series_df)
            
            # Create future dataframe
            future = self.prophet_model.make_future_dataframe(periods=periods)
            
            # Generate forecast
            forecast = self.prophet_model.predict(future)
            
            # Calculate forecast metrics
            forecast_period = forecast.tail(periods)
            historical_mean = time_series_df['y'].mean()
            forecast_mean = forecast_period['yhat'].mean()
            
            forecast_results = {
                'model_type': 'Prophet',
                'forecast_data': forecast,
                'forecast_period': forecast_period,
                'historical_mean': round(historical_mean, 2),
                'forecast_mean': round(forecast_mean, 2),
                'change_percentage': round(((forecast_mean - historical_mean) / historical_mean) * 100, 1),
                'forecast_periods': periods,
                'confidence_intervals': {
                    'lower': forecast_period['yhat_lower'].tolist(),
                    'upper': forecast_period['yhat_upper'].tolist()
                }
            }
            
            self.forecast_data = forecast_results
            
            self.logger.info(f"Prophet forecast complete. Mean forecast: {forecast_mean:.2f}")
            
            return forecast_results
            
        except Exception as e:
            self.logger.error(f"Error in Prophet forecasting: {str(e)}")
            # Fallback to ARIMA
            return self.create_arima_forecast(time_series_df, periods)
    
    def create_arima_forecast(self, time_series_df: pd.DataFrame, periods: int = 30) -> dict:
        """
        Create ARIMA-based forecast as fallback
        
        Args:
            time_series_df: Time series data
            periods: Number of periods to forecast
            
        Returns:
            Dictionary with forecast results
        """
        self.logger.info(f"Creating ARIMA forecast for {periods} periods")
        
        try:
            # Prepare data
            y_series = time_series_df['y'].values
            
            # Simple ARIMA model (1,1,1) for healthcare data
            self.arima_model = ARIMA(y_series, order=(1, 1, 1))
            fitted_model = self.arima_model.fit()
            
            # Generate forecast
            forecast_result = fitted_model.forecast(steps=periods)
            
            # Create confidence intervals (simple approach)
            forecast_std = np.std(y_series[-10:])  # Use recent volatility
            lower_ci = forecast_result - 1.96 * forecast_std
            upper_ci = forecast_result + 1.96 * forecast_std
            
            # Create forecast dataframe
            last_date = pd.to_datetime(time_series_df['ds'].iloc[-1])
            future_dates = [last_date + timedelta(days=i+1) for i in range(periods)]
            
            forecast_df = pd.DataFrame({
                'ds': future_dates,
                'yhat': forecast_result,
                'yhat_lower': lower_ci,
                'yhat_upper': upper_ci
            })
            
            # Calculate metrics
            historical_mean = time_series_df['y'].mean()
            forecast_mean = forecast_result.mean()
            
            forecast_results = {
                'model_type': 'ARIMA',
                'forecast_data': forecast_df,
                'forecast_period': forecast_df,
                'historical_mean': round(historical_mean, 2),
                'forecast_mean': round(forecast_mean, 2),
                'change_percentage': round(((forecast_mean - historical_mean) / historical_mean) * 100, 1),
                'forecast_periods': periods,
                'confidence_intervals': {
                    'lower': lower_ci.tolist(),
                    'upper': upper_ci.tolist()
                }
            }
            
            self.forecast_data = forecast_results
            
            self.logger.info(f"ARIMA forecast complete. Mean forecast: {forecast_mean:.2f}")
            
            return forecast_results
            
        except Exception as e:
            self.logger.error(f"Error in ARIMA forecasting: {str(e)}")
            raise
    
    def create_forecast_visualizations(self, time_series_df: pd.DataFrame, forecast_results: dict) -> dict:
        """
        Create comprehensive visualizations for time series forecast
        
        Args:
            time_series_df: Historical time series data
            forecast_results: Forecast results
            
        Returns:
            Dictionary with visualization objects
        """
        self.logger.info("Creating forecast visualizations")
        
        visualizations = {}
        
        try:
            # 1. Main forecast plot
            fig_forecast = go.Figure()
            
            # Historical data
            fig_forecast.add_trace(go.Scatter(
                x=time_series_df['ds'],
                y=time_series_df['y'],
                mode='lines+markers',
                name='Historical',
                line=dict(color='blue')
            ))
            
            # Forecast data
            forecast_data = forecast_results['forecast_period']
            fig_forecast.add_trace(go.Scatter(
                x=forecast_data['ds'] if 'ds' in forecast_data.columns else range(len(forecast_data)),
                y=forecast_data['yhat'],
                mode='lines+markers',
                name='Forecast',
                line=dict(color='red', dash='dash')
            ))
            
            # Confidence intervals
            if 'yhat_lower' in forecast_data.columns:
                fig_forecast.add_trace(go.Scatter(
                    x=forecast_data['ds'] if 'ds' in forecast_data.columns else range(len(forecast_data)),
                    y=forecast_data['yhat_upper'],
                    mode='lines',
                    name='Upper CI',
                    line=dict(color='rgba(255,0,0,0)'),
                    showlegend=False
                ))
                
                fig_forecast.add_trace(go.Scatter(
                    x=forecast_data['ds'] if 'ds' in forecast_data.columns else range(len(forecast_data)),
                    y=forecast_data['yhat_lower'],
                    mode='lines',
                    name='Confidence Interval',
                    fill='tonexty',
                    fillcolor='rgba(255,0,0,0.2)',
                    line=dict(color='rgba(255,0,0,0)')
                ))
            
            fig_forecast.update_layout(
                title=f"Healthcare Cost Forecast ({forecast_results['model_type']})",
                xaxis_title="Date",
                yaxis_title="Total Claims Amount",
                hovermode='x unified'
            )
            
            visualizations['forecast_plot'] = fig_forecast
            
            # 2. Seasonal patterns (if available)
            if self.seasonal_patterns and 'monthly_patterns' in self.seasonal_patterns:
                monthly_data = self.seasonal_patterns['monthly_patterns']
                if monthly_data:
                    fig_seasonal = px.bar(
                        x=list(monthly_data.keys()),
                        y=list(monthly_data.values()),
                        title="Monthly Healthcare Claims Patterns",
                        labels={'x': 'Month', 'y': 'Average Claims Amount'}
                    )
                    visualizations['seasonal_patterns'] = fig_seasonal
            
            # 3. Forecast summary metrics
            metrics_text = f"""
            <b>Forecast Summary:</b><br>
            Model: {forecast_results['model_type']}<br>
            Historical Average: ${forecast_results['historical_mean']:,.2f}<br>
            Forecast Average: ${forecast_results['forecast_mean']:,.2f}<br>
            Change: {forecast_results['change_percentage']:+.1f}%<br>
            Forecast Periods: {forecast_results['forecast_periods']}
            """
            
            visualizations['summary_metrics'] = metrics_text
            
            self.logger.info(f"Created {len(visualizations)} forecast visualizations")
            
            return visualizations
            
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {str(e)}")
            return {'error': str(e)}
    
    def calculate_business_forecast_impact(self, forecast_results: dict, risk_threshold: float = 1.2) -> dict:
        """
        Calculate business impact of cost forecasting
        
        Args:
            forecast_results: Forecast results
            risk_threshold: Threshold for identifying concerning cost increases
            
        Returns:
            Dictionary with business impact analysis
        """
        self.logger.info("Calculating business impact of cost forecasting")
        
        try:
            historical_mean = forecast_results['historical_mean']
            forecast_mean = forecast_results['forecast_mean']
            change_pct = forecast_results['change_percentage']
            
            # Risk assessment
            risk_level = "Low"
            if forecast_mean > historical_mean * risk_threshold:
                risk_level = "High"
            elif forecast_mean > historical_mean * 1.1:
                risk_level = "Medium"
            
            # Cost impact calculations
            annual_forecast = forecast_mean * 365  # Assume daily data
            annual_historical = historical_mean * 365
            annual_difference = annual_forecast - annual_historical
            
            # Investigation and prevention costs
            if change_pct > 20:  # Significant increase
                investigation_budget = abs(annual_difference) * 0.05  # 5% of increase
                prevention_investment = abs(annual_difference) * 0.03  # 3% of increase
            else:
                investigation_budget = historical_mean * 365 * 0.02  # 2% of baseline
                prevention_investment = historical_mean * 365 * 0.01  # 1% of baseline
            
            business_impact = {
                'risk_assessment': {
                    'risk_level': risk_level,
                    'change_percentage': change_pct,
                    'risk_threshold_exceeded': forecast_mean > historical_mean * risk_threshold
                },
                'financial_projections': {
                    'annual_historical_cost': round(annual_historical, 2),
                    'annual_forecast_cost': round(annual_forecast, 2),
                    'annual_difference': round(annual_difference, 2),
                    'monthly_forecast_cost': round(forecast_mean * 30, 2)
                },
                'recommended_actions': {
                    'investigation_budget': round(investigation_budget, 2),
                    'prevention_investment': round(prevention_investment, 2),
                    'monitoring_frequency': 'Weekly' if risk_level == 'High' else 'Monthly'
                },
                'strategic_insights': {
                    'trend_direction': 'Increasing' if change_pct > 5 else 'Decreasing' if change_pct < -5 else 'Stable',
                    'volatility_assessment': 'High' if len(forecast_results['confidence_intervals']['upper']) > 0 and 
                                           np.std(forecast_results['confidence_intervals']['upper']) > forecast_mean * 0.2 else 'Normal',
                    'planning_horizon': f"{forecast_results['forecast_periods']} periods"
                }
            }
            
            self.logger.info(f"Business impact analysis complete. Risk level: {risk_level}")
            
            return business_impact
            
        except Exception as e:
            self.logger.error(f"Error calculating business impact: {str(e)}")
            raise

# Integration function for main analytics pipeline
def run_time_series_analysis(claims_df: pd.DataFrame, forecast_periods: int = 30) -> tuple:
    """
    Complete time series analysis pipeline
    
    Args:
        claims_df: Claims data
        forecast_periods: Number of periods to forecast
        
    Returns:
        Tuple of (forecaster, time_series_data, forecast_results, visualizations, business_impact)
    """
    # Initialize forecaster
    forecaster = HealthcareCostForecaster()
    
    # Prepare time series data
    time_series_data = forecaster.prepare_time_series_data(claims_df, freq='D')
    
    # Analyze seasonal patterns
    seasonal_analysis = forecaster.analyze_seasonal_patterns(time_series_data)
    
    # Create forecast
    forecast_results = forecaster.create_prophet_forecast(time_series_data, forecast_periods)
    
    # Create visualizations
    visualizations = forecaster.create_forecast_visualizations(time_series_data, forecast_results)
    
    # Calculate business impact
    business_impact = forecaster.calculate_business_forecast_impact(forecast_results)
    
    return forecaster, time_series_data, forecast_results, visualizations, business_impact

# Example usage
if __name__ == "__main__":
    print("Healthcare Time Series Forecasting Module")
    print("Ready for integration with fraud detection platform")
    print("Supports Prophet and ARIMA models with healthcare-specific seasonality")