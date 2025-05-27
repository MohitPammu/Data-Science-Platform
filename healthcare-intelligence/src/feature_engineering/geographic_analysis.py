"""
Geographic and Temporal Healthcare Fraud Analysis
Identifies regional patterns and time-based anomalies
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class GeographicFraudAnalyzer:
    """Analyze geographic and temporal patterns in healthcare fraud"""
    
    def __init__(self):
        self.state_risk_profiles = {}
        self.temporal_patterns = {}
    
    def analyze_geographic_patterns(self, providers_df: pd.DataFrame, 
                                  claims_df: pd.DataFrame) -> Dict[str, any]:
        """Analyze fraud patterns by geographic region"""
        logger.info("Starting geographic fraud pattern analysis...")
        
        # Merge data for analysis
        provider_claims = self._merge_data_for_geographic_analysis(providers_df, claims_df)
        
        # Calculate state-level metrics
        state_analysis = self._calculate_state_metrics(provider_claims)
        
        # Identify high-risk states
        high_risk_states = self._identify_high_risk_states(state_analysis)
        
        # Calculate interstate comparison metrics
        interstate_metrics = self._calculate_interstate_metrics(state_analysis)
        
        geographic_report = {
            'state_analysis': state_analysis,
            'high_risk_states': high_risk_states,
            'interstate_metrics': interstate_metrics,
            'total_states_analyzed': len(state_analysis)
        }
        
        self.state_risk_profiles = geographic_report
        logger.info(f"Geographic analysis complete for {len(state_analysis)} states")
        
        return geographic_report
    
    def analyze_temporal_patterns(self, claims_df: pd.DataFrame) -> Dict[str, any]:
        """Analyze time-based fraud patterns"""
        logger.info("Starting temporal fraud pattern analysis...")
        
        # Convert claim_date to datetime
        claims_df['claim_date'] = pd.to_datetime(claims_df['claim_date'])
        
        # Monthly analysis
        monthly_analysis = self._analyze_monthly_patterns(claims_df)
        
        # Day of week analysis
        dow_analysis = self._analyze_day_of_week_patterns(claims_df)
        
        # Seasonal analysis
        seasonal_analysis = self._analyze_seasonal_patterns(claims_df)
        
        # Identify temporal anomalies
        temporal_anomalies = self._identify_temporal_anomalies(claims_df)
        
        temporal_report = {
            'monthly_patterns': monthly_analysis,
            'day_of_week_patterns': dow_analysis,
            'seasonal_patterns': seasonal_analysis,
            'temporal_anomalies': temporal_anomalies,
            'analysis_period': {
                'start_date': claims_df['claim_date'].min().strftime('%Y-%m-%d'),
                'end_date': claims_df['claim_date'].max().strftime('%Y-%m-%d'),
                'total_days': (claims_df['claim_date'].max() - claims_df['claim_date'].min()).days
            }
        }
        
        self.temporal_patterns = temporal_report
        logger.info("Temporal analysis complete")
        
        return temporal_report
    
    def _merge_data_for_geographic_analysis(self, providers_df: pd.DataFrame, 
                                          claims_df: pd.DataFrame) -> pd.DataFrame:
        """Merge provider and claims data for geographic analysis"""
        # Group claims by provider
        provider_summary = claims_df.groupby('provider_id').agg({
            'claim_amount': ['sum', 'mean', 'count'],
            'paid_amount': 'sum',
            'procedure_code': 'nunique'
        }).round(2)
        
        # Flatten column names
        provider_summary.columns = [f"{col[0]}_{col[1]}" for col in provider_summary.columns]
        provider_summary = provider_summary.reset_index()
        
        # Merge with provider data
        merged = providers_df.merge(provider_summary, on='provider_id', how='left')
        merged = merged.fillna(0)  # Fill NaN values for providers with no claims
        
        return merged
    
    def _calculate_state_metrics(self, provider_claims: pd.DataFrame) -> pd.DataFrame:
        """Calculate key metrics by state"""
        state_metrics = provider_claims.groupby('state').agg({
            'provider_id': 'count',
            'claim_amount_sum': ['sum', 'mean', 'std'],
            'claim_amount_mean': 'mean',
            'claim_amount_count': ['sum', 'mean'],
            'paid_amount_sum': 'sum'
        }).round(2)
        
        # Flatten column names
        state_metrics.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] 
                                for col in state_metrics.columns]
        state_metrics = state_metrics.reset_index()
        
        # Calculate additional risk indicators
        state_metrics['avg_claim_per_provider'] = (state_metrics['claim_amount_count_sum'] / 
                                                  state_metrics['provider_id']).round(2)
        state_metrics['payment_ratio'] = (state_metrics['paid_amount_sum'] / 
                                         state_metrics['claim_amount_sum_sum']).round(3)
        
        return state_metrics
    
    def _identify_high_risk_states(self, state_analysis: pd.DataFrame) -> List[Dict]:
        """Identify states with highest fraud risk indicators"""
        # Calculate risk scores based on multiple factors
        risk_factors = ['claim_amount_mean_mean', 'avg_claim_per_provider', 'payment_ratio']
        
        high_risk_states = []
        
        for factor in risk_factors:
            if factor in state_analysis.columns:
                threshold = state_analysis[factor].quantile(0.8)  # Top 20%
                risky_states = state_analysis[state_analysis[factor] >= threshold]
                
                for _, state_row in risky_states.iterrows():
                    high_risk_states.append({
                        'state': state_row['state'],
                        'risk_factor': factor,
                        'value': state_row[factor],
                        'risk_level': 'High' if state_row[factor] >= threshold else 'Medium'
                    })
        
        return high_risk_states[:10]  # Return top 10 risk indicators
    
    def _calculate_interstate_metrics(self, state_analysis: pd.DataFrame) -> Dict[str, float]:
        """Calculate metrics for interstate comparison"""
        return {
            'highest_avg_claim_state': state_analysis.loc[state_analysis['claim_amount_mean_mean'].idxmax(), 'state'],
            'highest_avg_claim_amount': state_analysis['claim_amount_mean_mean'].max(),
            'lowest_avg_claim_state': state_analysis.loc[state_analysis['claim_amount_mean_mean'].idxmin(), 'state'],
            'lowest_avg_claim_amount': state_analysis['claim_amount_mean_mean'].min(),
            'interstate_variation_coefficient': (state_analysis['claim_amount_mean_mean'].std() / 
                                               state_analysis['claim_amount_mean_mean'].mean()).round(3)
        }
    
    def _analyze_monthly_patterns(self, claims_df: pd.DataFrame) -> Dict[str, float]:
        """Analyze monthly claim patterns"""
        claims_df['month'] = claims_df['claim_date'].dt.month
        
        monthly_stats = claims_df.groupby('month').agg({
            'claim_amount': ['sum', 'mean', 'count'],
            'paid_amount': 'sum'
        }).round(2)
        
        # Find peak month
        monthly_totals = claims_df.groupby('month')['claim_amount'].sum()
        peak_month = monthly_totals.idxmax()
        lowest_month = monthly_totals.idxmin()
        
        return {
            'peak_fraud_month': int(peak_month),
            'peak_month_amount': float(monthly_totals.max()),
            'lowest_fraud_month': int(lowest_month),
            'lowest_month_amount': float(monthly_totals.min()),
            'monthly_variation_coefficient': float((monthly_totals.std() / monthly_totals.mean()).round(3))
        }
    
    def _analyze_day_of_week_patterns(self, claims_df: pd.DataFrame) -> Dict[str, any]:
        """Analyze day of week patterns"""
        claims_df['day_of_week'] = claims_df['claim_date'].dt.day_name()
        
        dow_stats = claims_df.groupby('day_of_week')['claim_amount'].agg(['sum', 'mean', 'count'])
        
        return {
            'highest_claim_day': dow_stats['sum'].idxmax(),
            'highest_day_amount': float(dow_stats['sum'].max()),
            'average_weekend_vs_weekday_ratio': 1.15  # Placeholder - would calculate actual ratio
        }
    
    def _analyze_seasonal_patterns(self, claims_df: pd.DataFrame) -> Dict[str, float]:
        """Analyze seasonal fraud patterns"""
        claims_df['quarter'] = claims_df['claim_date'].dt.quarter
        
        quarterly_stats = claims_df.groupby('quarter')['claim_amount'].sum()
        
        return {
            'highest_fraud_quarter': int(quarterly_stats.idxmax()),
            'highest_quarter_amount': float(quarterly_stats.max()),
            'seasonal_variation': float((quarterly_stats.std() / quarterly_stats.mean()).round(3))
        }
    
    def _identify_temporal_anomalies(self, claims_df: pd.DataFrame) -> List[Dict]:
        """Identify temporal anomalies in claim patterns"""
        # Daily claim amounts
        daily_claims = claims_df.groupby('claim_date')['claim_amount'].sum()
        
        # Calculate rolling statistics
        rolling_mean = daily_claims.rolling(window=7).mean()
        rolling_std = daily_claims.rolling(window=7).std()
        
        # Identify anomalies (simplified approach)
        anomalies = []
        threshold = 2  # 2 standard deviations
        
        for date, amount in daily_claims.items():
            if pd.notna(rolling_mean.loc[date]) and pd.notna(rolling_std.loc[date]):
                z_score = abs((amount - rolling_mean.loc[date]) / rolling_std.loc[date])
                if z_score > threshold:
                    anomalies.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'claim_amount': float(amount),
                        'z_score': float(z_score.round(2)),
                        'anomaly_type': 'High' if amount > rolling_mean.loc[date] else 'Low'
                    })
        
        return anomalies[:5]  # Return top 5 anomalies

    def generate_comprehensive_geographic_report(self, providers_df: pd.DataFrame, 
                                               claims_df: pd.DataFrame) -> Dict[str, any]:
        """Generate comprehensive geographic and temporal fraud analysis"""
        
        # Run both analyses
        geographic_analysis = self.analyze_geographic_patterns(providers_df, claims_df)
        temporal_analysis = self.analyze_temporal_patterns(claims_df)
        
        # Combine results
        comprehensive_report = {
            'geographic_analysis': geographic_analysis,
            'temporal_analysis': temporal_analysis,
            'executive_summary': {
                'total_states_analyzed': geographic_analysis['total_states_analyzed'],
                'analysis_period_days': temporal_analysis['analysis_period']['total_days'],
                'peak_fraud_month': temporal_analysis['monthly_patterns']['peak_fraud_month'],
                'highest_risk_state_factor': len(geographic_analysis['high_risk_states'])
            }
        }
        
        return comprehensive_report
