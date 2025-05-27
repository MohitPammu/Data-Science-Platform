"""
Simplified Geographic Analysis - Working Version
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class SimpleGeographicAnalyzer:
    """Simple, working geographic fraud analyzer"""
    
    def analyze_geographic_patterns(self, providers_df: pd.DataFrame, 
                                  claims_df: pd.DataFrame) -> dict:
        """Analyze fraud patterns by state"""
        logger.info("Starting geographic analysis...")
        
        # Merge data
        provider_summary = claims_df.groupby('provider_id').agg({
            'claim_amount': ['sum', 'mean', 'count'],
            'paid_amount': 'sum'
        }).round(2)
        
        provider_summary.columns = [f'{col[0]}_{col[1]}' for col in provider_summary.columns]
        provider_summary = provider_summary.reset_index()
        
        merged = providers_df.merge(provider_summary, on='provider_id', how='left')
        merged = merged.fillna(0)
        
        # State analysis
        state_stats = merged.groupby('state').agg({
            'provider_id': 'count',
            'claim_amount_sum': 'sum',
            'claim_amount_mean': 'mean',
            'paid_amount_sum': 'sum'
        }).round(2).reset_index()
        
        # Find highest risk states
        high_risk_states = []
        for _, row in state_stats.nlargest(5, 'claim_amount_sum').iterrows():
            high_risk_states.append({
                'state': row['state'],
                'total_claims': row['claim_amount_sum'],
                'avg_claim': row['claim_amount_mean'],
                'provider_count': row['provider_id']
            })
        
        return {
            'state_analysis': state_stats.to_dict('records'),
            'high_risk_states': high_risk_states,
            'total_states': len(state_stats)
        }
    
    def analyze_temporal_patterns(self, claims_df: pd.DataFrame) -> dict:
        """Analyze temporal patterns"""
        logger.info("Starting temporal analysis...")
        
        claims_df['claim_date'] = pd.to_datetime(claims_df['claim_date'])
        claims_df['month'] = claims_df['claim_date'].dt.month
        claims_df['quarter'] = claims_df['claim_date'].dt.quarter
        
        # Monthly patterns
        monthly_totals = claims_df.groupby('month')['claim_amount'].sum()
        
        # Quarterly patterns  
        quarterly_totals = claims_df.groupby('quarter')['claim_amount'].sum()
        
        return {
            'peak_month': int(monthly_totals.idxmax()),
            'peak_month_amount': float(monthly_totals.max()),
            'peak_quarter': int(quarterly_totals.idxmax()),
            'peak_quarter_amount': float(quarterly_totals.max()),
            'total_period_days': (claims_df['claim_date'].max() - claims_df['claim_date'].min()).days
        }
    
    def generate_report(self, providers_df: pd.DataFrame, claims_df: pd.DataFrame) -> dict:
        """Generate complete geographic and temporal report"""
        geographic = self.analyze_geographic_patterns(providers_df, claims_df)
        temporal = self.analyze_temporal_patterns(claims_df)
        
        return {
            'geographic_analysis': geographic,
            'temporal_analysis': temporal,
            'summary': {
                'states_analyzed': geographic['total_states'],
                'analysis_days': temporal['total_period_days'],
                'peak_fraud_month': temporal['peak_month']
            }
        }
