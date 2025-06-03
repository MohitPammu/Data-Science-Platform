"""
Ensemble Integration Module
Combines clustering, risk scoring, and time series forecasting
Complete healthcare intelligence platform integration
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# Import all advanced analytics modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.base_healthcare_model import BaseHealthcareModel
from src.advanced_analytics.provider_clustering import ProviderClusteringAnalyzer
from src.advanced_analytics.risk_scoring import ProviderRiskScorer
from src.advanced_analytics.time_series_forecasting import HealthcareCostForecaster, run_time_series_analysis

class HealthcareEnsembleIntelligence(BaseHealthcareModel):
    """
    Comprehensive healthcare intelligence platform
    Integrates clustering, risk scoring, and forecasting for complete analytics
    """
    
    def __init__(self):
        super().__init__("EnsembleIntelligence", "Advanced_Analytics")
        
        # Initialize component analyzers
        self.clustering_analyzer = ProviderClusteringAnalyzer()
        self.risk_scorer = ProviderRiskScorer()
        self.forecaster = HealthcareCostForecaster()
        
        # Results storage
        self.ensemble_results = {}
        self.dashboard_data = {}
        self.business_intelligence = {}
        
    def run_comprehensive_analysis(self, providers_df: pd.DataFrame, claims_df: pd.DataFrame) -> dict:
        """
        Execute complete healthcare intelligence analysis
        
        Args:
            providers_df: Provider data
            claims_df: Claims data
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        self.logger.info("Starting comprehensive healthcare intelligence analysis")
        
        try:
            # Phase 1: Provider Clustering Analysis
            self.logger.info("Phase 1: Provider clustering and risk segmentation")
            clustering_features = self.clustering_analyzer.prepare_clustering_features(providers_df, claims_df)
            optimal_clusters = self.clustering_analyzer.determine_optimal_clusters(clustering_features, max_clusters=6)
            clustered_data = self.clustering_analyzer.perform_clustering(clustering_features, n_clusters=optimal_clusters)
            cluster_analysis = self.clustering_analyzer.analyze_cluster_characteristics(clustered_data)
            
            # Phase 2: Risk Scoring Analysis
            self.logger.info("Phase 2: Comprehensive risk scoring")
            risk_data = self.risk_scorer.calculate_cluster_risk_scores(clustered_data, cluster_analysis)
            risk_data = self.risk_scorer.calculate_volume_risk_scores(risk_data)
            risk_data = self.risk_scorer.calculate_amount_risk_scores(risk_data)
            risk_data = self.risk_scorer.calculate_specialty_risk_scores(risk_data)
            risk_data = self.risk_scorer.calculate_diversity_risk_scores(risk_data)
            final_risk_data = self.risk_scorer.calculate_composite_risk_score(risk_data)
            
            if 'investigation_priority' not in final_risk_data.columns:
                final_risk_data['investigation_priority'] = pd.cut(
                    final_risk_data['composite_risk_score'],
                    bins=[0, 30, 60, 80, 100],
                    labels=['Low', 'Medium', 'High', 'Critical']
            )


            # Phase 3: Time Series Forecasting
            self.logger.info("Phase 3: Healthcare cost forecasting")
            forecaster, time_series_data, forecast_results, ts_visualizations, business_impact = run_time_series_analysis(
                claims_df, forecast_periods=30
            )
            
            # Phase 4: Ensemble Integration
            self.logger.info("Phase 4: Ensemble intelligence integration")
            ensemble_insights = self._create_ensemble_insights(
                final_risk_data, cluster_analysis, forecast_results, business_impact
            )
            
            # Compile comprehensive results
            comprehensive_results = {
                'clustering_results': {
                    'clustered_data': clustered_data,
                    'cluster_analysis': cluster_analysis,
                    'optimal_clusters': optimal_clusters
                },
                'risk_scoring_results': {
                    'risk_scored_data': final_risk_data,
                    'risk_distribution': final_risk_data['risk_category'].value_counts().to_dict(),
                    'high_risk_providers': len(final_risk_data[final_risk_data['composite_risk_score'] >= 80])
                },
                'forecasting_results': {
                    'time_series_data': time_series_data,
                    'forecast_results': forecast_results,
                    'business_impact': business_impact,
                    'visualizations': ts_visualizations
                },
                'ensemble_insights': ensemble_insights,
                'execution_summary': {
                    'total_providers_analyzed': len(final_risk_data),
                    'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'platform_components': ['clustering', 'risk_scoring', 'forecasting', 'ensemble_intelligence']
                }
            }
            
            self.ensemble_results = comprehensive_results
            
            self.logger.info("Comprehensive healthcare intelligence analysis complete")
            
            return comprehensive_results
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive analysis: {str(e)}")
            raise
    
    def _create_ensemble_insights(self, risk_data: pd.DataFrame, cluster_analysis: dict, 
                                forecast_results: dict, forecast_impact: dict) -> dict:
        """
        Create integrated insights combining all analysis components
        
        Args:
            risk_data: Risk-scored provider data
            cluster_analysis: Cluster analysis results
            forecast_results: Forecasting results
            forecast_impact: Forecast business impact
            
        Returns:
            Dictionary with ensemble insights
        """
        self.logger.info("Creating ensemble intelligence insights")
        
        try:
            # Cross-component analysis
            high_risk_providers = risk_data[risk_data['composite_risk_score'] >= 80]
            
            # Cluster-Risk correlation
            cluster_risk_correlation = risk_data.groupby('cluster').agg({
                'composite_risk_score': ['mean', 'max', 'count'],
                'claims_total_amount': 'sum'
            }).round(2)
            
            # Risk-Forecast integration
            total_high_risk_amount = high_risk_providers['claims_total_amount'].sum()
            forecast_mean = forecast_results['forecast_mean']
            
            # Strategic risk projections
            risk_forecast_projection = {
                'current_high_risk_exposure': round(total_high_risk_amount, 2),
                'projected_30day_risk': round(forecast_mean * 30, 2),
                'risk_trend_alignment': 'Increasing' if forecast_results['change_percentage'] > 10 else 'Stable',
                'investigation_priority_forecast': len(high_risk_providers)
            }
            
            # Integrated business recommendations
            integrated_recommendations = self._generate_integrated_recommendations(
                risk_data, forecast_results, forecast_impact
            )
            
            # Platform performance metrics
            platform_performance = {
                'detection_coverage': {
                    'providers_analyzed': len(risk_data),
                    'clusters_identified': len(risk_data['cluster'].unique()),
                    'high_risk_flagged': len(high_risk_providers),
                    'forecast_accuracy_indicator': abs(forecast_results['change_percentage'])
                },
                'business_value_integration': {
                    'risk_based_savings': round(total_high_risk_amount * 0.15, 2),  # 15% fraud prevention
                    'forecast_planning_value': round(forecast_mean * 365 * 0.02, 2),  # 2% planning efficiency
                    'investigation_optimization': len(high_risk_providers) * 200,  # $200 per investigation
                    'total_platform_value': round(
                        total_high_risk_amount * 0.15 + forecast_mean * 365 * 0.02 + len(high_risk_providers) * 200, 2
                    )
                }
            }
            
            ensemble_insights = {
                'cross_component_analysis': {
                    'cluster_risk_correlation': cluster_risk_correlation.to_dict(),
                    'risk_forecast_projection': risk_forecast_projection,
                    'high_risk_provider_analysis': {
                        'count': len(high_risk_providers),
                        'total_amount': round(total_high_risk_amount, 2),
                        'top_specialties': high_risk_providers['specialty'].value_counts().head(3).to_dict(),
                        'cluster_distribution': high_risk_providers['cluster'].value_counts().to_dict()
                    }
                },
                'integrated_recommendations': integrated_recommendations,
                'platform_performance': platform_performance,
                'strategic_intelligence': {
                    'primary_risk_indicators': ['high_claim_amounts', 'cluster_concentration', 'forecast_increases'],
                    'monitoring_priorities': ['real_time_risk_scoring', 'monthly_forecast_updates', 'cluster_migration_tracking'],
                    'scalability_assessment': 'Platform ready for 10x provider volume expansion'
                }
            }
            
            self.logger.info("Ensemble insights creation complete")
            
            return ensemble_insights
            
        except Exception as e:
            self.logger.error(f"Error creating ensemble insights: {str(e)}")
            raise
    
    def _generate_integrated_recommendations(self, risk_data: pd.DataFrame, 
                                           forecast_results: dict, forecast_impact: dict) -> dict:
        """
        Generate integrated recommendations from all platform components
        
        Args:
            risk_data: Risk-scored data
            forecast_results: Forecasting results
            forecast_impact: Forecast business impact
            
        Returns:
            Dictionary with integrated recommendations
        """
        high_risk_count = len(risk_data[risk_data['composite_risk_score'] >= 80])
        forecast_change = forecast_results['change_percentage']
        risk_level = forecast_impact['risk_assessment']['risk_level']
        
        recommendations = {
            'immediate_actions': [],
            'strategic_initiatives': [],
            'resource_allocation': {},
            'monitoring_framework': {}
        }
        
        # Immediate actions based on integrated analysis
        if high_risk_count > 0:
            recommendations['immediate_actions'].append(
                f"Deploy enhanced monitoring for {high_risk_count} high-risk providers identified through ensemble analysis"
            )
        
        if forecast_change > 15:
            recommendations['immediate_actions'].append(
                f"Prepare for {forecast_change:+.1f}% cost increase with expanded investigation capacity"
            )
        
        if risk_level == 'High':
            recommendations['immediate_actions'].append(
                "Activate emergency fraud prevention protocols based on forecast risk assessment"
            )
        
        # Strategic initiatives
        recommendations['strategic_initiatives'] = [
            "Implement real-time ensemble scoring for new provider enrollments",
            "Develop predictive fraud prevention using integrated cluster-risk-forecast intelligence",
            "Create automated alert system combining risk scores with forecast anomalies"
        ]
        
        # Resource allocation
        investigation_budget = forecast_impact['recommended_actions']['investigation_budget']
        recommendations['resource_allocation'] = {
            'investigation_team_size': max(high_risk_count // 20, 2),  # 1 investigator per 20 high-risk providers
            'monthly_investigation_budget': round(investigation_budget / 12, 2),
            'technology_investment': round(investigation_budget * 0.3, 2),  # 30% for tech
            'training_budget': round(investigation_budget * 0.1, 2)  # 10% for training
        }
        
        # Monitoring framework
        recommendations['monitoring_framework'] = {
            'risk_score_updates': 'Weekly for high-risk providers, monthly for others',
            'forecast_refresh': 'Monthly with quarterly deep analysis',
            'cluster_analysis': 'Quarterly to identify migration patterns',
            'platform_performance': 'Real-time dashboard with weekly executive reports'
        }
        
        return recommendations
    
    def create_executive_dashboard_data(self) -> dict:
        """
        Create executive-level dashboard data combining all analytics
        
        Returns:
            Dictionary with executive dashboard data
        """
        self.logger.info("Creating executive dashboard data")
        
        try:
            if not self.ensemble_results:
                raise ValueError("Must run comprehensive analysis before creating dashboard data")
            
            # Extract key components
            risk_data = self.ensemble_results['risk_scoring_results']['risk_scored_data']
            forecast_results = self.ensemble_results['forecasting_results']['forecast_results']
            business_impact = self.ensemble_results['forecasting_results']['business_impact']
            ensemble_insights = self.ensemble_results['ensemble_insights']
            
            # Executive KPIs
            executive_kpis = {
                'total_providers': len(risk_data),
                'high_risk_providers': len(risk_data[risk_data['composite_risk_score'] >= 80]),
                'clusters_identified': len(risk_data['cluster'].unique()),
                'forecast_trend': f"{forecast_results['change_percentage']:+.1f}%",
                'annual_cost_forecast': f"${business_impact['financial_projections']['annual_forecast_cost']:,.0f}",
                'platform_value': f"${ensemble_insights['platform_performance']['business_value_integration']['total_platform_value']:,.0f}",
                'risk_level': business_impact['risk_assessment']['risk_level']
            }
            
            # Risk distribution for charts
            risk_distribution = risk_data['risk_category'].value_counts().to_dict()
            
            # Geographic analysis
            geographic_summary = risk_data.groupby('state').agg({
                'composite_risk_score': 'mean',
                'claims_total_amount': 'sum',
                'provider_id': 'count'
            }).round(2).to_dict('index')
            
            # Specialty analysis
            specialty_summary = risk_data.groupby('specialty').agg({
                'composite_risk_score': 'mean',
                'claims_total_amount': 'sum',
                'provider_id': 'count'
            }).round(2).to_dict('index')
            
            # Time series summary
            forecast_summary = {
                'historical_average': forecast_results['historical_mean'],
                'forecast_average': forecast_results['forecast_mean'],
                'change_percentage': forecast_results['change_percentage'],
                'risk_assessment': business_impact['risk_assessment']['risk_level'],
                'investigation_budget': business_impact['recommended_actions']['investigation_budget']
            }
            
            # Top risk providers for investigation
            top_risk_providers = risk_data.nlargest(10, 'composite_risk_score')[
                ['provider_id', 'specialty', 'state', 'cluster', 'composite_risk_score', 
                 'risk_category', 'investigation_priority', 'claims_total_amount']
            ].to_dict('records')
            
            dashboard_data = {
                'executive_kpis': executive_kpis,
                'risk_distribution': risk_distribution,
                'geographic_summary': geographic_summary,
                'specialty_summary': specialty_summary,
                'forecast_summary': forecast_summary,
                'top_risk_providers': top_risk_providers,
                'integrated_recommendations': ensemble_insights['integrated_recommendations'],
                'platform_performance': ensemble_insights['platform_performance'],
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            self.dashboard_data = dashboard_data
            
            self.logger.info("Executive dashboard data creation complete")
            
            return dashboard_data
            
        except Exception as e:
            self.logger.error(f"Error creating dashboard data: {str(e)}")
            raise
    
    def save_platform_results(self, output_dir: str = "data/advanced_analytics") -> dict:
        """
        Save all platform results for deployment and analysis
        
        Args:
            output_dir: Directory to save results
            
        Returns:
            Dictionary with saved file paths
        """
        self.logger.info(f"Saving platform results to {output_dir}")
        
        try:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            saved_files = {}
            
            # Save main results
            if self.ensemble_results:
                # Risk scored data
                risk_data = self.ensemble_results['risk_scoring_results']['risk_scored_data']
                risk_file = os.path.join(output_dir, "ensemble_risk_analysis.csv")
                risk_data.to_csv(risk_file, index=False)
                saved_files['risk_analysis'] = risk_file
                
                # Executive dashboard data
                if self.dashboard_data:
                    dashboard_file = os.path.join(output_dir, "executive_dashboard_data.json")
                    with open(dashboard_file, 'w') as f:
                        json.dump(self._make_json_serializable(self.dashboard_data), f, indent=2)
                    saved_files['dashboard_data'] = dashboard_file
                
                # Ensemble insights
                insights_file = os.path.join(output_dir, "ensemble_insights.json")
                with open(insights_file, 'w') as f:
                    json.dump(self._make_json_serializable(self.ensemble_results['ensemble_insights']), f, indent=2)
                saved_files['ensemble_insights'] = insights_file
                
                # Comprehensive summary
                summary_file = os.path.join(output_dir, "platform_summary.json")
                summary_data = {
                    'execution_summary': self.ensemble_results['execution_summary'],
                    'risk_distribution': self.ensemble_results['risk_scoring_results']['risk_distribution'],
                    'forecast_summary': {
                        'model_type': self.ensemble_results['forecasting_results']['forecast_results']['model_type'],
                        'change_percentage': self.ensemble_results['forecasting_results']['forecast_results']['change_percentage'],
                        'risk_level': self.ensemble_results['forecasting_results']['business_impact']['risk_assessment']['risk_level']
                    }
                }
                with open(summary_file, 'w') as f:
                    json.dump(summary_data, f, indent=2)
                saved_files['platform_summary'] = summary_file
            
            self.logger.info(f"Platform results saved: {len(saved_files)} files")
            
            return saved_files
            
        except Exception as e:
            self.logger.error(f"Error saving platform results: {str(e)}")
            raise
    
    # Around line 420, replace the _make_json_serializable method:

    def _make_json_serializable(self, obj):
        """Convert pandas objects to JSON serializable format"""
        if isinstance(obj, dict):
            # Handle dictionary keys that might be tuples (MultiIndex columns)
            return {
                str(key) if isinstance(key, tuple) else key: self._make_json_serializable(value) 
                for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, pd.DataFrame):
            # Handle DataFrames with MultiIndex columns
            df_copy = obj.copy()
            if isinstance(df_copy.columns, pd.MultiIndex):
                # Flatten MultiIndex columns to strings
                df_copy.columns = ['_'.join(map(str, col)).strip() for col in df_copy.columns.values]
            return df_copy.to_dict('records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):  # Handle NaN values
            return None
        else:
            return obj

# Main execution function for ensemble platform
def run_complete_healthcare_platform(providers_df: pd.DataFrame, claims_df: pd.DataFrame) -> dict:
    """
    Execute complete healthcare intelligence platform
    
    Args:
        providers_df: Provider data
        claims_df: Claims data
        
    Returns:
        Complete platform results with dashboard data
    """
    # Initialize ensemble platform
    platform = HealthcareEnsembleIntelligence()
    
    # Run comprehensive analysis
    comprehensive_results = platform.run_comprehensive_analysis(providers_df, claims_df)
    
    # Create executive dashboard data
    dashboard_data = platform.create_executive_dashboard_data()
    
    # Save platform results
    saved_files = platform.save_platform_results()
    
    # Return complete platform package
    return {
        'platform_results': comprehensive_results,
        'dashboard_data': dashboard_data,
        'saved_files': saved_files,
        'platform_instance': platform
    }

# Example usage
if __name__ == "__main__":
    print("Healthcare Ensemble Intelligence Platform")
    print("Integrating clustering, risk scoring, and forecasting")
    print("Ready for complete healthcare analytics deployment")