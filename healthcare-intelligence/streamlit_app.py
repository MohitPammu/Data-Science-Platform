"""
Healthcare Intelligence Platform - Interactive Dashboard
Streamlit application for comprehensive healthcare fraud detection analytics
Data Product Owner approach with stakeholder-friendly interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Healthcare Intelligence Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f77b4, #17becf);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .risk-high { border-left-color: #d62728 !important; }
    .risk-medium { border-left-color: #ff7f0e !important; }
    .risk-low { border-left-color: #2ca02c !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_platform_data():
    """Load all platform data files"""
    try:
        # Load executive dashboard data
        with open('data/advanced_analytics/executive_dashboard_data.json', 'r') as f:
            dashboard_data = json.load(f)
        
        # Load ensemble insights
        with open('data/advanced_analytics/ensemble_insights.json', 'r') as f:
            ensemble_insights = json.load(f)
        
        # Load platform summary
        with open('data/advanced_analytics/platform_summary.json', 'r') as f:
            platform_summary = json.load(f)
        
        # Load risk analysis CSV
        risk_analysis = pd.read_csv('data/advanced_analytics/ensemble_risk_analysis.csv')
        
        return dashboard_data, ensemble_insights, platform_summary, risk_analysis
    
    except FileNotFoundError as e:
        st.error(f"Data files not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

def create_roi_calculator():
    """Interactive ROI calculator for different organization sizes"""
    st.subheader("💰 ROI Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Organization Parameters:**")
        org_size = st.selectbox("Organization Size", 
                               ["Small Payer (10K claims)", "Medium Payer (50K claims)", "Large Payer (100K claims)", "Custom"])
        
        if org_size == "Custom":
            annual_claims = st.number_input("Annual Claims Volume", min_value=1000, max_value=1000000, value=50000)
            avg_claim_amount = st.number_input("Average Claim Amount ($)", min_value=100, max_value=5000, value=1500)
        else:
            # Predefined scenarios
            scenarios = {
                "Small Payer (10K claims)": (10000, 1500),
                "Medium Payer (50K claims)": (50000, 1500),
                "Large Payer (100K claims)": (100000, 1500)
            }
            annual_claims, avg_claim_amount = scenarios[org_size]
        
        fraud_rate = st.slider("Estimated Fraud Rate (%)", min_value=1.0, max_value=10.0, value=3.0) / 100
    
    with col2:
        st.write("**Platform Parameters:**")
        detection_rate = st.slider("Detection Rate (%)", min_value=70, max_value=95, value=80) / 100
        recovery_rate = st.slider("Recovery Rate (%)", min_value=60, max_value=90, value=75) / 100
        investigation_cost = st.number_input("Cost per Investigation ($)", min_value=100, max_value=500, value=200)
    
    # Calculate ROI
    total_claims_value = annual_claims * avg_claim_amount
    estimated_fraud = total_claims_value * fraud_rate
    detected_fraud = estimated_fraud * detection_rate
    recovered_amount = detected_fraud * recovery_rate
    
    # Investigation costs (assume 15% of claims need investigation)
    investigations_needed = annual_claims * 0.15
    total_investigation_cost = investigations_needed * investigation_cost
    
    # Platform cost (estimated)
    platform_cost = 50000  # Annual platform cost
    
    # Net benefit
    net_annual_benefit = recovered_amount - total_investigation_cost - platform_cost
    roi_percentage = (net_annual_benefit / platform_cost) * 100 if platform_cost > 0 else 0
    
    # Display results
    st.write("**ROI Analysis Results:**")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Claims Value", f"${total_claims_value:,.0f}")
    with col2:
        st.metric("Estimated Fraud", f"${estimated_fraud:,.0f}")
    with col3:
        st.metric("Amount Recovered", f"${recovered_amount:,.0f}")
    with col4:
        st.metric("Net Annual Benefit", f"${net_annual_benefit:,.0f}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Investigation Costs", f"${total_investigation_cost:,.0f}")
    with col2:
        st.metric("Platform Cost", f"${platform_cost:,.0f}")
    with col3:
        st.metric("ROI", f"{roi_percentage:,.1f}%", delta=f"{roi_percentage-200:.1f}%" if roi_percentage > 200 else None)

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Healthcare Intelligence Platform</h1>
        <p>Comprehensive Fraud Detection & Risk Management System</p>
        <p><em>Data Product Owner Approach • Production-Ready Analytics • Business Intelligence</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    dashboard_data, ensemble_insights, platform_summary, risk_analysis = load_platform_data()
    
    # Sidebar for navigation
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.selectbox("Select Analysis", [
        "Executive Dashboard",
        "Provider Risk Analysis", 
        "Fraud Detection Intelligence",
        "Cost Forecasting",
        "ROI Calculator",
        "Platform Performance"
    ])
    
    # Executive Dashboard Page
    if page == "Executive Dashboard":
        st.title("📈 Executive Dashboard")
        
        # Key Performance Indicators
        st.subheader("🎯 Key Performance Indicators")
        
        exec_kpis = dashboard_data['executive_kpis']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Providers", exec_kpis['total_providers'])
        with col2:
            st.metric("High-Risk Providers", exec_kpis['high_risk_providers'])
        with col3:
            st.metric("Clusters Identified", exec_kpis['clusters_identified'])
        with col4:
            st.metric("Risk Level", exec_kpis['risk_level'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Forecast Trend", exec_kpis['forecast_trend'])
        with col2:
            st.metric("Annual Cost Forecast", exec_kpis['annual_cost_forecast'])
        with col3:
            st.metric("Platform Value", exec_kpis['platform_value'])
        
        # Risk Distribution Chart
        st.subheader("⚠️ Risk Distribution Analysis")
        risk_dist = dashboard_data['risk_distribution']
        
        fig_risk = px.pie(
            values=list(risk_dist.values()),
            names=list(risk_dist.keys()),
            title="Provider Risk Categories",
            color_discrete_map={
                'Low': '#2ca02c',
                'Medium': '#ff7f0e', 
                'High': '#d62728',
                'Critical': '#8b0000'
            }
        )
        st.plotly_chart(fig_risk, use_container_width=True)
        
        # Geographic Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🗺️ Geographic Risk Analysis")
            geo_data = dashboard_data['geographic_summary']
            
            if geo_data:
                geo_df = pd.DataFrame.from_dict(geo_data, orient='index')
                geo_df = geo_df.reset_index().rename(columns={'index': 'state'})
                
                fig_geo = px.bar(
                    geo_df, 
                    x='state', 
                    y='composite_risk_score',
                    title="Average Risk Score by State",
                    color='composite_risk_score',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig_geo, use_container_width=True)
        
        with col2:
            st.subheader("🏥 Specialty Risk Analysis")
            specialty_data = dashboard_data['specialty_summary']
            
            if specialty_data:
                specialty_df = pd.DataFrame.from_dict(specialty_data, orient='index')
                specialty_df = specialty_df.reset_index().rename(columns={'index': 'specialty'})
                
                fig_specialty = px.bar(
                    specialty_df, 
                    x='specialty', 
                    y='composite_risk_score',
                    title="Average Risk Score by Specialty",
                    color='composite_risk_score',
                    color_continuous_scale='Oranges'
                )
                fig_specialty.update_xaxes(tickangle=45)
                st.plotly_chart(fig_specialty, use_container_width=True)
    
    # Provider Risk Analysis Page
    elif page == "Provider Risk Analysis":
        st.title("⚠️ Provider Risk Analysis")
        
        # Risk score distribution
        st.subheader("📊 Risk Score Distribution")
        
        fig_hist = px.histogram(
            risk_analysis, 
            x='composite_risk_score',
            nbins=20,
            title="Distribution of Provider Risk Scores",
            labels={'composite_risk_score': 'Risk Score', 'count': 'Number of Providers'}
        )
        
        # Add risk threshold lines
        fig_hist.add_vline(x=60, line_dash="dash", line_color="orange", annotation_text="Medium Risk (60)")
        fig_hist.add_vline(x=80, line_dash="dash", line_color="red", annotation_text="High Risk (80)")
        
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Top risk providers table
        st.subheader("🚨 High-Risk Providers Requiring Investigation")
        
        high_risk_providers = risk_analysis[risk_analysis['composite_risk_score'] >= 60].nlargest(10, 'composite_risk_score')
        
        if len(high_risk_providers) > 0:
            display_cols = ['provider_id', 'specialty', 'state', 'cluster', 'composite_risk_score', 'risk_category', 'claims_total_amount']
            st.dataframe(
                high_risk_providers[display_cols].style.format({
                    'composite_risk_score': '{:.1f}',
                    'claims_total_amount': '${:,.2f}'
                }),
                use_container_width=True
            )
        else:
            st.info("No high-risk providers identified in current analysis.")
        
        # Risk components analysis
        st.subheader("🔍 Risk Components Breakdown")
        
        risk_components = ['cluster_risk_score', 'volume_risk_score', 'amount_risk_score', 'specialty_risk_score', 'diversity_risk_score']
        
        fig_components = go.Figure()
        
        for component in risk_components:
            if component in risk_analysis.columns:
                fig_components.add_trace(go.Box(
                    y=risk_analysis[component],
                    name=component.replace('_', ' ').title(),
                    boxpoints='outliers'
                ))
        
        fig_components.update_layout(
            title="Risk Score Components Distribution",
            yaxis_title="Risk Score",
            xaxis_title="Risk Component"
        )
        
        st.plotly_chart(fig_components, use_container_width=True)
    
    # Fraud Detection Intelligence Page
    elif page == "Fraud Detection Intelligence":
        st.title("🔍 Fraud Detection Intelligence")
        
        # Cluster analysis
        st.subheader("🎯 Provider Cluster Analysis")
        
        cluster_summary = risk_analysis.groupby('cluster').agg({
            'provider_id': 'count',
            'composite_risk_score': ['mean', 'max'],
            'claims_total_amount': 'sum'
        }).round(2)
        
        cluster_summary.columns = ['Provider Count', 'Avg Risk Score', 'Max Risk Score', 'Total Claims Amount']
        cluster_summary = cluster_summary.reset_index()
        
        st.dataframe(cluster_summary.style.format({
            'Avg Risk Score': '{:.1f}',
            'Max Risk Score': '{:.1f}',
            'Total Claims Amount': '${:,.2f}'
        }), use_container_width=True)
        
        # Cluster visualization
        fig_cluster = px.scatter(
            risk_analysis,
            x='claims_count',
            y='claims_total_amount',
            color='cluster',
            size='composite_risk_score',
            hover_data=['specialty', 'state'],
            title="Provider Clusters: Claims Volume vs Total Amount",
            labels={
                'claims_count': 'Number of Claims',
                'claims_total_amount': 'Total Claims Amount ($)'
            }
        )
        
        st.plotly_chart(fig_cluster, use_container_width=True)
        
        # Investigation priorities
        st.subheader("📋 Investigation Priorities")
        
        if 'investigation_priority' in risk_analysis.columns:
            priority_counts = risk_analysis['investigation_priority'].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_priority = px.pie(
                    values=priority_counts.values,
                    names=priority_counts.index,
                    title="Investigation Priority Distribution"
                )
                st.plotly_chart(fig_priority, use_container_width=True)
            
            with col2:
                st.write("**Investigation Workload:**")
                for priority, count in priority_counts.items():
                    st.metric(f"{priority} Priority", f"{count} providers")
    
    # Cost Forecasting Page
    elif page == "Cost Forecasting":
        st.title("📈 Healthcare Cost Forecasting")
        
        forecast_summary = dashboard_data.get('forecast_summary', {})
        
        if forecast_summary:
            # Forecast metrics
            st.subheader("🔮 Forecast Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Historical Average", f"${forecast_summary.get('historical_average', 0):,.2f}")
            with col2:
                st.metric("Forecast Average", f"${forecast_summary.get('forecast_average', 0):,.2f}")
            with col3:
                st.metric("Change", f"{forecast_summary.get('change_percentage', 0):+.1f}%")
            with col4:
                st.metric("Risk Assessment", forecast_summary.get('risk_assessment', 'Unknown'))
            
            # Business impact
            st.subheader("💼 Business Impact Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Investigation Budget", f"${forecast_summary.get('investigation_budget', 0):,.2f}")
            with col2:
                if forecast_summary.get('change_percentage', 0) > 15:
                    st.warning("⚠️ Significant cost increase forecasted - enhanced monitoring recommended")
                elif forecast_summary.get('change_percentage', 0) > 5:
                    st.info("ℹ️ Moderate cost increase expected - standard monitoring sufficient")
                else:
                    st.success("✅ Stable cost forecast - maintain current monitoring")
        
        else:
            st.info("Forecast data not available in current analysis.")
    
    # ROI Calculator Page
    elif page == "ROI Calculator":
        st.title("💰 ROI Calculator")
        create_roi_calculator()
    
    # Platform Performance Page
    elif page == "Platform Performance":
        st.title("🎯 Platform Performance")
        
        # Platform summary
        st.subheader("📊 Platform Execution Summary")
        
        exec_summary = platform_summary.get('execution_summary', {})
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Providers Analyzed", exec_summary.get('total_providers_analyzed', 'N/A'))
        with col2:
            st.metric("Analysis Date", exec_summary.get('analysis_date', 'N/A'))
        with col3:
            components = exec_summary.get('platform_components', [])
            st.metric("Platform Components", len(components))
        
        # Risk distribution summary
        if 'risk_distribution' in platform_summary:
            st.subheader("⚠️ Risk Analysis Results")
            risk_dist = platform_summary['risk_distribution']
            
            for category, count in risk_dist.items():
                st.metric(f"{category} Risk Providers", count)
        
        # Forecast summary
        if 'forecast_summary' in platform_summary:
            st.subheader("📈 Forecasting Results")
            forecast = platform_summary['forecast_summary']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Model Type", forecast.get('model_type', 'N/A'))
            with col2:
                st.metric("Forecast Change", f"{forecast.get('change_percentage', 0):+.1f}%")
            with col3:
                st.metric("Risk Level", forecast.get('risk_level', 'N/A'))
        
        # Platform recommendations
        if ensemble_insights and 'integrated_recommendations' in ensemble_insights:
            st.subheader("💡 Platform Recommendations")
            recommendations = ensemble_insights['integrated_recommendations']
            
            if 'immediate_actions' in recommendations:
                st.write("**Immediate Actions:**")
                for action in recommendations['immediate_actions']:
                    st.write(f"• {action}")
            
            if 'strategic_initiatives' in recommendations:
                st.write("**Strategic Initiatives:**")
                for initiative in recommendations['strategic_initiatives']:
                    st.write(f"• {initiative}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p><strong>Healthcare Intelligence Platform</strong> | Data Product Owner Approach</p>
        <p>Last Updated: {}</p>
    </div>
    """.format(dashboard_data.get('last_updated', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))), unsafe_allow_html=True)

if __name__ == "__main__":
    main()