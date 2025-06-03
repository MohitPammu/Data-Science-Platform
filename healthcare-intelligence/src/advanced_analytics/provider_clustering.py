"""
Healthcare Provider Clustering Analytics
K-means clustering for healthcare provider risk segmentation
Advanced analytics module for fraud detection platform
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import base model class for consistency
import sys
sys.path.append('.')
from src.models.base_healthcare_model import BaseHealthcareModel

class ProviderClusteringAnalyzer(BaseHealthcareModel):
    """
    K-means clustering for healthcare provider risk segmentation
    Identifies provider groups based on billing patterns and fraud indicators
    """
    
    def __init__(self):
        super().__init__("ProviderClustering", "Advanced_Analytics")
        self.kmeans_model = None
        self.scaler = StandardScaler()
        self.cluster_labels = None
        self.cluster_analysis = {}
        self.optimal_clusters = None
        
    def prepare_clustering_features(self, providers_df: pd.DataFrame, claims_df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for provider clustering analysis
        
        Args:
            providers_df: Provider demographic and specialty data
            claims_df: Claims data with amounts and procedure codes
            
        Returns:
            DataFrame with clustering features for each provider
        """
        self.logger.info("Engineering features for provider clustering analysis")
        
        try:
            # Aggregate claims data by provider
            provider_claims = claims_df.groupby('provider_id').agg({
                'claim_amount': ['count', 'sum', 'mean', 'std'],
                'procedure_code': 'nunique',
                'diagnosis_code': 'nunique'
            }).round(2)
            
            # Flatten column names
            provider_claims.columns = [
                'claims_count', 'claims_total_amount', 'claims_avg_amount', 
                'claim_amount_std', 'unique_procedures', 'unique_diagnoses'
            ]
            
            # Fill NaN standard deviations (providers with single claims)
            provider_claims['claim_amount_std'] = provider_claims['claim_amount_std'].fillna(0)
            
            # Merge with provider information
            clustering_data = providers_df.merge(
                provider_claims, 
                left_on='provider_id', 
                right_index=True, 
                how='inner'
            )
            
            # Engineer additional features
            clustering_data['avg_procedures_per_claim'] = (
                clustering_data['unique_procedures'] / clustering_data['claims_count']
            ).round(3)

            clustering_data['avg_diagnoses_per_claim'] = (
                clustering_data['unique_diagnoses'] / clustering_data['claims_count']
            ).round(3)

            clustering_data['procedure_concentration'] = (
                clustering_data['unique_procedures'] / clustering_data['claims_count']
            ).round(3)

            clustering_data['volume_percentile_in_specialty'] = (
                clustering_data.groupby('specialty')['claims_count']
                .rank(pct=True)
                .round(3)
            )
            
            self.logger.info(f"Prepared clustering features for {len(clustering_data)} providers")
            
            return clustering_data
            
        except Exception as e:
            self.logger.error(f"Error preparing clustering features: {str(e)}")
            raise
    
    def determine_optimal_clusters(self, features_df: pd.DataFrame, max_clusters: int = 10) -> int:
        """
        Determine optimal number of clusters using elbow method and silhouette analysis
        
        Args:
            features_df: DataFrame with clustering features
            max_clusters: Maximum number of clusters to evaluate
            
        Returns:
            Optimal number of clusters
        """
        self.logger.info(f"Determining optimal clusters (max: {max_clusters})")
        
        # Select numerical features for clustering
        numerical_features = features_df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove ID columns
        numerical_features = [col for col in numerical_features if 'id' not in col.lower()]
        
        X = features_df[numerical_features].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        # Calculate metrics for different cluster numbers
        inertias = []
        silhouette_scores = []
        calinski_scores = []
        
        cluster_range = range(2, max_clusters + 1)
        
        for n_clusters in cluster_range:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_scaled, cluster_labels))
            calinski_scores.append(calinski_harabasz_score(X_scaled, cluster_labels))
        
        # Find optimal clusters using silhouette score
        optimal_idx = np.argmax(silhouette_scores)
        optimal_clusters = cluster_range[optimal_idx]
        
        # Store analysis results
        self.cluster_analysis = {
            'cluster_range': list(cluster_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'calinski_scores': calinski_scores,
            'optimal_clusters': optimal_clusters,
            'features_used': numerical_features
        }
        
        self.optimal_clusters = optimal_clusters
        self.logger.info(f"Optimal clusters determined: {optimal_clusters}")
        
        return optimal_clusters
    
    def perform_clustering(self, features_df: pd.DataFrame, n_clusters: int = None) -> pd.DataFrame:
        """
        Perform K-means clustering on provider data
        
        Args:
            features_df: DataFrame with clustering features
            n_clusters: Number of clusters (if None, uses optimal)
            
        Returns:
            DataFrame with cluster assignments
        """
        if n_clusters is None:
            if self.optimal_clusters is None:
                n_clusters = self.determine_optimal_clusters(features_df)
            else:
                n_clusters = self.optimal_clusters
        
        self.logger.info(f"Performing K-means clustering with {n_clusters} clusters")
        
        try:
            # Select numerical features
            numerical_features = features_df.select_dtypes(include=[np.number]).columns.tolist()
            numerical_features = [col for col in numerical_features if 'id' not in col.lower()]
            
            X = features_df[numerical_features].fillna(0)
            X_scaled = self.scaler.fit_transform(X)
            
            # Perform clustering
            self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = self.kmeans_model.fit_predict(X_scaled)
            
            # Add cluster labels to original data
            clustered_data = features_df.copy()
            clustered_data['cluster'] = cluster_labels
            
            # Calculate cluster centers in original scale
            cluster_centers = self.scaler.inverse_transform(self.kmeans_model.cluster_centers_)
            cluster_centers_df = pd.DataFrame(
                cluster_centers, 
                columns=numerical_features,
                index=[f'Cluster_{i}' for i in range(n_clusters)]
            )
            
            # Store results
            self.cluster_labels = cluster_labels
            self.cluster_centers = cluster_centers_df
            
            # Log cluster summary
            cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
            self.logger.info(f"Clustering complete. Cluster sizes: {cluster_counts.to_dict()}")
            
            return clustered_data
            
        except Exception as e:
            self.logger.error(f"Error performing clustering: {str(e)}")
            raise
    
    def analyze_cluster_characteristics(self, clustered_data: pd.DataFrame) -> dict:
        """
        Analyze characteristics of each cluster
        
        Args:
            clustered_data: DataFrame with cluster assignments
            
        Returns:
            Dictionary with cluster analysis
        """
        self.logger.info("Analyzing cluster characteristics")
        
        analysis = {}
        
        # Overall cluster summary
        cluster_summary = clustered_data.groupby('cluster').agg({
            'claims_count': ['count', 'mean', 'median'],
            'claims_total_amount': ['mean', 'median', 'sum'],
            'avg_claim_amount': ['mean', 'median'],
            'unique_procedures': ['mean', 'median'],
            'specialty': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Mixed'
        }).round(2)
        
        analysis['cluster_summary'] = cluster_summary
        
        # Risk characteristics by cluster
        risk_analysis = {}
        for cluster_id in sorted(clustered_data['cluster'].unique()):
            cluster_data = clustered_data[clustered_data['cluster'] == cluster_id]
            
            risk_analysis[f'Cluster_{cluster_id}'] = {
                'size': len(cluster_data),
                'avg_total_amount': cluster_data['claims_total_amount'].mean(),
                'avg_claims_count': cluster_data['claims_count'].mean(),
                'dominant_specialty': cluster_data['specialty'].mode().iloc[0] if len(cluster_data['specialty'].mode()) > 0 else 'Mixed',
                'high_volume_providers': len(cluster_data[cluster_data['claims_count'] > cluster_data['claims_count'].quantile(0.9)]),
                'procedure_diversity': cluster_data['unique_procedures'].mean()
            }
        
        analysis['risk_characteristics'] = risk_analysis
        
        # Specialty distribution by cluster
        specialty_dist = clustered_data.groupby(['cluster', 'specialty']).size().unstack(fill_value=0)
        analysis['specialty_distribution'] = specialty_dist
        
        self.logger.info(f"Cluster analysis complete for {len(clustered_data['cluster'].unique())} clusters")
        
        return analysis
    
    def create_cluster_visualizations(self, clustered_data: pd.DataFrame) -> dict:
        """
        Create comprehensive visualizations for cluster analysis
        
        Args:
            clustered_data: DataFrame with cluster assignments
            
        Returns:
            Dictionary with visualization objects
        """
        self.logger.info("Creating cluster visualizations")
        
        visualizations = {}
        
        # 1. Cluster optimization plots
        if hasattr(self, 'cluster_analysis') and self.cluster_analysis:
            fig_optimization = make_subplots(
                rows=1, cols=3,
                subplot_titles=('Elbow Method', 'Silhouette Score', 'Calinski-Harabasz Score')
            )
            
            # Elbow method
            fig_optimization.add_trace(
                go.Scatter(
                    x=self.cluster_analysis['cluster_range'],
                    y=self.cluster_analysis['inertias'],
                    mode='lines+markers',
                    name='Inertia',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
            
            # Silhouette score
            fig_optimization.add_trace(
                go.Scatter(
                    x=self.cluster_analysis['cluster_range'],
                    y=self.cluster_analysis['silhouette_scores'],
                    mode='lines+markers',
                    name='Silhouette',
                    line=dict(color='green')
                ),
                row=1, col=2
            )
            
            # Calinski-Harabasz score
            fig_optimization.add_trace(
                go.Scatter(
                    x=self.cluster_analysis['cluster_range'],
                    y=self.cluster_analysis['calinski_scores'],
                    mode='lines+markers',
                    name='Calinski-Harabasz',
                    line=dict(color='red')
                ),
                row=1, col=3
            )
            
            # Highlight optimal
            optimal_k = self.cluster_analysis['optimal_clusters']
            for col in range(1, 4):
                fig_optimization.add_vline(
                    x=optimal_k, 
                    line_dash="dash", 
                    line_color="orange",
                    row=1, col=col
                )
            
            fig_optimization.update_layout(
                title=f"Cluster Optimization Analysis (Optimal: {optimal_k} clusters)",
                height=400,
                showlegend=False
            )
            
            visualizations['optimization'] = fig_optimization
        
        # 2. PCA visualization of clusters
        numerical_features = clustered_data.select_dtypes(include=[np.number]).columns.tolist()
        numerical_features = [col for col in numerical_features if 'id' not in col.lower() and col != 'cluster']
        
        X = clustered_data[numerical_features].fillna(0)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(StandardScaler().fit_transform(X))
        
        fig_pca = px.scatter(
            x=X_pca[:, 0], 
            y=X_pca[:, 1],
            color=clustered_data['cluster'].astype(str),
            title=f"Provider Clusters in PCA Space (Explained Variance: {pca.explained_variance_ratio_.sum():.1%})",
            labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', 
                   'y': f'PC2 ({pca.explained_variance_ratio_[1]:.1%})'},
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        
        visualizations['pca_clusters'] = fig_pca
        
        # 3. Cluster characteristics heatmap
        cluster_stats = clustered_data.groupby('cluster')[numerical_features].mean()
        
        fig_heatmap = px.imshow(
            cluster_stats.T,
            labels=dict(x="Cluster", y="Features", color="Normalized Value"),
            title="Cluster Characteristics Heatmap",
            color_continuous_scale="RdYlBu_r"
        )
        
        visualizations['characteristics_heatmap'] = fig_heatmap
        
        # 4. Cluster size and total amount comparison
        cluster_summary = clustered_data.groupby('cluster').agg({
            'provider_id': 'count',
            'claims_total_amount': 'sum'
        }).rename(columns={'provider_id': 'provider_count'})
        
        fig_summary = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Providers per Cluster', 'Total Amount by Cluster'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        fig_summary.add_trace(
            go.Bar(
                x=cluster_summary.index,
                y=cluster_summary['provider_count'],
                name='Provider Count',
                marker_color='lightblue'
            ),
            row=1, col=1
        )
        
        fig_summary.add_trace(
            go.Bar(
                x=cluster_summary.index,
                y=cluster_summary['claims_total_amount'],
                name='Total Amount',
                marker_color='lightcoral'
            ),
            row=1, col=2
        )
        
        fig_summary.update_layout(
            title="Cluster Business Impact Analysis",
            height=400,
            showlegend=False
        )
        
        visualizations['business_impact'] = fig_summary
        
        self.logger.info(f"Created {len(visualizations)} cluster visualizations")
        
        return visualizations
    
    def generate_cluster_insights(self, clustered_data: pd.DataFrame, analysis: dict) -> dict:
        """
        Generate business insights from cluster analysis
        
        Args:
            clustered_data: DataFrame with cluster assignments
            analysis: Cluster analysis results
            
        Returns:
            Dictionary with actionable insights
        """
        self.logger.info("Generating business insights from cluster analysis")
        
        insights = {
            'summary': {},
            'risk_assessment': {},
            'recommendations': {}
        }
        
        # Summary insights
        n_clusters = len(clustered_data['cluster'].unique())
        total_providers = len(clustered_data)
        
        insights['summary'] = {
            'total_clusters_identified': n_clusters,
            'total_providers_analyzed': total_providers,
            'clustering_approach': 'K-means with feature scaling',
            'primary_segmentation_factors': ['billing_volume', 'claim_amounts', 'procedure_diversity']
        }
        
        # Risk assessment by cluster
        for cluster_id in sorted(clustered_data['cluster'].unique()):
            cluster_data = clustered_data[clustered_data['cluster'] == cluster_id]
            
            # Calculate risk indicators
            avg_claim_amount = cluster_data['avg_claim_amount'].mean()
            total_volume = cluster_data['claims_total_amount'].sum()
            provider_count = len(cluster_data)
            
            # Risk level determination
            if avg_claim_amount > clustered_data['avg_claim_amount'].quantile(0.8):
                risk_level = "High"
            elif avg_claim_amount > clustered_data['avg_claim_amount'].quantile(0.6):
                risk_level = "Medium"
            else:
                risk_level = "Low"
            
            insights['risk_assessment'][f'Cluster_{cluster_id}'] = {
                'risk_level': risk_level,
                'provider_count': provider_count,
                'avg_claim_amount': round(avg_claim_amount, 2),
                'total_volume': round(total_volume, 2),
                'investigation_priority': 'High' if risk_level == 'High' and provider_count > 10 else 'Medium' if risk_level in ['High', 'Medium'] else 'Low'
            }
        
        # Business recommendations
        high_risk_clusters = [k for k, v in insights['risk_assessment'].items() if v['risk_level'] == 'High']
        
        insights['recommendations'] = {
            'immediate_actions': [
                f"Focus fraud investigation resources on {len(high_risk_clusters)} high-risk clusters",
                "Implement enhanced monitoring for providers in high-risk clusters",
                "Review billing patterns of providers with above-average claim amounts"
            ],
            'strategic_actions': [
                "Develop cluster-specific fraud detection rules",
                "Create provider risk scoring based on cluster characteristics",
                "Establish cluster-based investigation workflows"
            ],
            'monitoring_suggestions': [
                "Track cluster migration patterns over time",
                "Monitor for new providers entering high-risk clusters",
                "Analyze seasonal changes in cluster characteristics"
            ]
        }
        
        self.logger.info("Business insights generation complete")
        
        return insights

# Example usage and testing
if __name__ == "__main__":
    # This would be run as part of the main analytics pipeline
    print("Provider Clustering Analytics Module")
    print("Ready for integration with healthcare fraud detection platform")