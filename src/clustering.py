"""
Clustering Module - CareerGapAI

This module implements unsupervised learning for career path clustering.
Uses KMeans (primary) and Hierarchical Clustering (validation) to identify
career trajectory patterns and employee segmentation groups.

Career Cluster Types Expected:
- Fast-track performers
- Stable long-term contributors
- Early-career explorers
- Promotion-stalled employees
- High-risk stagnation profiles

Author: CareerGapAI Team
Date: 2026
Version: 1.0.0
"""

import logging
from typing import Tuple, Optional, Dict, List, Any
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class CareerPathClustering:
    """
    Career path clustering using KMeans and Hierarchical methods.
    
    This class identifies groups of employees with similar career progression patterns,
    enabling targeted retention and career development interventions.
    
    Attributes:
        X (np.ndarray): Feature matrix for clustering
        df_features (pd.DataFrame): Original feature dataframe
        feature_names (list): Names of features used in clustering
        kmeans_model (KMeans): Fitted KMeans model
        hierarchical_model (AgglomerativeClustering): Fitted Hierarchical model
        optimal_k (int): Optimal number of clusters determined
        evaluation_metrics (Dict): Clustering quality metrics
        cluster_labels (Dict): Assigned cluster labels for each employee
    """
    
    def __init__(self, feature_matrix: pd.DataFrame, feature_names: Optional[List[str]] = None):
        """
        Initialize the CareerPathClustering class.
        
        Args:
            feature_matrix (pd.DataFrame): Feature matrix for clustering (already scaled)
            feature_names (List[str], optional): Names of features used
        """
        self.df_features = feature_matrix.copy()
        self.X = feature_matrix.values
        self.feature_names = feature_names or feature_matrix.columns.tolist()
        
        self.kmeans_model = None
        self.hierarchical_model = None
        self.optimal_k = None
        self.evaluation_metrics = {}
        self.cluster_labels = {}
        self.cluster_profiles = {}
        
        logger.info(f"CareerPathClustering initialized with {self.X.shape[0]} samples, "
                   f"{self.X.shape[1]} features")
    
    def determine_optimal_clusters(self, k_range: range = range(2, 11),
                                  method: str = 'elbow_silhouette') -> Dict[str, Any]:
        """
        Determine optimal number of clusters using multiple methods.
        
        Methods:
        - 'elbow': Elbow method on inertia
        - 'silhouette': Silhouette score
        - 'elbow_silhouette': Combination of both (balanced approach)
        
        Args:
            k_range (range): Range of k values to test
            method (str): Method for determining optimal k
        
        Returns:
            Dict: Analysis results with recommended k
        """
        inertias = []
        silhouette_scores = []
        davies_bouldin_scores = []
        
        logger.info(f"Testing cluster sizes from {k_range.start} to {k_range.stop-1}...")
        
        for k in k_range:
            # Fit KMeans
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.X)
            
            # Calculate metrics
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.X, labels))
            davies_bouldin_scores.append(davies_bouldin_score(self.X, labels))
            
            logger.info(f"  k={k}: Inertia={kmeans.inertia_:.2f}, "
                       f"Silhouette={silhouette_scores[-1]:.3f}, "
                       f"Davies-Bouldin={davies_bouldin_scores[-1]:.3f}")
        
        # Determine optimal k
        if method == 'elbow':
            # Find elbow point (steepest change in inertia)
            differences = np.diff(inertias)
            optimal_k = np.argmax(np.diff(differences)) + 2
        
        elif method == 'silhouette':
            # Maximum silhouette score
            optimal_k = k_range.start + np.argmax(silhouette_scores)
        
        elif method == 'elbow_silhouette':
            # Balanced approach: combine both methods
            # Normalize metrics
            inertia_normalized = (np.array(inertias) - min(inertias)) / (max(inertias) - min(inertias))
            silhouette_normalized = 1 - (np.array(silhouette_scores) - min(silhouette_scores)) / \
                                      (max(silhouette_scores) - min(silhouette_scores))
            
            combined_score = 0.5 * inertia_normalized + 0.5 * silhouette_normalized
            optimal_k = k_range.start + np.argmin(combined_score)
        
        self.optimal_k = optimal_k
        
        results = {
            'k_values': list(k_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'davies_bouldin_scores': davies_bouldin_scores,
            'optimal_k': optimal_k,
            'method': method
        }
        
        logger.info(f"Optimal number of clusters determined: k={optimal_k}")
        
        return results
    
    def fit_kmeans(self, n_clusters: Optional[int] = None, random_state: int = 42) -> KMeans:
        """
        Fit KMeans clustering model.
        
        Args:
            n_clusters (int, optional): Number of clusters. Uses optimal_k if not provided.
            random_state (int): Random seed for reproducibility
        
        Returns:
            KMeans: Fitted KMeans model
        """
        if n_clusters is None:
            if self.optimal_k is None:
                raise ValueError("Provide n_clusters or run determine_optimal_clusters() first")
            n_clusters = self.optimal_k
        
        logger.info(f"Fitting KMeans with k={n_clusters}...")
        
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=random_state,
                                   n_init=10, max_iter=300)
        kmeans_labels = self.kmeans_model.fit_predict(self.X)
        
        # Calculate evaluation metrics
        silhouette = silhouette_score(self.X, kmeans_labels)
        davies_bouldin = davies_bouldin_score(self.X, kmeans_labels)
        calinski_harabasz = calinski_harabasz_score(self.X, kmeans_labels)
        
        self.evaluation_metrics['kmeans'] = {
            'n_clusters': n_clusters,
            'inertia': self.kmeans_model.inertia_,
            'silhouette_score': silhouette,
            'davies_bouldin_score': davies_bouldin,
            'calinski_harabasz_score': calinski_harabasz
        }
        
        self.cluster_labels['kmeans'] = kmeans_labels
        
        logger.info(f"KMeans fit complete. Silhouette Score: {silhouette:.3f}")
        
        return self.kmeans_model
    
    def fit_hierarchical(self, n_clusters: Optional[int] = None,
                        linkage_method: str = 'ward') -> AgglomerativeClustering:
        """
        Fit Hierarchical Clustering model (for validation).
        
        Linkage methods:
        - 'ward': Minimizes variance (recommended)
        - 'complete': Maximum distance between clusters
        - 'average': Average distance between clusters
        - 'single': Minimum distance between clusters
        
        Args:
            n_clusters (int, optional): Number of clusters. Uses optimal_k if not provided.
            linkage_method (str): Linkage method
        
        Returns:
            AgglomerativeClustering: Fitted model
        """
        if n_clusters is None:
            if self.optimal_k is None:
                raise ValueError("Provide n_clusters or run determine_optimal_clusters() first")
            n_clusters = self.optimal_k
        
        logger.info(f"Fitting Hierarchical Clustering with k={n_clusters}, "
                   f"linkage={linkage_method}...")
        
        self.hierarchical_model = AgglomerativeClustering(n_clusters=n_clusters,
                                                         linkage=linkage_method)
        hierarchical_labels = self.hierarchical_model.fit_predict(self.X)
        
        # Calculate evaluation metrics
        silhouette = silhouette_score(self.X, hierarchical_labels)
        davies_bouldin = davies_bouldin_score(self.X, hierarchical_labels)
        calinski_harabasz = calinski_harabasz_score(self.X, hierarchical_labels)
        
        self.evaluation_metrics['hierarchical'] = {
            'n_clusters': n_clusters,
            'linkage_method': linkage_method,
            'silhouette_score': silhouette,
            'davies_bouldin_score': davies_bouldin,
            'calinski_harabasz_score': calinski_harabasz
        }
        
        self.cluster_labels['hierarchical'] = hierarchical_labels
        
        logger.info(f"Hierarchical Clustering fit complete. Silhouette Score: {silhouette:.3f}")
        
        return self.hierarchical_model
    
    def get_cluster_assignments(self, method: str = 'kmeans') -> np.ndarray:
        """
        Get cluster assignments for all samples.
        
        Args:
            method (str): 'kmeans' or 'hierarchical'
        
        Returns:
            np.ndarray: Cluster assignments
        """
        if method not in self.cluster_labels:
            raise ValueError(f"Model {method} not fitted yet")
        
        return self.cluster_labels[method]
    
    def profile_clusters(self, original_df: pd.DataFrame,
                        method: str = 'kmeans') -> pd.DataFrame:
        """
        Create comprehensive profiles of identified clusters.
        
        Analyzes:
        - Average career metrics per cluster
        - Size and composition
        - Key characteristics
        
        Args:
            original_df (pd.DataFrame): Original dataset with all features
            method (str): 'kmeans' or 'hierarchical'
        
        Returns:
            pd.DataFrame: Cluster profiles
        """
        if method not in self.cluster_labels:
            raise ValueError(f"Model {method} not fitted yet")
        
        labels = self.cluster_labels[method]
        df_with_clusters = original_df.copy()
        df_with_clusters['Cluster'] = labels
        
        # Create profiles
        profiles = []
        
        for cluster_id in sorted(df_with_clusters['Cluster'].unique()):
            cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster_id]
            
            profile = {
                'Cluster': cluster_id,
                'Size': len(cluster_data),
                'Percentage': (len(cluster_data) / len(df_with_clusters)) * 100
            }
            
            # Add statistics for numeric columns
            numeric_cols = cluster_data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                profile[f'{col}_mean'] = cluster_data[col].mean()
                profile[f'{col}_median'] = cluster_data[col].median()
                profile[f'{col}_std'] = cluster_data[col].std()
            
            profiles.append(profile)
        
        profiles_df = pd.DataFrame(profiles)
        self.cluster_profiles[method] = profiles_df
        
        logger.info(f"Generated profiles for {len(profiles)} clusters using {method}")
        
        return profiles_df
    
    def label_clusters(self, original_df: pd.DataFrame,
                      method: str = 'kmeans') -> Dict[int, str]:
        """
        Assign meaningful labels to clusters based on career characteristics.
        
        Labels based on:
        - Promotion gap ratio
        - Role stagnation index
        - Training intensity
        - Manager stability
        - Years at company
        
        Returns:
            Dict: Mapping of cluster_id to cluster label
        """
        if method not in self.cluster_labels:
            raise ValueError(f"Model {method} not fitted yet")
        
        labels = self.cluster_labels[method]
        df_with_clusters = original_df.copy()
        df_with_clusters['Cluster'] = labels
        
        cluster_labels_map = {}
        
        for cluster_id in sorted(df_with_clusters['Cluster'].unique()):
            cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster_id]
            
            # Calculate key metrics
            avg_tenure = cluster_data['YearsAtCompany'].mean() if 'YearsAtCompany' in cluster_data else 0
            avg_promotion_gap = cluster_data['YearsSinceLastPromotion'].mean() \
                if 'YearsSinceLastPromotion' in cluster_data else 0
            avg_training = cluster_data['TrainingTimesLastYear'].mean() \
                if 'TrainingTimesLastYear' in cluster_data else 0
            avg_role_years = cluster_data['YearsInCurrentRole'].mean() \
                if 'YearsInCurrentRole' in cluster_data else 0
            
            # Labeling logic
            if avg_promotion_gap > 5 and avg_role_years > 4:
                label = "High-Risk Stagnation"
            elif avg_promotion_gap < 2 and avg_tenure < 5:
                label = "Early-Career Explorers"
            elif avg_tenure > 10 and avg_training > 3:
                label = "Stable High-Growth Contributors"
            elif avg_tenure > 8 and avg_promotion_gap < 3:
                label = "Fast-Track Performers"
            elif avg_tenure > 10 and avg_promotion_gap > 3:
                label = "Stable Long-Term Contributors"
            else:
                label = f"Career Cluster {cluster_id}"
            
            cluster_labels_map[cluster_id] = label
            logger.info(f"Cluster {cluster_id}: {label} (n={len(cluster_data)}, "
                       f"avg_tenure={avg_tenure:.1f} years)")
        
        return cluster_labels_map
    
    def get_evaluation_metrics(self) -> Dict[str, Any]:
        """
        Get all clustering evaluation metrics.
        
        Returns:
            Dict: Evaluation metrics for all fitted models
        """
        return self.evaluation_metrics
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare KMeans and Hierarchical clustering performance.
        
        Returns:
            pd.DataFrame: Comparison table
        """
        comparison = []
        
        for model_name in ['kmeans', 'hierarchical']:
            if model_name in self.evaluation_metrics:
                metrics = self.evaluation_metrics[model_name].copy()
                metrics['Model'] = model_name
                comparison.append(metrics)
        
        comparison_df = pd.DataFrame(comparison)
        logger.info(f"Model Comparison:\n{comparison_df}")
        
        return comparison_df
    
    def plot_elbow_curve(self, elbow_results: Dict[str, Any],
                        figsize: Tuple[int, int] = (12, 5)) -> None:
        """
        Plot elbow curve for optimal k determination.
        
        Args:
            elbow_results (Dict): Results from determine_optimal_clusters()
            figsize (Tuple): Figure size
        """
        k_values = elbow_results['k_values']
        inertias = elbow_results['inertias']
        silhouette = elbow_results['silhouette_scores']
        optimal_k = elbow_results['optimal_k']
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Elbow plot
        axes[0].plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
        axes[0].axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
        axes[0].set_xlabel('Number of Clusters (k)', fontsize=11)
        axes[0].set_ylabel('Inertia', fontsize=11)
        axes[0].set_title('Elbow Method', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Silhouette plot
        axes[1].plot(k_values, silhouette, 'go-', linewidth=2, markersize=8)
        axes[1].axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
        axes[1].set_xlabel('Number of Clusters (k)', fontsize=11)
        axes[1].set_ylabel('Silhouette Score', fontsize=11)
        axes[1].set_title('Silhouette Analysis', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        logger.info("Plotted elbow and silhouette curves")
        return fig
    
    def plot_cluster_distribution(self, method: str = 'kmeans',
                                 figsize: Tuple[int, int] = (10, 5)) -> None:
        """
        Plot distribution of employees across clusters.
        
        Args:
            method (str): 'kmeans' or 'hierarchical'
            figsize (Tuple): Figure size
        """
        if method not in self.cluster_labels:
            logger.warning(f"Model {method} not fitted")
            return
        
        labels = self.cluster_labels[method]
        unique, counts = np.unique(labels, return_counts=True)
        
        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(unique, counts, color='steelblue', edgecolor='black', alpha=0.7)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel('Cluster', fontsize=11)
        ax.set_ylabel('Number of Employees', fontsize=11)
        ax.set_title(f'Cluster Distribution ({method.upper()})', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        logger.info(f"Plotted cluster distribution for {method}")
        return fig
    
    def get_cluster_summary(self, method: str = 'kmeans') -> str:
        """
        Generate comprehensive clustering summary report.
        
        Args:
            method (str): 'kmeans' or 'hierarchical'
        
        Returns:
            str: Formatted clustering report
        """
        if method not in self.evaluation_metrics:
            return f"Model {method} not fitted yet"
        
        report = []
        report.append("=" * 80)
        report.append(f"CLUSTERING REPORT - {method.upper()}")
        report.append("=" * 80)
        
        metrics = self.evaluation_metrics[method]
        report.append(f"\nNumber of Clusters: {metrics['n_clusters']}")
        
        if 'inertia' in metrics:
            report.append(f"Inertia: {metrics['inertia']:.2f}")
        
        report.append(f"Silhouette Score: {metrics['silhouette_score']:.3f}")
        report.append(f"Davies-Bouldin Score: {metrics['davies_bouldin_score']:.3f} (lower is better)")
        report.append(f"Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.2f}")
        
        if method in self.cluster_profiles:
            report.append(f"\nCluster Profiles:")
            report.append(self.cluster_profiles[method].to_string())
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)


if __name__ == "__main__":
    """
    Example usage of the CareerPathClustering class.
    """
    # Load preprocessed data
    # from data_loader import DataLoader
    # from preprocessing import DataPreprocessor
    
    # loader = DataLoader('data')
    # df = loader.load_data('sample.csv')
    # preprocessor = DataPreprocessor(df)
    # df_processed = preprocessor.apply_full_preprocessing_pipeline()
    
    # Initialize clustering
    # clustering = CareerPathClustering(df_processed)
    
    # Determine optimal clusters
    # elbow_results = clustering.determine_optimal_clusters(k_range=range(2, 11))
    
    # Fit models
    # clustering.fit_kmeans()
    # clustering.fit_hierarchical()
    
    # Compare models
    # comparison = clustering.compare_models()
    
    # Label clusters
    # cluster_labels = clustering.label_clusters(df)
    
    # Get summary
    # summary = clustering.get_cluster_summary()
    # print(summary)
    
    print("CareerPathClustering module ready for use.")
