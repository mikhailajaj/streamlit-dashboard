"""
Advanced Feature Engineering for AWS Cost Analysis
Creates derived features that improve ML model performance and interpretability
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class AWSFeatureEngineer:
    """
    Feature engineering for AWS resources.
    
    Creates three categories of features:
    1. Cost Efficiency Features - Understand cost per unit of resource
    2. Context Features - Business context (environment, owner, instance family)
    3. Comparative Features - How does this resource compare to peers?
    """
    
    def __init__(self):
        self.feature_metadata = {}
        self.is_fitted = False
        
    def engineer_ec2_features(self, ec2_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features for EC2 instances.
        
        Args:
            ec2_df: EC2 DataFrame with raw features
            
        Returns:
            DataFrame with additional engineered features
        """
        df = ec2_df.copy()
        
        # Handle missing values for critical columns
        if 'CostUSD' not in df.columns:
            raise ValueError("CostUSD column is required")
        
        # Drop completely NULL rows
        df = df.dropna(how='all')
        
        # === COST EFFICIENCY FEATURES (HIGH IMPACT) ===
        
        # Cost per CPU hour (accounting for utilization)
        if 'CPUUtilization' in df.columns:
            df['cost_per_cpu_utilized'] = df.apply(
                lambda row: row['CostUSD'] / (row['CPUUtilization'] / 100 + 0.01) 
                if pd.notna(row['CPUUtilization']) and pd.notna(row['CostUSD']) else np.nan,
                axis=1
            )
        
        # Cost per memory GB (if memory utilization available)
        if 'MemoryUtilization' in df.columns:
            df['cost_per_memory_utilized'] = df.apply(
                lambda row: row['CostUSD'] / (row['MemoryUtilization'] / 100 + 0.01)
                if pd.notna(row['MemoryUtilization']) and pd.notna(row['CostUSD']) else np.nan,
                axis=1
            )
        
        # Idle cost (money wasted on unused capacity)
        if 'CPUUtilization' in df.columns:
            df['idle_cost'] = df.apply(
                lambda row: row['CostUSD'] * (1 - row['CPUUtilization'] / 100)
                if pd.notna(row['CPUUtilization']) and pd.notna(row['CostUSD']) else np.nan,
                axis=1
            )
        
        # Waste score (percentage of cost that's wasted)
        if 'idle_cost' in df.columns:
            df['waste_score'] = df.apply(
                lambda row: (row['idle_cost'] / row['CostUSD']) * 100
                if pd.notna(row['idle_cost']) and row['CostUSD'] > 0 else 0,
                axis=1
            )
        
        # Utilization balance (how balanced are CPU and Memory usage?)
        if 'CPUUtilization' in df.columns and 'MemoryUtilization' in df.columns:
            df['utilization_balance'] = df.apply(
                lambda row: min(row['CPUUtilization'], row['MemoryUtilization']) / 
                           (max(row['CPUUtilization'], row['MemoryUtilization']) + 0.01)
                if pd.notna(row['CPUUtilization']) and pd.notna(row['MemoryUtilization']) else np.nan,
                axis=1
            )
        
        # Efficiency score (combined CPU and Memory utilization)
        if 'CPUUtilization' in df.columns and 'MemoryUtilization' in df.columns:
            df['efficiency_score'] = df.apply(
                lambda row: (row['CPUUtilization'] + row['MemoryUtilization']) / 2
                if pd.notna(row['CPUUtilization']) and pd.notna(row['MemoryUtilization']) else np.nan,
                axis=1
            )
        
        # === CONTEXT FEATURES (HIGH IMPACT) ===
        
        # Extract instance family (c5, r5, m5, t3, etc.)
        if 'InstanceType' in df.columns:
            df['instance_family'] = df['InstanceType'].apply(
                lambda x: self._extract_instance_family(x) if pd.notna(x) else 'unknown'
            )
        
        # Extract instance size (xlarge, large, medium, etc.)
        if 'InstanceType' in df.columns:
            df['instance_size'] = df['InstanceType'].apply(
                lambda x: self._extract_instance_size(x) if pd.notna(x) else 'unknown'
            )
        
        # Is production environment?
        if 'Tags' in df.columns:
            df['is_prod'] = df['Tags'].apply(
                lambda x: self._is_production(x) if pd.notna(x) else False
            )
        
        # Extract owner from tags
        if 'Tags' in df.columns:
            df['owner'] = df['Tags'].apply(
                lambda x: self._extract_owner(x) if pd.notna(x) else 'unknown'
            )
        
        # Extract environment from tags
        if 'Tags' in df.columns:
            df['environment'] = df['Tags'].apply(
                lambda x: self._extract_environment(x) if pd.notna(x) else 'unknown'
            )
        
        # Is instance stopped? (wasting money on storage)
        if 'State' in df.columns:
            df['is_stopped'] = df['State'].apply(
                lambda x: 1 if x == 'stopped' else 0 if pd.notna(x) else 0
            )
        
        # === COMPARATIVE FEATURES (MEDIUM IMPACT) ===
        
        # Compare to peers (same instance type)
        if 'InstanceType' in df.columns and 'CPUUtilization' in df.columns:
            df = self._add_peer_comparison_features(df)
        
        # Cost percentile within instance type
        if 'InstanceType' in df.columns and 'CostUSD' in df.columns:
            df['cost_percentile'] = df.groupby('InstanceType')['CostUSD'].rank(pct=True)
        
        # Utilization percentile within instance type
        if 'InstanceType' in df.columns and 'CPUUtilization' in df.columns:
            df['cpu_percentile'] = df.groupby('InstanceType')['CPUUtilization'].rank(pct=True)
        
        # === DERIVED BUSINESS METRICS ===
        
        # Monthly cost projection
        df['monthly_cost_projected'] = df['CostUSD'] * 1  # Already monthly in the data
        
        # Annual cost projection
        df['annual_cost_projected'] = df['CostUSD'] * 12
        
        # Cost category (low, medium, high)
        if 'CostUSD' in df.columns:
            df['cost_category'] = pd.cut(
                df['CostUSD'],
                bins=[0, 0.20, 0.50, np.inf],
                labels=['low', 'medium', 'high'],
                include_lowest=True
            )
        
        # Utilization category
        if 'CPUUtilization' in df.columns:
            df['utilization_category'] = pd.cut(
                df['CPUUtilization'],
                bins=[0, 25, 50, 75, 100],
                labels=['idle', 'low', 'medium', 'high'],
                include_lowest=True
            )
        
        # Network intensity (relative network usage)
        if 'NetworkIn_Bps' in df.columns and 'NetworkOut_Bps' in df.columns:
            df['network_total'] = df['NetworkIn_Bps'] + df['NetworkOut_Bps']
            df['network_intensity'] = pd.cut(
                df['network_total'],
                bins=[0, 100000, 500000, np.inf],
                labels=['low', 'medium', 'high'],
                include_lowest=True
            )
        
        self.is_fitted = True
        return df
    
    def engineer_s3_features(self, s3_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features for S3 buckets.
        
        Args:
            s3_df: S3 DataFrame with raw features
            
        Returns:
            DataFrame with additional engineered features
        """
        df = s3_df.copy()
        
        # Drop completely NULL rows
        df = df.dropna(how='all')
        
        if 'MonthlyCostUSD' not in df.columns:
            raise ValueError("MonthlyCostUSD column is required")
        
        # === COST EFFICIENCY FEATURES ===
        
        # Cost per GB
        if 'TotalSizeGB' in df.columns:
            df['cost_per_gb'] = df.apply(
                lambda row: row['MonthlyCostUSD'] / (row['TotalSizeGB'] + 0.01)
                if pd.notna(row['TotalSizeGB']) and pd.notna(row['MonthlyCostUSD']) else np.nan,
                axis=1
            )
        
        # Cost per object
        if 'ObjectCount' in df.columns:
            df['cost_per_object'] = df.apply(
                lambda row: row['MonthlyCostUSD'] / (row['ObjectCount'] + 1)
                if pd.notna(row['ObjectCount']) and pd.notna(row['MonthlyCostUSD']) else np.nan,
                axis=1
            )
        
        # Storage density (objects per GB)
        if 'ObjectCount' in df.columns and 'TotalSizeGB' in df.columns:
            df['storage_density'] = df.apply(
                lambda row: row['ObjectCount'] / (row['TotalSizeGB'] + 0.01)
                if pd.notna(row['ObjectCount']) and pd.notna(row['TotalSizeGB']) else np.nan,
                axis=1
            )
        
        # === CONTEXT FEATURES ===
        
        # Is using optimal storage class?
        if 'StorageClass' in df.columns:
            df['is_standard_storage'] = df['StorageClass'].apply(
                lambda x: 1 if x == 'STANDARD' else 0 if pd.notna(x) else 0
            )
        
        # Has encryption enabled?
        if 'Encryption' in df.columns:
            df['has_encryption'] = df['Encryption'].apply(
                lambda x: 1 if x == 'Enabled' else 0 if pd.notna(x) else 0
            )
        
        # Extract bucket purpose from name (if possible)
        if 'BucketName' in df.columns:
            df['bucket_purpose'] = df['BucketName'].apply(
                lambda x: self._extract_bucket_purpose(x) if pd.notna(x) else 'unknown'
            )
        
        # === COMPARATIVE FEATURES ===
        
        # Cost percentile
        df['cost_percentile'] = df['MonthlyCostUSD'].rank(pct=True)
        
        # Size percentile
        if 'TotalSizeGB' in df.columns:
            df['size_percentile'] = df['TotalSizeGB'].rank(pct=True)
        
        # === OPTIMIZATION OPPORTUNITIES ===
        
        # Lifecycle policy candidate (STANDARD storage with high cost)
        if 'StorageClass' in df.columns:
            df['lifecycle_candidate'] = df.apply(
                lambda row: 1 if (row.get('StorageClass') == 'STANDARD' and 
                                  row.get('MonthlyCostUSD', 0) > 10) else 0,
                axis=1
            )
        
        # High cost per GB (inefficient storage)
        if 'cost_per_gb' in df.columns:
            df['high_cost_per_gb'] = df['cost_per_gb'].apply(
                lambda x: 1 if pd.notna(x) and x > 0.05 else 0  # $0.05 per GB threshold
            )
        
        # === DERIVED BUSINESS METRICS ===
        
        # Annual cost projection
        df['annual_cost_projected'] = df['MonthlyCostUSD'] * 12
        
        # Daily cost
        df['daily_cost'] = df['MonthlyCostUSD'] / 30
        
        # Cost category
        df['cost_category'] = pd.cut(
            df['MonthlyCostUSD'],
            bins=[0, 10, 50, 100, np.inf],
            labels=['low', 'medium', 'high', 'critical'],
            include_lowest=True
        )
        
        # Size category
        if 'TotalSizeGB' in df.columns:
            df['size_category'] = pd.cut(
                df['TotalSizeGB'],
                bins=[0, 100, 1000, 10000, np.inf],
                labels=['small', 'medium', 'large', 'xlarge'],
                include_lowest=True
            )
        
        return df
    
    def _extract_instance_family(self, instance_type: str) -> str:
        """Extract instance family from instance type (e.g., 'c5' from 'c5.xlarge')"""
        if pd.isna(instance_type):
            return 'unknown'
        
        parts = instance_type.split('.')
        if len(parts) > 0:
            # Extract family (e.g., 'c5', 'r5', 'm5', 't3')
            family = ''.join([c for c in parts[0] if not c.isdigit()])
            generation = ''.join([c for c in parts[0] if c.isdigit()])
            return f"{family}{generation}" if generation else family
        
        return 'unknown'
    
    def _extract_instance_size(self, instance_type: str) -> str:
        """Extract instance size from instance type (e.g., 'xlarge' from 'c5.xlarge')"""
        if pd.isna(instance_type):
            return 'unknown'
        
        parts = instance_type.split('.')
        if len(parts) > 1:
            return parts[1]
        
        return 'unknown'
    
    def _is_production(self, tags: str) -> bool:
        """Check if Environment tag indicates production"""
        if pd.isna(tags):
            return False
        
        tags_lower = str(tags).lower()
        return 'prod' in tags_lower or 'production' in tags_lower
    
    def _extract_owner(self, tags: str) -> str:
        """Extract owner from tags"""
        if pd.isna(tags):
            return 'unknown'
        
        # Parse tags (format: "Owner=Alice,Environment=Dev")
        tag_pairs = str(tags).split(',')
        for pair in tag_pairs:
            if '=' in pair:
                key, value = pair.split('=', 1)
                if key.strip().lower() == 'owner':
                    return value.strip()
        
        return 'unknown'
    
    def _extract_environment(self, tags: str) -> str:
        """Extract environment from tags"""
        if pd.isna(tags):
            return 'unknown'
        
        # Parse tags
        tag_pairs = str(tags).split(',')
        for pair in tag_pairs:
            if '=' in pair:
                key, value = pair.split('=', 1)
                if key.strip().lower() == 'environment':
                    return value.strip()
        
        return 'unknown'
    
    def _extract_bucket_purpose(self, bucket_name: str) -> str:
        """Infer bucket purpose from name"""
        if pd.isna(bucket_name):
            return 'unknown'
        
        name_lower = str(bucket_name).lower()
        
        if 'log' in name_lower:
            return 'logs'
        elif 'backup' in name_lower:
            return 'backups'
        elif 'data' in name_lower:
            return 'data'
        elif 'static' in name_lower or 'asset' in name_lower:
            return 'static_assets'
        elif 'archive' in name_lower:
            return 'archive'
        else:
            return 'general'
    
    def _add_peer_comparison_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features comparing each instance to its peers (same instance type).
        """
        # Calculate peer group statistics
        peer_stats = df.groupby('InstanceType').agg({
            'CPUUtilization': ['mean', 'std'],
            'CostUSD': ['mean', 'std']
        }).reset_index()
        
        peer_stats.columns = [
            'InstanceType', 
            'peer_mean_cpu', 'peer_std_cpu',
            'peer_mean_cost', 'peer_std_cost'
        ]
        
        # Merge back to original DataFrame
        df = df.merge(peer_stats, on='InstanceType', how='left')
        
        # Calculate differences from peer mean
        df['cpu_vs_peers'] = df['CPUUtilization'] - df['peer_mean_cpu']
        df['cost_vs_peers'] = df['CostUSD'] - df['peer_mean_cost']
        
        # Calculate z-scores (how many standard deviations from mean)
        df['cpu_zscore'] = df.apply(
            lambda row: (row['CPUUtilization'] - row['peer_mean_cpu']) / (row['peer_std_cpu'] + 0.01)
            if pd.notna(row['peer_std_cpu']) else 0,
            axis=1
        )
        
        df['cost_zscore'] = df.apply(
            lambda row: (row['CostUSD'] - row['peer_mean_cost']) / (row['peer_std_cost'] + 0.01)
            if pd.notna(row['peer_std_cost']) else 0,
            axis=1
        )
        
        return df
    
    def get_feature_importance_guide(self) -> Dict[str, List[str]]:
        """
        Return guide of which features are most important for different use cases.
        
        Returns:
            Dictionary mapping use cases to important features
        """
        return {
            'cost_optimization': [
                'idle_cost',
                'waste_score',
                'cost_per_cpu_utilized',
                'is_stopped',
                'utilization_category'
            ],
            'anomaly_detection': [
                'cpu_zscore',
                'cost_zscore',
                'cpu_vs_peers',
                'cost_vs_peers',
                'waste_score'
            ],
            'clustering': [
                'efficiency_score',
                'cost_category',
                'utilization_balance',
                'instance_family',
                'environment'
            ],
            'forecasting': [
                'monthly_cost_projected',
                'is_prod',
                'instance_family',
                'environment'
            ]
        }
    
    def get_feature_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate summary statistics for engineered features.
        
        Args:
            df: DataFrame with engineered features
            
        Returns:
            Summary DataFrame with feature statistics
        """
        # Identify numeric engineered features
        engineered_features = [
            'cost_per_cpu_utilized', 'cost_per_memory_utilized', 'idle_cost',
            'waste_score', 'utilization_balance', 'efficiency_score',
            'cost_percentile', 'cpu_percentile'
        ]
        
        available_features = [f for f in engineered_features if f in df.columns]
        
        if not available_features:
            return pd.DataFrame()
        
        summary = df[available_features].describe().T
        summary['missing_pct'] = (df[available_features].isnull().sum() / len(df) * 100)
        
        return summary
