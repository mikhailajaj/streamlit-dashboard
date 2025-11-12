"""
Reusable library for AWS EC2 and S3 EDA analysis.
Contains functions for data loading, cleaning, analysis, and visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_datasets(ec2_path='data/aws_resources_compute.csv', 
                 s3_path='data/aws_resources_S3.csv'):
    """
    Load EC2 and S3 datasets from CSV files.
    
    Args:
        ec2_path (str): Path to EC2 dataset
        s3_path (str): Path to S3 dataset
        
    Returns:
        tuple: (ec2_df, s3_df) DataFrames
    """
    # Load EC2 data
    ec2_df = pd.read_csv(ec2_path)
    
    # Load S3 data
    s3_df = pd.read_csv(s3_path)
    
    return ec2_df, s3_df

def clean_ec2_data(df):
    """
    Clean and prepare EC2 data for analysis.
    
    Args:
        df (pd.DataFrame): Raw EC2 DataFrame
        
    Returns:
        pd.DataFrame: Cleaned EC2 DataFrame
    """
    # Drop rows with missing values
    df_clean = df.dropna().copy()
    
    # Convert CreationDate to datetime
    df_clean['CreationDate'] = pd.to_datetime(df_clean['CreationDate'])
    
    # Rename columns to match expected schema
    column_mapping = {
        'ResourceId': 'InstanceId',
        'CostUSD': 'CostPerHourUSD',
        'CreationDate': 'LaunchTime'
    }
    df_clean.rename(columns=column_mapping, inplace=True)
    
    return df_clean

def clean_s3_data(df):
    """
    Clean and prepare S3 data for analysis.
    
    Args:
        df (pd.DataFrame): Raw S3 DataFrame
        
    Returns:
        pd.DataFrame: Cleaned S3 DataFrame
    """
    df_clean = df.copy()
    
    # Convert CreationDate to datetime
    df_clean['CreationDate'] = pd.to_datetime(df_clean['CreationDate'])
    
    # Fill missing encryption values
    df_clean['Encryption'] = df_clean['Encryption'].fillna('None')
    
    # Rename columns to match expected schema
    column_mapping = {
        'CostUSD': 'MonthlyCostUSD',
        'VersionEnabled': 'VersioningEnabled'
    }
    df_clean.rename(columns=column_mapping, inplace=True)
    
    return df_clean

def detect_outliers(df, column, method='iqr'):
    """
    Detect outliers using IQR method.
    
    Args:
        df (pd.DataFrame): DataFrame
        column (str): Column name to check for outliers
        method (str): Method to use ('iqr' or 'zscore')
        
    Returns:
        pd.Series: Boolean series indicating outliers
    """
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (df[column] < lower_bound) | (df[column] > upper_bound)
    elif method == 'zscore':
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        return z_scores > 3

def create_ec2_visualizations(df, save_plots=True):
    """
    Create EC2 visualizations as required.
    
    Args:
        df (pd.DataFrame): EC2 DataFrame
        save_plots (bool): Whether to save plots to files
    """
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Histogram of CPU utilization
    axes[0, 0].hist(df['CPUUtilization'], bins=20, color='skyblue', alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('EC2 CPU Utilization Distribution', fontweight='bold')
    axes[0, 0].set_xlabel('CPU Utilization (%)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. CPU vs Cost scatter plot
    scatter = axes[0, 1].scatter(df['CPUUtilization'], df['CostPerHourUSD'], 
                                alpha=0.6, c='coral', s=50)
    axes[0, 1].set_title('EC2 CPU Utilization vs Cost', fontweight='bold')
    axes[0, 1].set_xlabel('CPU Utilization (%)')
    axes[0, 1].set_ylabel('Cost per Hour (USD)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Cost by Region
    region_costs = df.groupby('Region')['CostPerHourUSD'].mean().sort_values(ascending=False)
    axes[1, 0].bar(region_costs.index, region_costs.values, color='lightgreen', alpha=0.7)
    axes[1, 0].set_title('Average EC2 Cost per Hour by Region', fontweight='bold')
    axes[1, 0].set_xlabel('Region')
    axes[1, 0].set_ylabel('Average Cost per Hour (USD)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Instance State Distribution
    state_counts = df['State'].value_counts()
    axes[1, 1].pie(state_counts.values, labels=state_counts.index, autopct='%1.1f%%', 
                   colors=['lightcoral', 'lightskyblue', 'lightgreen'])
    axes[1, 1].set_title('EC2 Instance State Distribution', fontweight='bold')
    
    plt.tight_layout()
    if save_plots:
        plt.savefig('ec2_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_s3_visualizations(df, save_plots=True):
    """
    Create S3 visualizations as required.
    
    Args:
        df (pd.DataFrame): S3 DataFrame
        save_plots (bool): Whether to save plots to files
    """
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Total storage by region (bar chart)
    region_storage = df.groupby('Region')['TotalSizeGB'].sum().sort_values(ascending=False)
    axes[0, 0].bar(region_storage.index, region_storage.values, color='lightblue', alpha=0.7)
    axes[0, 0].set_title('Total S3 Storage by Region', fontweight='bold')
    axes[0, 0].set_xlabel('Region')
    axes[0, 0].set_ylabel('Total Storage (GB)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Cost vs Storage scatter plot
    scatter = axes[0, 1].scatter(df['TotalSizeGB'], df['MonthlyCostUSD'], 
                                alpha=0.6, c='orange', s=50)
    axes[0, 1].set_title('S3 Storage Size vs Monthly Cost', fontweight='bold')
    axes[0, 1].set_xlabel('Total Size (GB)')
    axes[0, 1].set_ylabel('Monthly Cost (USD)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Storage Class Distribution
    storage_counts = df['StorageClass'].value_counts()
    axes[1, 0].pie(storage_counts.values, labels=storage_counts.index, autopct='%1.1f%%',
                   colors=['lightcoral', 'lightskyblue', 'lightgreen'])
    axes[1, 0].set_title('S3 Storage Class Distribution', fontweight='bold')
    
    # 4. Encryption Status
    encryption_counts = df['Encryption'].value_counts()
    axes[1, 1].bar(encryption_counts.index, encryption_counts.values, 
                   color=['red' if x == 'None' else 'green' for x in encryption_counts.index], 
                   alpha=0.7)
    axes[1, 1].set_title('S3 Encryption Status', fontweight='bold')
    axes[1, 1].set_xlabel('Encryption Type')
    axes[1, 1].set_ylabel('Number of Buckets')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_plots:
        plt.savefig('s3_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def find_top_expensive_ec2(df, n=5):
    """
    Find top N most expensive EC2 instances.
    
    Args:
        df (pd.DataFrame): EC2 DataFrame
        n (int): Number of top instances to return
        
    Returns:
        pd.DataFrame: Top expensive instances
    """
    return df.nlargest(n, 'CostPerHourUSD')[['InstanceId', 'InstanceType', 'Region', 'CostPerHourUSD', 'CPUUtilization']]

def find_largest_s3_buckets(df, n=5):
    """
    Find top N largest S3 buckets.
    
    Args:
        df (pd.DataFrame): S3 DataFrame
        n (int): Number of top buckets to return
        
    Returns:
        pd.DataFrame: Largest buckets
    """
    return df.nlargest(n, 'TotalSizeGB')[['BucketName', 'Region', 'TotalSizeGB', 'MonthlyCostUSD', 'StorageClass']]

def compute_regional_stats(ec2_df, s3_df):
    """
    Compute regional statistics for both EC2 and S3.
    
    Args:
        ec2_df (pd.DataFrame): EC2 DataFrame
        s3_df (pd.DataFrame): S3 DataFrame
        
    Returns:
        tuple: (ec2_regional_stats, s3_regional_stats)
    """
    # EC2 regional stats
    ec2_stats = ec2_df.groupby('Region').agg({
        'CostPerHourUSD': ['mean', 'sum', 'count'],
        'CPUUtilization': 'mean',
        'MemoryUtilization': 'mean'
    }).round(2)
    
    # S3 regional stats
    s3_stats = s3_df.groupby('Region').agg({
        'TotalSizeGB': ['sum', 'mean', 'count'],
        'MonthlyCostUSD': ['sum', 'mean'],
        'ObjectCount': 'sum'
    }).round(2)
    
    return ec2_stats, s3_stats

def generate_optimization_recommendations(ec2_df, s3_df):
    """
    Generate optimization recommendations based on data analysis.
    
    Args:
        ec2_df (pd.DataFrame): EC2 DataFrame
        s3_df (pd.DataFrame): S3 DataFrame
        
    Returns:
        dict: Recommendations for EC2 and S3
    """
    recommendations = {
        'ec2': [],
        's3': []
    }
    
    # EC2 Recommendations
    # 1. Low CPU utilization instances
    low_cpu_instances = ec2_df[ec2_df['CPUUtilization'] < 10]
    if len(low_cpu_instances) > 0:
        recommendations['ec2'].append(
            f"Consider downsizing or terminating {len(low_cpu_instances)} instances with CPU utilization < 10%. "
            f"Potential monthly savings: ${(low_cpu_instances['CostPerHourUSD'].sum() * 24 * 30):.2f}"
        )
    
    # 2. Expensive instances with low utilization
    expensive_low_util = ec2_df[(ec2_df['CostPerHourUSD'] > ec2_df['CostPerHourUSD'].quantile(0.75)) & 
                                (ec2_df['CPUUtilization'] < 25)]
    if len(expensive_low_util) > 0:
        recommendations['ec2'].append(
            f"Review {len(expensive_low_util)} expensive instances with low CPU utilization. "
            f"Consider switching to spot instances or smaller instance types."
        )
    
    # S3 Recommendations
    # 1. Expensive buckets that could benefit from lifecycle policies
    expensive_standard = s3_df[(s3_df['StorageClass'] == 'STANDARD') & 
                              (s3_df['MonthlyCostUSD'] > s3_df['MonthlyCostUSD'].quantile(0.75))]
    if len(expensive_standard) > 0:
        recommendations['s3'].append(
            f"Consider implementing lifecycle policies for {len(expensive_standard)} expensive STANDARD storage buckets. "
            f"Potential savings by moving to IA or Glacier: 30-70% cost reduction."
        )
    
    # 2. Unencrypted buckets
    unencrypted = s3_df[s3_df['Encryption'] == 'None']
    if len(unencrypted) > 0:
        recommendations['s3'].append(
            f"Enable encryption for {len(unencrypted)} unencrypted S3 buckets to improve security compliance."
        )
    
    return recommendations

# ML Preprocessing and Feature Engineering Functions

def prepare_ml_features(ec2_df, s3_df):
    """
    Prepare and engineer features for ML models
    
    Args:
        ec2_df (pd.DataFrame): EC2 DataFrame
        s3_df (pd.DataFrame): S3 DataFrame
        
    Returns:
        tuple: (ec2_features, s3_features) with engineered features
    """
    # EC2 feature engineering
    ec2_features = ec2_df.copy()
    
    # Derived features
    ec2_features['cost_efficiency'] = ec2_features['CPUUtilization'] / (ec2_features['CostPerHourUSD'] + 0.01)
    ec2_features['memory_efficiency'] = ec2_features['MemoryUtilization'] / (ec2_features['CostPerHourUSD'] + 0.01)
    ec2_features['total_efficiency'] = (ec2_features['CPUUtilization'] + ec2_features['MemoryUtilization']) / (ec2_features['CostPerHourUSD'] + 0.01)
    ec2_features['daily_cost'] = ec2_features['CostPerHourUSD'] * 24
    ec2_features['monthly_cost'] = ec2_features['CostPerHourUSD'] * 24 * 30
    
    # Categorical encoding
    ec2_features['region_encoded'] = pd.Categorical(ec2_features['Region']).codes
    ec2_features['instance_type_encoded'] = pd.Categorical(ec2_features['InstanceType']).codes
    ec2_features['state_encoded'] = pd.Categorical(ec2_features['State']).codes
    
    # S3 feature engineering
    s3_features = s3_df.copy()
    
    # Derived features
    s3_features['cost_per_gb'] = s3_features['MonthlyCostUSD'] / (s3_features['TotalSizeGB'] + 0.01)
    s3_features['cost_per_object'] = s3_features['MonthlyCostUSD'] / (s3_features['ObjectCount'] + 0.01)
    s3_features['daily_cost'] = s3_features['MonthlyCostUSD'] / 30
    s3_features['storage_density'] = s3_features['ObjectCount'] / (s3_features['TotalSizeGB'] + 0.01)
    
    # Categorical encoding
    s3_features['region_encoded'] = pd.Categorical(s3_features['Region']).codes
    s3_features['storage_class_encoded'] = pd.Categorical(s3_features['StorageClass']).codes
    s3_features['encryption_encoded'] = pd.Categorical(s3_features['Encryption']).codes
    
    return ec2_features, s3_features

def create_time_features(df, date_column):
    """
    Create time-based features from date column
    
    Args:
        df (pd.DataFrame): DataFrame with date column
        date_column (str): Name of date column
        
    Returns:
        pd.DataFrame: DataFrame with time features
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Extract time components
    df['year'] = df[date_column].dt.year
    df['month'] = df[date_column].dt.month
    df['day'] = df[date_column].dt.day
    df['day_of_week'] = df[date_column].dt.dayofweek
    df['day_of_year'] = df[date_column].dt.dayofyear
    df['week_of_year'] = df[date_column].dt.isocalendar().week
    df['quarter'] = df[date_column].dt.quarter
    
    # Create cyclical features
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    return df

def normalize_features(df, columns, method='standard'):
    """
    Normalize numerical features
    
    Args:
        df (pd.DataFrame): DataFrame
        columns (list): Columns to normalize
        method (str): 'standard', 'minmax', or 'robust'
        
    Returns:
        tuple: (normalized_df, scaler)
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    
    df_normalized = df.copy()
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError("Method must be 'standard', 'minmax', or 'robust'")
    
    df_normalized[columns] = scaler.fit_transform(df[columns])
    
    return df_normalized, scaler

def detect_cost_patterns(ec2_df, s3_df):
    """
    Detect patterns in cost data for better ML model training
    
    Args:
        ec2_df (pd.DataFrame): EC2 DataFrame
        s3_df (pd.DataFrame): S3 DataFrame
        
    Returns:
        dict: Detected patterns and insights
    """
    patterns = {
        'ec2': {},
        's3': {},
        'combined': {}
    }
    
    # EC2 patterns
    patterns['ec2']['cost_distribution'] = {
        'mean': ec2_df['CostPerHourUSD'].mean(),
        'std': ec2_df['CostPerHourUSD'].std(),
        'skewness': ec2_df['CostPerHourUSD'].skew(),
        'kurtosis': ec2_df['CostPerHourUSD'].kurtosis()
    }
    
    patterns['ec2']['utilization_patterns'] = {
        'low_util_instances': len(ec2_df[ec2_df['CPUUtilization'] < 20]),
        'high_util_instances': len(ec2_df[ec2_df['CPUUtilization'] > 80]),
        'avg_cpu': ec2_df['CPUUtilization'].mean(),
        'avg_memory': ec2_df['MemoryUtilization'].mean()
    }
    
    # S3 patterns
    patterns['s3']['cost_distribution'] = {
        'mean': s3_df['MonthlyCostUSD'].mean(),
        'std': s3_df['MonthlyCostUSD'].std(),
        'skewness': s3_df['MonthlyCostUSD'].skew(),
        'kurtosis': s3_df['MonthlyCostUSD'].kurtosis()
    }
    
    patterns['s3']['storage_patterns'] = {
        'large_buckets': len(s3_df[s3_df['TotalSizeGB'] > s3_df['TotalSizeGB'].quantile(0.9)]),
        'expensive_buckets': len(s3_df[s3_df['MonthlyCostUSD'] > s3_df['MonthlyCostUSD'].quantile(0.9)]),
        'avg_size': s3_df['TotalSizeGB'].mean(),
        'avg_objects': s3_df['ObjectCount'].mean()
    }
    
    # Combined patterns
    total_ec2_monthly = ec2_df['CostPerHourUSD'].sum() * 24 * 30
    total_s3_monthly = s3_df['MonthlyCostUSD'].sum()
    
    patterns['combined']['cost_split'] = {
        'ec2_percentage': total_ec2_monthly / (total_ec2_monthly + total_s3_monthly) * 100,
        's3_percentage': total_s3_monthly / (total_ec2_monthly + total_s3_monthly) * 100,
        'total_monthly': total_ec2_monthly + total_s3_monthly
    }
    
    return patterns

def prepare_forecast_data(ec2_df, s3_df, aggregation='daily'):
    """
    Prepare time series data for forecasting models
    
    Args:
        ec2_df (pd.DataFrame): EC2 DataFrame
        s3_df (pd.DataFrame): S3 DataFrame
        aggregation (str): 'daily', 'weekly', or 'monthly'
        
    Returns:
        pd.DataFrame: Time series data ready for forecasting
    """
    # Ensure date columns are datetime
    ec2_df = ec2_df.copy()
    s3_df = s3_df.copy()
    
    ec2_df['date'] = pd.to_datetime(ec2_df['LaunchTime']).dt.date
    s3_df['date'] = pd.to_datetime(s3_df['CreationDate']).dt.date
    
    # Aggregate costs
    if aggregation == 'daily':
        ec2_daily = ec2_df.groupby('date')['CostPerHourUSD'].sum() * 24
        s3_daily = s3_df.groupby('date')['MonthlyCostUSD'].sum() / 30
        freq = 'D'
    elif aggregation == 'weekly':
        ec2_df['week'] = pd.to_datetime(ec2_df['date']).dt.to_period('W')
        s3_df['week'] = pd.to_datetime(s3_df['date']).dt.to_period('W')
        ec2_daily = ec2_df.groupby('week')['CostPerHourUSD'].sum() * 24 * 7
        s3_daily = s3_df.groupby('week')['MonthlyCostUSD'].sum() / 30 * 7
        freq = 'W'
    elif aggregation == 'monthly':
        ec2_df['month'] = pd.to_datetime(ec2_df['date']).dt.to_period('M')
        s3_df['month'] = pd.to_datetime(s3_df['date']).dt.to_period('M')
        ec2_daily = ec2_df.groupby('month')['CostPerHourUSD'].sum() * 24 * 30
        s3_daily = s3_df.groupby('month')['MonthlyCostUSD'].sum()
        freq = 'M'
    
    # Create complete date range
    if aggregation == 'daily':
        date_range = pd.date_range(
            start=min(ec2_daily.index.min(), s3_daily.index.min()),
            end=max(ec2_daily.index.max(), s3_daily.index.max()),
            freq=freq
        )
        date_col = 'ds'
    else:
        date_range = pd.period_range(
            start=min(ec2_daily.index.min(), s3_daily.index.min()),
            end=max(ec2_daily.index.max(), s3_daily.index.max()),
            freq=freq
        )
        date_col = 'ds'
    
    # Combine data
    forecast_data = pd.DataFrame({
        date_col: date_range,
        'ec2_cost': ec2_daily.reindex(date_range, fill_value=0),
        's3_cost': s3_daily.reindex(date_range, fill_value=0)
    })
    
    forecast_data['y'] = forecast_data['ec2_cost'] + forecast_data['s3_cost']
    
    if aggregation != 'daily':
        forecast_data[date_col] = forecast_data[date_col].dt.to_timestamp()
    
    return forecast_data

def calculate_optimization_scores(ec2_df, s3_df):
    """
    Calculate optimization scores for resources
    
    Args:
        ec2_df (pd.DataFrame): EC2 DataFrame
        s3_df (pd.DataFrame): S3 DataFrame
        
    Returns:
        tuple: (ec2_scores, s3_scores)
    """
    # EC2 optimization scores
    ec2_scores = ec2_df.copy()
    
    # Utilization score (lower is worse)
    ec2_scores['utilization_score'] = (ec2_scores['CPUUtilization'] + ec2_scores['MemoryUtilization']) / 200
    
    # Cost efficiency score (higher cost per unit utilization is worse)
    ec2_scores['cost_efficiency_score'] = 1 / (ec2_scores['CostPerHourUSD'] / (ec2_scores['CPUUtilization'] + 1))
    
    # Combined optimization score (0-1, higher means needs more optimization)
    ec2_scores['optimization_score'] = (
        (1 - ec2_scores['utilization_score']) * 0.6 +  # 60% weight on utilization
        ec2_scores['cost_efficiency_score'] * 0.4      # 40% weight on cost efficiency
    )
    
    # S3 optimization scores
    s3_scores = s3_df.copy()
    
    # Cost per GB score (higher cost per GB suggests optimization opportunity)
    s3_scores['cost_per_gb_score'] = s3_scores['MonthlyCostUSD'] / (s3_scores['TotalSizeGB'] + 1)
    s3_scores['cost_per_gb_score'] = s3_scores['cost_per_gb_score'] / s3_scores['cost_per_gb_score'].max()
    
    # Storage class efficiency (STANDARD storage is less efficient for infrequent access)
    s3_scores['storage_class_score'] = (s3_scores['StorageClass'] == 'STANDARD').astype(float)
    
    # Combined optimization score
    s3_scores['optimization_score'] = (
        s3_scores['cost_per_gb_score'] * 0.7 +     # 70% weight on cost efficiency
        s3_scores['storage_class_score'] * 0.3     # 30% weight on storage class
    )
    
    return ec2_scores, s3_scores

def validate_ml_data(ec2_df, s3_df):
    """
    Validate data quality for ML models
    
    Args:
        ec2_df (pd.DataFrame): EC2 DataFrame
        s3_df (pd.DataFrame): S3 DataFrame
        
    Returns:
        dict: Validation results and recommendations
    """
    validation_results = {
        'ec2': {'passed': True, 'issues': []},
        's3': {'passed': True, 'issues': []},
        'overall': {'passed': True, 'recommendations': []}
    }
    
    # EC2 validation
    required_ec2_cols = ['CPUUtilization', 'MemoryUtilization', 'CostPerHourUSD', 'Region']
    for col in required_ec2_cols:
        if col not in ec2_df.columns:
            validation_results['ec2']['issues'].append(f"Missing required column: {col}")
            validation_results['ec2']['passed'] = False
    
    # Check for missing values
    if ec2_df[required_ec2_cols].isnull().sum().sum() > 0:
        validation_results['ec2']['issues'].append("Missing values detected in required columns")
    
    # Check for negative costs
    if (ec2_df['CostPerHourUSD'] < 0).any():
        validation_results['ec2']['issues'].append("Negative costs detected")
    
    # S3 validation
    required_s3_cols = ['TotalSizeGB', 'MonthlyCostUSD', 'ObjectCount', 'Region']
    for col in required_s3_cols:
        if col not in s3_df.columns:
            validation_results['s3']['issues'].append(f"Missing required column: {col}")
            validation_results['s3']['passed'] = False
    
    # Check for missing values
    if s3_df[required_s3_cols].isnull().sum().sum() > 0:
        validation_results['s3']['issues'].append("Missing values detected in required columns")
    
    # Check for negative costs or sizes
    if (s3_df['MonthlyCostUSD'] < 0).any() or (s3_df['TotalSizeGB'] < 0).any():
        validation_results['s3']['issues'].append("Negative costs or storage sizes detected")
    
    # Overall validation
    if not validation_results['ec2']['passed'] or not validation_results['s3']['passed']:
        validation_results['overall']['passed'] = False
    
    # Generate recommendations
    if len(ec2_df) < 10:
        validation_results['overall']['recommendations'].append("Limited EC2 data - ML models may be less accurate")
    
    if len(s3_df) < 5:
        validation_results['overall']['recommendations'].append("Limited S3 data - ML models may be less accurate")
    
    return validation_results