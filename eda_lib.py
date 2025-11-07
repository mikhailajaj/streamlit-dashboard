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