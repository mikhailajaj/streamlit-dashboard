"""
Configuration settings for ML models
Centralized parameters and feature definitions
"""

# Model Hyperparameters
MODEL_CONFIG = {
    'forecaster': {
        'prophet': {
            'yearly_seasonality': True,
            'weekly_seasonality': True,
            'daily_seasonality': False,
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10.0,
            'interval_width': 0.8
        },
        'arima': {
            'order': (1, 1, 1),
            'seasonal_order': (1, 1, 1, 7)
        }
    },
    
    'anomaly_detector': {
        'contamination': 0.1,
        'n_estimators': 100,
        'max_samples': 'auto',
        'random_state': 42
    },
    
    'clusterer': {
        'n_clusters_range': (2, 10),
        'random_state': 42,
        'init': 'k-means++',
        'n_init': 10,
        'max_iter': 300
    },
    
    'optimizer': {
        'random_forest': {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }
    }
}

# Feature Engineering Configuration
FEATURE_CONFIG = {
    'ec2_features': [
        'CPUUtilization',
        'MemoryUtilization', 
        'CostPerHourUSD',
        'Region',
        'InstanceType',
        'State'
    ],
    
    's3_features': [
        'TotalSizeGB',
        'MonthlyCostUSD',
        'ObjectCount',
        'Region',
        'StorageClass',
        'Encryption'
    ],
    
    'derived_features': {
        'ec2': [
            'cost_efficiency',  # CPU / Cost
            'memory_efficiency',  # Memory / Cost
            'total_efficiency',  # (CPU + Memory) / Cost
            'daily_cost',  # CostPerHour * 24
            'monthly_cost'  # CostPerHour * 24 * 30
        ],
        's3': [
            'cost_per_gb',  # MonthlyCost / TotalSize
            'cost_per_object',  # MonthlyCost / ObjectCount
            'daily_cost',  # MonthlyCost / 30
            'storage_density'  # ObjectCount / TotalSize
        ]
    }
}

# Business Rules for Optimization
OPTIMIZATION_RULES = {
    'ec2': {
        'low_utilization_threshold': 20,  # CPU %
        'high_cost_threshold': 1.0,  # USD per hour
        'optimization_confidence_threshold': 0.7,
        'savings_estimates': {
            'downsize': 0.3,  # 30% savings
            'terminate': 0.9,  # 90% savings
            'spot_instance': 0.6  # 60% savings
        }
    },
    
    's3': {
        'high_cost_threshold': 100,  # USD per month
        'lifecycle_candidate_storage': 'STANDARD',
        'optimization_confidence_threshold': 0.7,
        'savings_estimates': {
            'lifecycle_ia': 0.4,  # 40% savings to IA
            'lifecycle_glacier': 0.7,  # 70% savings to Glacier
            'intelligent_tiering': 0.2  # 20% savings
        }
    }
}

# Anomaly Detection Thresholds
ANOMALY_CONFIG = {
    'cost_spike_multiplier': 2.0,  # Cost is 2x normal
    'utilization_anomaly_threshold': 0.1,  # Bottom 10% or top 10%
    'regional_cost_variance_threshold': 0.3,  # 30% variance from mean
    'seasonal_adjustment': True
}

# Clustering Configuration
CLUSTERING_CONFIG = {
    'feature_weights': {
        'cost': 0.4,
        'utilization': 0.3,
        'efficiency': 0.3
    },
    'cluster_labels': {
        0: 'High Cost, Low Efficiency',
        1: 'Moderate Cost, Good Efficiency', 
        2: 'Low Cost, High Efficiency',
        3: 'High Cost, High Efficiency',
        4: 'Variable Performance'
    }
}

# Forecasting Configuration
FORECAST_CONFIG = {
    'default_periods': 30,  # days
    'confidence_intervals': [0.8, 0.95],
    'trend_analysis_window': 90,  # days
    'seasonality_components': ['yearly', 'weekly'],
    'holiday_effects': False  # Can be enhanced with AWS-specific events
}

# Performance Metrics Configuration
METRICS_CONFIG = {
    'forecast_accuracy_threshold': 0.15,  # 15% MAPE
    'anomaly_precision_threshold': 0.8,
    'clustering_silhouette_threshold': 0.5,
    'optimization_recall_threshold': 0.7
}

# Dashboard Display Configuration
DISPLAY_CONFIG = {
    'max_recommendations': 10,
    'chart_height': 400,
    'color_schemes': {
        'cost_forecast': ['#1f77b4', '#ff7f0e', '#2ca02c'],
        'anomalies': ['#d62728', '#2ca02c'],
        'clusters': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
        'optimization': ['#ff7f0e', '#2ca02c', '#d62728']
    },
    'chart_templates': {
        'forecast': 'plotly_white',
        'anomaly': 'plotly_dark',
        'cluster': 'plotly_white',
        'optimization': 'plotly_white'
    }
}

# Alert Configuration
ALERT_CONFIG = {
    'cost_spike_alert': True,
    'anomaly_alert': True,
    'optimization_opportunity_alert': True,
    'forecast_trend_alert': True,
    'alert_thresholds': {
        'cost_increase': 0.2,  # 20% increase
        'anomaly_score': -0.5,
        'optimization_savings': 50  # USD
    }
}

# Data Validation Rules
VALIDATION_CONFIG = {
    'required_columns': {
        'ec2': ['CPUUtilization', 'CostPerHourUSD', 'Region'],
        's3': ['TotalSizeGB', 'MonthlyCostUSD', 'Region']
    },
    'data_quality_checks': {
        'missing_data_threshold': 0.1,  # 10% missing allowed
        'outlier_detection': True,
        'data_freshness_hours': 24
    }
}

def get_model_config(model_type):
    """Get configuration for specific model type"""
    return MODEL_CONFIG.get(model_type, {})

def get_feature_config(resource_type):
    """Get feature configuration for EC2 or S3"""
    return FEATURE_CONFIG.get(f'{resource_type}_features', [])

def get_optimization_rules(resource_type):
    """Get optimization rules for specific resource type"""
    return OPTIMIZATION_RULES.get(resource_type, {})

def validate_config():
    """Validate configuration settings"""
    required_sections = ['forecaster', 'anomaly_detector', 'clusterer', 'optimizer']
    
    for section in required_sections:
        if section not in MODEL_CONFIG:
            raise ValueError(f"Missing configuration section: {section}")
    
    return True