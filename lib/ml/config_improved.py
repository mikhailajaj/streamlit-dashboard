"""
IMPROVED Configuration settings for ML models
Enhanced with scenario modeling, validation, and data quality settings
"""

# Model Hyperparameters (IMPROVED)
ML_CONFIG = {
    'scenario_modeling': {
        'scenarios': {
            'baseline': {
                'name': 'Baseline (No Optimization)',
                'growth_rate_annual': 0.02,
                'optimization_percentage': 0.0,
                'description': 'Current trajectory with minimal growth'
            },
            'conservative': {
                'name': 'Conservative Growth',
                'growth_rate_annual': 0.05,
                'optimization_percentage': 0.10,
                'description': '5% annual growth, 10% optimization achieved'
            },
            'aggressive_growth': {
                'name': 'Aggressive Growth',
                'growth_rate_annual': 0.15,
                'optimization_percentage': 0.25,
                'description': '15% annual growth, 25% optimization achieved'
            },
            'optimized': {
                'name': 'Fully Optimized',
                'growth_rate_annual': 0.05,
                'optimization_percentage': 0.40,
                'description': '5% growth with aggressive 40% optimization'
            }
        },
        'projection_months': 12,
        'confidence_level': 0.95
    },
    
    'anomaly_detection': {
        'contamination_range': [0.05, 0.10, 0.15],  # Test multiple values
        'n_estimators': 100,
        'max_samples': 'auto',
        'random_state': 42,
        'severity_thresholds': {
            'low': 30,
            'medium': 60,
            'high': 80,
            'critical': 90
        },
        'anomaly_types': [
            'cost_spike',
            'idle_waste',
            'performance_anomaly',
            'efficiency_issue'
        ],
        'local_detection': True  # Enable per-instance-type detection
    },
    
    'clustering': {
        'k_range': (2, 10),  # Auto-select best K in this range
        'min_cluster_size': 5,
        'random_state': 42,
        'init': 'k-means++',
        'n_init': 10,
        'max_iter': 300,
        'silhouette_threshold': 0.5,  # Minimum acceptable quality
        'quality_ratings': {
            0.7: 'Excellent',
            0.5: 'Good',
            0.3: 'Fair',
            0.0: 'Poor'
        }
    },
    
    'optimization': {
        'approach': 'unsupervised',  # Changed from supervised
        'multi_factor_analysis': True,
        'factors': [
            'cpu_utilization',
            'memory_utilization',
            'cost_level',
            'state',
            'waste_score',
            'efficiency_score'
        ],
        'thresholds': {
            'idle_cpu': 5,
            'underutilized_cpu': 25,
            'underutilized_memory': 25,
            'high_cost_percentile': 0.75
        },
        'savings_estimates': {
            'terminate_idle': 0.95,
            'terminate_stopped': 0.85,
            'downsize': 0.40,
            'right_size': 0.25,
            'reserved_instances': 0.30,
            'spot_instances': 0.60,
            's3_lifecycle_ia': 0.40,
            's3_lifecycle_glacier': 0.70,
            's3_intelligent_tiering': 0.25
        },
        'confidence_levels': {
            'high': 80,
            'medium': 60,
            'low': 50
        }
    },
    
    'validation': {
        'test_size': 0.2,
        'cv_folds': 5,
        'random_state': 42,
        'stratify': True,
        'metrics': {
            'regression': ['mse', 'rmse', 'mae', 'mape', 'r2'],
            'classification': ['accuracy', 'precision', 'recall', 'f1'],
            'clustering': ['silhouette', 'calinski_harabasz', 'davies_bouldin']
        },
        'baseline_strategies': ['mean', 'median'],
        'performance_thresholds': {
            'r2_min': 0.5,
            'mape_max': 20.0,  # 20% max error
            'silhouette_min': 0.5,
            'accuracy_min': 0.7
        }
    },
    
    'data_quality': {
        'missing_data_threshold': 0.4,  # 40% missing is problematic
        'quality_score_min': 70,  # Minimum acceptable quality (0-100)
        'cleaning_strategies': ['auto', 'conservative', 'aggressive'],
        'imputation': {
            'numeric_strategy': 'median',
            'categorical_strategy': 'mode',
            'drop_column_threshold': 0.9  # Drop if >90% missing
        },
        'validation_checks': [
            'negative_costs',
            'invalid_utilization',
            'invalid_regions',
            'future_dates',
            'duplicates'
        ]
    }
}

# Feature Engineering Configuration (ENHANCED)
FEATURE_CONFIG = {
    'ec2_cost_efficiency_features': [
        'cost_per_cpu_utilized',
        'cost_per_memory_utilized',
        'idle_cost',
        'waste_score',
        'utilization_balance',
        'efficiency_score'
    ],
    
    'ec2_context_features': [
        'instance_family',
        'instance_size',
        'is_prod',
        'owner',
        'environment',
        'is_stopped'
    ],
    
    'ec2_comparative_features': [
        'peer_mean_cpu',
        'peer_mean_cost',
        'cpu_vs_peers',
        'cost_vs_peers',
        'cpu_zscore',
        'cost_zscore',
        'cost_percentile',
        'cpu_percentile'
    ],
    
    's3_cost_efficiency_features': [
        'cost_per_gb',
        'cost_per_object',
        'storage_density'
    ],
    
    's3_context_features': [
        'is_standard_storage',
        'has_encryption',
        'bucket_purpose'
    ],
    
    's3_optimization_features': [
        'lifecycle_candidate',
        'high_cost_per_gb'
    ],
    
    'feature_importance_by_use_case': {
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
        ]
    }
}

# Business Rules for Optimization (REFINED)
OPTIMIZATION_RULES = {
    'ec2': {
        'idle_threshold': 5,  # CPU %
        'underutilized_threshold': 25,  # CPU %
        'high_cost_percentile': 0.75,
        'waste_score_threshold': 70,
        'priority_rules': {
            'critical': {
                'conditions': ['optimization_score > 80', 'potential_savings > 50'],
                'description': 'Immediate action required'
            },
            'high': {
                'conditions': ['optimization_score > 70', 'potential_savings > 20'],
                'description': 'Address within this week'
            },
            'medium': {
                'conditions': ['optimization_score > 50', 'potential_savings > 10'],
                'description': 'Address within this month'
            },
            'low': {
                'conditions': ['optimization_score > 30'],
                'description': 'Monitor and review'
            }
        }
    },
    
    's3': {
        'high_cost_threshold': 10,  # USD per month
        'lifecycle_candidate_storage': 'STANDARD',
        'cost_per_gb_threshold': 0.05,
        'priority_rules': {
            'high': {
                'conditions': ['cost > 50', 'storage_class == STANDARD'],
                'description': 'Significant savings opportunity'
            },
            'medium': {
                'conditions': ['cost > 10', 'storage_class == STANDARD'],
                'description': 'Good optimization candidate'
            }
        }
    }
}

# Anomaly Detection Thresholds (CONTEXT-AWARE)
ANOMALY_CONFIG = {
    'global_detection': True,
    'local_detection': True,  # Per instance type
    'severity_calculation': 'normalized',  # 0-100 scale
    'type_classification': True,
    'classification_rules': {
        'idle_waste': {
            'conditions': ['cpu < 10', 'cost > 0'],
            'severity_boost': 1.5
        },
        'cost_spike': {
            'conditions': ['cost > p75', 'cpu > 25'],
            'severity_boost': 1.3
        },
        'efficiency_issue': {
            'conditions': ['efficiency_score < 30'],
            'severity_boost': 1.2
        }
    }
}

# Clustering Configuration (AUTO-TUNED)
CLUSTERING_CONFIG = {
    'auto_k_selection': True,
    'k_selection_methods': ['elbow', 'silhouette'],
    'feature_weights': {
        'cost': 0.4,
        'utilization': 0.3,
        'efficiency': 0.3
    },
    'cluster_naming': 'automatic',  # Based on characteristics
    'cluster_interpretation': True,
    'quality_validation': True
}

# Scenario Modeling Configuration (NEW)
SCENARIO_CONFIG = {
    'projection_periods': [3, 6, 12, 24],  # Months
    'growth_assumptions': {
        'baseline': 0.02,
        'conservative': 0.05,
        'moderate': 0.10,
        'aggressive': 0.15
    },
    'optimization_assumptions': {
        'none': 0.0,
        'light': 0.10,
        'moderate': 0.25,
        'aggressive': 0.40
    },
    'visualization': {
        'show_confidence_bands': True,
        'compare_to_baseline': True,
        'show_cumulative': True
    }
}

# Performance Metrics Configuration (ENHANCED)
METRICS_CONFIG = {
    'validation_required': True,
    'baseline_comparison': True,
    'cross_validation': True,
    'confidence_intervals': True,
    'thresholds': {
        'scenario_modeling': {
            'min_data_points': 10,
            'max_projection_error': 0.20  # 20%
        },
        'anomaly_detection': {
            'min_silhouette': 0.3,
            'max_false_positive_rate': 0.20
        },
        'clustering': {
            'min_silhouette': 0.5,
            'min_cluster_size': 5
        },
        'optimization': {
            'min_confidence': 0.5,
            'min_savings_threshold': 5  # USD
        }
    }
}

# Dashboard Display Configuration (IMPROVED)
DISPLAY_CONFIG = {
    'max_recommendations': 10,
    'chart_height': 400,
    'show_confidence_scores': True,
    'show_validation_metrics': True,
    'color_schemes': {
        'scenarios': ['#d62728', '#ff7f0e', '#9467bd', '#2ca02c'],
        'anomalies': {
            'normal': '#2ca02c',
            'cost_spike': '#d62728',
            'idle_waste': '#ff7f0e',
            'efficiency_issue': '#9467bd',
            'performance_anomaly': '#8c564b'
        },
        'clusters': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
        'priorities': {
            'CRITICAL': '#d62728',
            'HIGH': '#ff7f0e',
            'MEDIUM': '#ffbb00',
            'LOW': '#2ca02c'
        }
    }
}

# Data Validation Rules (COMPREHENSIVE)
VALIDATION_CONFIG = {
    'required_columns': {
        'ec2': ['ResourceId', 'CostUSD', 'Region'],
        's3': ['BucketName', 'MonthlyCostUSD', 'Region']
    },
    'data_quality_checks': {
        'completeness_check': True,
        'validity_check': True,
        'consistency_check': True,
        'timeliness_check': False,  # No historical data
        'uniqueness_check': True
    },
    'quality_thresholds': {
        'min_completeness': 0.6,  # 60% complete data
        'max_duplicates': 0.05,  # 5% duplicates allowed
        'min_validity': 0.95  # 95% valid values
    }
}

def get_ml_config(component: str = None):
    """Get ML configuration."""
    if component:
        return ML_CONFIG.get(component, {})
    return ML_CONFIG

def get_feature_config(feature_type: str = None):
    """Get feature engineering configuration."""
    if feature_type:
        return FEATURE_CONFIG.get(feature_type, [])
    return FEATURE_CONFIG

def get_optimization_rules(resource_type: str):
    """Get optimization rules for specific resource type."""
    return OPTIMIZATION_RULES.get(resource_type, {})

def validate_config():
    """Validate configuration settings."""
    required_sections = [
        'scenario_modeling', 'anomaly_detection', 
        'clustering', 'optimization', 'validation', 'data_quality'
    ]
    
    for section in required_sections:
        if section not in ML_CONFIG:
            raise ValueError(f"Missing configuration section: {section}")
    
    return True

# Export all configs
__all__ = [
    'ML_CONFIG',
    'FEATURE_CONFIG',
    'OPTIMIZATION_RULES',
    'ANOMALY_CONFIG',
    'CLUSTERING_CONFIG',
    'SCENARIO_CONFIG',
    'METRICS_CONFIG',
    'DISPLAY_CONFIG',
    'VALIDATION_CONFIG',
    'get_ml_config',
    'get_feature_config',
    'get_optimization_rules',
    'validate_config'
]
