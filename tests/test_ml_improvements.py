"""
Test Suite for ML Improvements
Validates all critical fixes and enhancements
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lib.ml.scenario_modeler import AWSCostScenarioModeler
from lib.ml.feature_engineering import AWSFeatureEngineer
from lib.ml.validation import ModelValidator
from lib.ml.data_quality import DataQualityAnalyzer


class TestScenarioModeling:
    """Test scenario modeling (replacement for time series forecasting)"""
    
    @pytest.fixture
    def sample_ec2_data(self):
        """Create sample EC2 data"""
        np.random.seed(42)
        return pd.DataFrame({
            'ResourceId': [f'i-{i}' for i in range(100)],
            'CostUSD': np.random.uniform(0.1, 2.0, 100),
            'CPUUtilization': np.random.uniform(5, 95, 100),
            'MemoryUtilization': np.random.uniform(10, 90, 100),
            'Region': np.random.choice(['us-east-1', 'us-west-2', 'eu-west-1'], 100),
            'InstanceType': np.random.choice(['c5.xlarge', 'm5.large', 'r5.large'], 100),
            'State': np.random.choice(['running', 'stopped'], 100, p=[0.9, 0.1])
        })
    
    @pytest.fixture
    def sample_s3_data(self):
        """Create sample S3 data"""
        np.random.seed(42)
        return pd.DataFrame({
            'BucketName': [f'bucket-{i}' for i in range(50)],
            'MonthlyCostUSD': np.random.uniform(5, 200, 50),
            'TotalSizeGB': np.random.uniform(10, 5000, 50),
            'ObjectCount': np.random.randint(100, 100000, 50),
            'Region': np.random.choice(['us-east-1', 'us-west-2', 'eu-west-1'], 50),
            'StorageClass': np.random.choice(['STANDARD', 'INTELLIGENT_TIERING'], 50, p=[0.7, 0.3])
        })
    
    def test_scenario_modeler_initialization(self):
        """Test that scenario modeler initializes correctly"""
        modeler = AWSCostScenarioModeler()
        assert modeler is not None
        assert not modeler.is_fitted
    
    def test_baseline_analysis(self, sample_ec2_data, sample_s3_data):
        """Test baseline cost analysis"""
        modeler = AWSCostScenarioModeler()
        baseline = modeler.analyze_baseline(sample_ec2_data, sample_s3_data)
        
        assert 'total_monthly' in baseline
        assert 'ec2_monthly' in baseline
        assert 's3_monthly' in baseline
        assert baseline['total_monthly'] > 0
        assert baseline['ec2_count'] == 100
        assert baseline['s3_count'] == 50
    
    def test_optimization_opportunities(self, sample_ec2_data, sample_s3_data):
        """Test identification of optimization opportunities"""
        modeler = AWSCostScenarioModeler()
        opportunities = modeler.identify_optimization_opportunities(sample_ec2_data, sample_s3_data)
        
        assert 'idle_instances' in opportunities
        assert 'underutilized_instances' in opportunities
        assert 'total_potential_monthly_savings' in opportunities
        assert opportunities['total_potential_monthly_savings'] >= 0
    
    def test_scenario_generation(self, sample_ec2_data, sample_s3_data):
        """Test scenario generation"""
        modeler = AWSCostScenarioModeler()
        modeler.analyze_baseline(sample_ec2_data, sample_s3_data)
        modeler.identify_optimization_opportunities(sample_ec2_data, sample_s3_data)
        scenarios = modeler.generate_scenarios(months=12)
        
        assert 'baseline' in scenarios
        assert 'conservative' in scenarios
        assert 'optimized' in scenarios
        assert modeler.is_fitted
        
        # Check that each scenario has required columns
        for scenario_name, scenario_data in scenarios.items():
            assert 'month' in scenario_data.columns
            assert 'monthly_cost' in scenario_data.columns
            assert 'cumulative_cost' in scenario_data.columns
            assert len(scenario_data) == 12
    
    def test_scenario_comparison(self, sample_ec2_data, sample_s3_data):
        """Test scenario comparison calculations"""
        modeler = AWSCostScenarioModeler()
        modeler.analyze_baseline(sample_ec2_data, sample_s3_data)
        modeler.identify_optimization_opportunities(sample_ec2_data, sample_s3_data)
        modeler.generate_scenarios(months=12)
        
        comparison = modeler.calculate_scenario_comparisons()
        
        assert len(comparison) >= 2  # At least baseline and one other
        assert 'savings_vs_baseline_12m' in comparison.columns
    
    def test_recommendations_generation(self, sample_ec2_data, sample_s3_data):
        """Test recommendation generation"""
        modeler = AWSCostScenarioModeler()
        modeler.identify_optimization_opportunities(sample_ec2_data, sample_s3_data)
        recommendations = modeler.generate_recommendations()
        
        assert isinstance(recommendations, list)
        if len(recommendations) > 0:
            rec = recommendations[0]
            assert 'priority' in rec
            assert 'category' in rec
            assert 'action' in rec
            assert 'potential_monthly_savings' in rec


class TestFeatureEngineering:
    """Test feature engineering capabilities"""
    
    @pytest.fixture
    def sample_ec2_data(self):
        """Create sample EC2 data with various patterns"""
        np.random.seed(42)
        return pd.DataFrame({
            'ResourceId': [f'i-{i}' for i in range(50)],
            'CostUSD': np.random.uniform(0.1, 2.0, 50),
            'CPUUtilization': np.random.uniform(5, 95, 50),
            'MemoryUtilization': np.random.uniform(10, 90, 50),
            'InstanceType': np.random.choice(['c5.xlarge', 'm5.large', 'r5.large', 't3.small'], 50),
            'Region': np.random.choice(['us-east-1', 'us-west-2'], 50),
            'State': np.random.choice(['running', 'stopped'], 50),
            'Tags': [f'Owner=Alice,Environment={"Prod" if i % 3 == 0 else "Dev"}' for i in range(50)],
            'NetworkIn_Bps': np.random.randint(10000, 1000000, 50),
            'NetworkOut_Bps': np.random.randint(10000, 1000000, 50)
        })
    
    def test_feature_engineer_initialization(self):
        """Test feature engineer initialization"""
        engineer = AWSFeatureEngineer()
        assert engineer is not None
    
    def test_cost_efficiency_features(self, sample_ec2_data):
        """Test cost efficiency feature creation"""
        engineer = AWSFeatureEngineer()
        enriched = engineer.engineer_ec2_features(sample_ec2_data)
        
        # Check for cost efficiency features
        assert 'cost_per_cpu_utilized' in enriched.columns
        assert 'idle_cost' in enriched.columns
        assert 'waste_score' in enriched.columns
        assert 'efficiency_score' in enriched.columns
    
    def test_context_features(self, sample_ec2_data):
        """Test context feature extraction"""
        engineer = AWSFeatureEngineer()
        enriched = engineer.engineer_ec2_features(sample_ec2_data)
        
        # Check for context features
        assert 'instance_family' in enriched.columns
        assert 'instance_size' in enriched.columns
        assert 'is_prod' in enriched.columns
        assert 'owner' in enriched.columns
        assert 'environment' in enriched.columns
    
    def test_comparative_features(self, sample_ec2_data):
        """Test peer comparison features"""
        engineer = AWSFeatureEngineer()
        enriched = engineer.engineer_ec2_features(sample_ec2_data)
        
        # Check for comparative features
        assert 'peer_mean_cpu' in enriched.columns
        assert 'cpu_vs_peers' in enriched.columns
        assert 'cost_percentile' in enriched.columns
    
    def test_feature_validity(self, sample_ec2_data):
        """Test that engineered features have valid values"""
        engineer = AWSFeatureEngineer()
        enriched = engineer.engineer_ec2_features(sample_ec2_data)
        
        # Waste score should be 0-100
        assert enriched['waste_score'].min() >= 0
        assert enriched['waste_score'].max() <= 100
        
        # Efficiency score should be 0-100
        assert enriched['efficiency_score'].min() >= 0
        assert enriched['efficiency_score'].max() <= 100


class TestModelValidation:
    """Test model validation framework"""
    
    def test_validator_initialization(self):
        """Test validator initialization"""
        validator = ModelValidator()
        assert validator.test_size == 0.2
        assert validator.cv_folds == 5
    
    def test_regression_metrics(self):
        """Test regression metrics calculation"""
        validator = ModelValidator()
        
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        
        metrics = validator.calculate_regression_metrics(y_true, y_pred, 'test_model')
        
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2_score' in metrics
        assert 'mape' in metrics
        assert metrics['rmse'] > 0
        assert metrics['r2_score'] <= 1
    
    def test_classification_metrics(self):
        """Test classification metrics calculation"""
        validator = ModelValidator()
        
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 0, 1, 0, 0, 1, 1, 1])
        
        metrics = validator.calculate_classification_metrics(y_true, y_pred, model_name='test_classifier')
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_clustering_metrics(self):
        """Test clustering quality metrics"""
        validator = ModelValidator()
        
        # Create simple clustered data
        X = np.vstack([
            np.random.randn(20, 2) + [0, 0],
            np.random.randn(20, 2) + [5, 5]
        ])
        labels = np.array([0] * 20 + [1] * 20)
        
        metrics = validator.calculate_clustering_metrics(X, labels, 'test_clustering')
        
        assert 'silhouette_score' in metrics
        assert 'n_clusters' in metrics
        assert metrics['n_clusters'] == 2
        assert -1 <= metrics['silhouette_score'] <= 1
    
    def test_baseline_comparison(self):
        """Test baseline comparison"""
        validator = ModelValidator()
        
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        y_pred = np.array([1.2, 2.1, 3.1, 3.9, 5.2, 5.8, 7.1, 8.2, 8.9, 10.1])
        
        comparison = validator.baseline_comparison(y_true, y_pred, baseline_strategy='mean')
        
        assert 'model_mae' in comparison
        assert 'baseline_mae' in comparison
        assert 'mae_improvement_pct' in comparison
        assert 'is_better_than_baseline' in comparison


class TestDataQuality:
    """Test data quality analysis"""
    
    @pytest.fixture
    def data_with_nulls(self):
        """Create data with missing values"""
        df = pd.DataFrame({
            'col1': [1, 2, 3, np.nan, 5],
            'col2': [np.nan, np.nan, np.nan, np.nan, np.nan],
            'col3': [1, 2, 3, 4, 5]
        })
        return df
    
    @pytest.fixture
    def data_pagination_issue(self):
        """Create data simulating pagination issue"""
        # 80 valid rows followed by 20 NULL rows
        valid_data = pd.DataFrame({
            'ResourceId': [f'i-{i}' for i in range(80)],
            'CostUSD': np.random.uniform(0.1, 2.0, 80),
            'CPUUtilization': np.random.uniform(5, 95, 80)
        })
        
        null_data = pd.DataFrame({
            'ResourceId': [np.nan] * 20,
            'CostUSD': [np.nan] * 20,
            'CPUUtilization': [np.nan] * 20
        })
        
        return pd.concat([valid_data, null_data], ignore_index=True)
    
    def test_analyzer_initialization(self):
        """Test data quality analyzer initialization"""
        analyzer = DataQualityAnalyzer()
        assert analyzer is not None
    
    def test_missing_data_analysis(self, data_with_nulls):
        """Test missing data analysis"""
        analyzer = DataQualityAnalyzer()
        report = analyzer.analyze_missing_data(data_with_nulls, 'test_dataset')
        
        assert 'total_rows' in report
        assert 'total_columns' in report
        assert 'completely_null_rows' in report
        assert 'data_quality_score' in report
        assert report['total_rows'] == 5
        assert report['total_columns'] == 3
    
    def test_pagination_detection(self, data_pagination_issue):
        """Test detection of pagination issue"""
        analyzer = DataQualityAnalyzer()
        report = analyzer.analyze_missing_data(data_pagination_issue, 'ec2_data')
        
        assert report['is_pagination_issue'] == True
        assert report['completely_null_rows'] == 20
    
    def test_cleaning_recommendations(self, data_pagination_issue):
        """Test cleaning strategy recommendations"""
        analyzer = DataQualityAnalyzer()
        report = analyzer.analyze_missing_data(data_pagination_issue, 'ec2_data')
        recommendations = analyzer.recommend_cleaning_strategy(report)
        
        assert len(recommendations) > 0
        assert recommendations[0]['priority'] == 'HIGH'
        assert 'Pagination' in recommendations[0]['issue']
    
    def test_data_cleaning(self, data_pagination_issue):
        """Test automatic data cleaning"""
        analyzer = DataQualityAnalyzer()
        cleaned_df, report = analyzer.clean_dataset(data_pagination_issue, strategy='auto')
        
        assert len(cleaned_df) == 80  # Should drop 20 NULL rows
        assert 'actions_taken' in report
        assert report['rows_removed'] == 20
    
    def test_quality_score_calculation(self):
        """Test quality score calculation"""
        analyzer = DataQualityAnalyzer()
        
        # Perfect data
        perfect_data = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': [10, 20, 30, 40, 50]
        })
        
        report = analyzer.analyze_missing_data(perfect_data, 'perfect')
        assert report['data_quality_score'] == 100.0


# Run tests if executed directly
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
