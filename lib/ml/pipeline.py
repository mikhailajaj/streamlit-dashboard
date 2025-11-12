"""
Automated ML Pipeline for AWS Cost Analysis
Handles model training, validation, and real-time predictions with caching
"""

import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import joblib
import os

# Import ML models with error handling
try:
    from lib.ml.models import (
        AWSCostForecaster, 
        AWSAnomalyDetector, 
        AWSResourceClusterer, 
        AWSOptimizationPredictor,
        PROPHET_AVAILABLE,
        ARIMA_AVAILABLE
    )
    ML_MODELS_AVAILABLE = True
except ImportError as e:
    try:
        # Fallback to relative import
        from .models import (
            AWSCostForecaster, 
            AWSAnomalyDetector, 
            AWSResourceClusterer, 
            AWSOptimizationPredictor,
            PROPHET_AVAILABLE,
            ARIMA_AVAILABLE
        )
        ML_MODELS_AVAILABLE = True
    except ImportError as e2:
        ML_MODELS_AVAILABLE = False
        print(f"🔄 ML Pipeline running in fallback mode: {e2}")
    
    # Create dummy classes for fallback
    class AWSCostForecaster:
        def __init__(self, *args, **kwargs):
            pass
        def fit(self, *args, **kwargs):
            return self
        def predict(self, *args, **kwargs):
            return pd.DataFrame()
    
    class AWSAnomalyDetector:
        def __init__(self, *args, **kwargs):
            pass
        def fit(self, *args, **kwargs):
            return self
        def predict_anomalies(self, *args, **kwargs):
            return pd.DataFrame()
    
    class AWSResourceClusterer:
        def __init__(self, *args, **kwargs):
            pass
        def fit(self, *args, **kwargs):
            return self
        def get_cluster_insights(self, *args, **kwargs):
            return []
    
    class AWSOptimizationPredictor:
        def __init__(self, *args, **kwargs):
            pass
        def fit(self, *args, **kwargs):
            return self
        def generate_smart_recommendations(self, *args, **kwargs):
            return []
    
    PROPHET_AVAILABLE = False
    ARIMA_AVAILABLE = False

class AWSMLPipeline:
    """Complete ML pipeline for AWS cost optimization"""
    
    def __init__(self, cache_dir='ml_cache'):
        self.cache_dir = cache_dir
        self.models = {}
        self.last_training_time = None
        self.model_performance = {}
        
        # Create cache directory
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
    
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def load_cached_predictions(_self, data_hash):
        """Load cached predictions if available"""
        cache_file = os.path.join(_self.cache_dir, f'predictions_{data_hash}.joblib')
        if os.path.exists(cache_file):
            return joblib.load(cache_file)
        return None
    
    def save_predictions(self, predictions, data_hash):
        """Save predictions to cache"""
        cache_file = os.path.join(self.cache_dir, f'predictions_{data_hash}.joblib')
        joblib.dump(predictions, cache_file)
    
    def get_data_hash(self, ec2_df, s3_df):
        """Generate hash for data versioning"""
        data_str = f"{len(ec2_df)}_{len(s3_df)}_{ec2_df['CostPerHourUSD'].sum():.2f}_{s3_df['MonthlyCostUSD'].sum():.2f}"
        return str(hash(data_str))
    
    def should_retrain_models(self, ec2_df, s3_df):
        """Determine if models need retraining"""
        # Retrain if:
        # 1. No previous training
        # 2. Data has changed significantly
        # 3. More than 24 hours since last training
        
        if self.last_training_time is None:
            return True
        
        if datetime.now() - self.last_training_time > timedelta(days=1):
            return True
        
        # Check if data size changed significantly
        current_hash = self.get_data_hash(ec2_df, s3_df)
        previous_hash = getattr(self, '_last_data_hash', None)
        
        return current_hash != previous_hash
    
    def train_all_models(self, ec2_df, s3_df, force_retrain=False):
        """Train all ML models"""
        
        if not force_retrain and not self.should_retrain_models(ec2_df, s3_df):
            st.info("Using cached models (trained within last 24 hours)")
            return self._load_cached_models()
        
        st.info("Training ML models... This may take a few minutes.")
        
        training_progress = st.progress(0)
        status_text = st.empty()
        
        try:
            # 1. Cost Forecaster
            status_text.text("Training Cost Forecasting Model...")
            training_progress.progress(25)
            
            self.models['forecaster'] = AWSCostForecaster(model_type='prophet')
            self.models['forecaster'].fit(ec2_df, s3_df)
            
            # 2. Anomaly Detector
            status_text.text("Training Anomaly Detection Model...")
            training_progress.progress(50)
            
            self.models['anomaly_detector'] = AWSAnomalyDetector(contamination=0.1)
            self.models['anomaly_detector'].fit(ec2_df, s3_df)
            
            # 3. Resource Clusterer
            status_text.text("Training Resource Clustering Model...")
            training_progress.progress(75)
            
            self.models['clusterer'] = AWSResourceClusterer(n_clusters=5)
            self.models['clusterer'].fit(ec2_df, s3_df)
            
            # 4. Optimization Predictor
            status_text.text("Training Optimization Prediction Model...")
            training_progress.progress(90)
            
            self.models['optimizer'] = AWSOptimizationPredictor()
            self.models['optimizer'].fit(ec2_df, s3_df)
            
            training_progress.progress(100)
            status_text.text("All models trained successfully!")
            
            # Save models and update training time
            self._save_models()
            self.last_training_time = datetime.now()
            self._last_data_hash = self.get_data_hash(ec2_df, s3_df)
            
            st.success("✅ All ML models trained successfully!")
            
        except Exception as e:
            st.error(f"Error training models: {str(e)}")
            return False
        
        return True
    
    def _save_models(self):
        """Save model metadata (not the actual models to avoid pickle issues)"""
        # Don't pickle the models themselves, just save metadata
        # Models will be retrained when needed (they train quickly)
        model_metadata = {
            'training_time': self.last_training_time,
            'data_hash': getattr(self, '_last_data_hash', None),
            'model_types': list(self.models.keys())
        }
        
        metadata_file = os.path.join(self.cache_dir, 'model_metadata.joblib')
        try:
            joblib.dump(model_metadata, metadata_file)
        except Exception as e:
            # If saving fails, just log it and continue
            print(f"Warning: Could not save model metadata: {e}")
    
    def _load_cached_models(self):
        """Load cached model metadata (models need to be retrained)"""
        metadata_file = os.path.join(self.cache_dir, 'model_metadata.joblib')
        
        if os.path.exists(metadata_file):
            try:
                metadata = joblib.load(metadata_file)
                self.last_training_time = metadata['training_time']
                self._last_data_hash = metadata.get('data_hash')
                # Models themselves are not cached, return False to trigger retraining
                return False
            except Exception as e:
                # If loading metadata fails, just continue
                return False
        
        return False
    
    def generate_all_predictions(self, ec2_df, s3_df):
        """Generate predictions from all models"""
        
        data_hash = self.get_data_hash(ec2_df, s3_df)
        
        # Try loading cached predictions
        cached_predictions = self.load_cached_predictions(data_hash)
        if cached_predictions is not None:
            st.info("Using cached predictions")
            return cached_predictions
        
        # Ensure models are trained
        if not self.models or not all(key in self.models for key in ['forecaster', 'anomaly_detector', 'clusterer', 'optimizer']):
            if not self.train_all_models(ec2_df, s3_df):
                return None
        
        predictions = {}
        
        try:
            # 1. Cost Forecasting
            with st.spinner("Generating cost forecasts..."):
                forecast = self.models['forecaster'].predict(periods=30)
                predictions['forecast'] = {
                    'data': forecast,
                    'chart': self.models['forecaster'].plot_forecast(forecast)
                }
            
            # 2. Anomaly Detection
            with st.spinner("Detecting cost anomalies..."):
                anomalies = self.models['anomaly_detector'].predict_anomalies()
                predictions['anomalies'] = {
                    'data': anomalies,
                    'chart': self.models['anomaly_detector'].plot_anomalies(anomalies)
                }
            
            # 3. Resource Clustering
            with st.spinner("Analyzing resource clusters..."):
                cluster_insights = self.models['clusterer'].get_cluster_insights()
                predictions['clusters'] = {
                    'insights': cluster_insights,
                    'chart': self.models['clusterer'].plot_clusters()
                }
            
            # 4. Optimization Predictions
            with st.spinner("Generating optimization recommendations..."):
                optimization_results = self.models['optimizer'].predict_optimizations(ec2_df, s3_df)
                smart_recommendations = self.models['optimizer'].generate_smart_recommendations(optimization_results)
                predictions['optimization'] = {
                    'results': optimization_results,
                    'recommendations': smart_recommendations
                }
            
            # Cache predictions
            self.save_predictions(predictions, data_hash)
            
        except Exception as e:
            st.error(f"Error generating predictions: {str(e)}")
            return None
        
        return predictions
    
    def get_model_info(self):
        """Get information about trained models"""
        if not self.models:
            return "No models trained yet"
        
        info = {
            'training_time': self.last_training_time,
            'models_available': list(self.models.keys()),
            'cache_dir': self.cache_dir
        }
        
        return info
    
    def clear_cache(self):
        """Clear all cached models and predictions"""
        try:
            import shutil
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)
                os.makedirs(self.cache_dir)
            
            self.models = {}
            self.last_training_time = None
            st.success("Cache cleared successfully!")
            
        except Exception as e:
            st.error(f"Error clearing cache: {str(e)}")

class MLMetrics:
    """Track and display ML model performance metrics"""
    
    @staticmethod
    def display_forecast_metrics(forecast_data):
        """Display forecasting model metrics"""
        # Handle both DataFrame (Prophet/Linear) and Series (ARIMA)
        if isinstance(forecast_data, pd.Series):
            # ARIMA returns a Series
            recent_data = forecast_data.tail(30)
            
            metrics = {
                'Forecast Period': '30 days',
                'Avg Daily Cost': f"${recent_data.mean():.2f}",
                'Max Daily Cost': f"${recent_data.max():.2f}",
                'Min Daily Cost': f"${recent_data.min():.2f}",
                'Monthly Projection': f"${recent_data.sum():.2f}"
            }
        elif isinstance(forecast_data, pd.DataFrame) and 'yhat' in forecast_data.columns:
            # Prophet/Linear returns a DataFrame with 'yhat' column
            recent_data = forecast_data.tail(30)
            
            metrics = {
                'Forecast Period': '30 days',
                'Avg Daily Cost': f"${recent_data['yhat'].mean():.2f}",
                'Max Daily Cost': f"${recent_data['yhat'].max():.2f}",
                'Min Daily Cost': f"${recent_data['yhat'].min():.2f}",
                'Monthly Projection': f"${recent_data['yhat'].sum():.2f}"
            }
        else:
            # Fallback for unsupported format
            return {
                'Forecast Period': 'N/A',
                'Avg Daily Cost': 'N/A',
                'Max Daily Cost': 'N/A',
                'Min Daily Cost': 'N/A',
                'Monthly Projection': 'N/A'
            }
        
        return metrics
    
    @staticmethod
    def display_anomaly_metrics(anomaly_data):
        """Display anomaly detection metrics"""
        total_regions = len(anomaly_data)
        anomalous_regions = anomaly_data['Is_Anomaly'].sum()
        
        metrics = {
            'Total Regions': total_regions,
            'Anomalous Regions': anomalous_regions,
            'Anomaly Rate': f"{(anomalous_regions/total_regions)*100:.1f}%",
            'Most Anomalous': anomaly_data.loc[anomaly_data['Anomaly_Score'].idxmin(), 'Region']
        }
        
        return metrics
    
    @staticmethod
    def display_cluster_metrics(cluster_insights):
        """Display clustering metrics"""
        total_resources = sum(insight['size'] for insight in cluster_insights)
        
        metrics = {
            'Total Clusters': len(cluster_insights),
            'Total Resources': total_resources,
            'Avg Cluster Size': f"{total_resources/len(cluster_insights):.1f}",
            'Most Efficient Cluster': max(cluster_insights, key=lambda x: x['avg_efficiency'])['cluster_id']
        }
        
        return metrics
    
    @staticmethod
    def display_optimization_metrics(recommendations):
        """Display optimization metrics"""
        if not recommendations:
            return {'message': 'No optimization opportunities found'}
        
        total_savings = sum(rec['potential_savings'] for rec in recommendations)
        ec2_count = sum(1 for rec in recommendations if rec['type'] == 'EC2')
        s3_count = sum(1 for rec in recommendations if rec['type'] == 'S3')
        
        metrics = {
            'Total Recommendations': len(recommendations),
            'EC2 Optimizations': ec2_count,
            'S3 Optimizations': s3_count,
            'Potential Monthly Savings': f"${total_savings:.2f}",
            'Potential Annual Savings': f"${total_savings * 12:.2f}"
        }
        
        return metrics