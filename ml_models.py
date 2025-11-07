"""
Advanced Machine Learning Models for AWS Cost Analysis and Optimization
Implements: Time Series Forecasting, Anomaly Detection, Clustering, and Classification
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error, silhouette_score

# Optional ML libraries with graceful fallbacks

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError as e:
    PROPHET_AVAILABLE = False
    print(f"⚠️  Prophet not available: {e}")

try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except ImportError as e:
    ARIMA_AVAILABLE = False
    print(f"⚠️  ARIMA not available: {e}")
import joblib
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class AWSCostForecaster:
    """Time Series Forecasting for AWS Cost Prediction"""
    
    def __init__(self, model_type='prophet'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def prepare_timeseries_data(self, ec2_df, s3_df):
        """Prepare time series data for forecasting"""
        # Aggregate daily costs
        ec2_df['date'] = pd.to_datetime(ec2_df['LaunchTime']).dt.date
        s3_df['date'] = pd.to_datetime(s3_df['CreationDate']).dt.date
        
        # Calculate daily costs
        ec2_daily = ec2_df.groupby('date')['CostPerHourUSD'].sum() * 24
        s3_daily = s3_df.groupby('date')['MonthlyCostUSD'].sum() / 30  # Convert to daily
        
        # Combine and create time series
        date_range = pd.date_range(
            start=min(ec2_daily.index.min(), s3_daily.index.min()),
            end=max(ec2_daily.index.max(), s3_daily.index.max()),
            freq='D'
        )
        
        ts_data = pd.DataFrame({
            'ds': date_range,
            'ec2_cost': ec2_daily.reindex(date_range, fill_value=0),
            's3_cost': s3_daily.reindex(date_range, fill_value=0)
        })
        ts_data['y'] = ts_data['ec2_cost'] + ts_data['s3_cost']  # Total cost
        
        return ts_data
    
    def fit(self, ec2_df, s3_df):
        """Train the forecasting model"""
        self.data = self.prepare_timeseries_data(ec2_df, s3_df)
        
        if self.model_type == 'prophet' and PROPHET_AVAILABLE:
            self.model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05
            )
            self.model.fit(self.data[['ds', 'y']])
        
        elif self.model_type == 'arima' and ARIMA_AVAILABLE:
            # Simple ARIMA model
            self.model = ARIMA(self.data['y'], order=(1, 1, 1))
            self.model = self.model.fit()
        
        else:
            # Fallback to simple linear regression if Prophet/ARIMA unavailable
            from sklearn.linear_model import LinearRegression
            print(f"🔄 Using Linear Regression fallback for {self.model_type}")
            
            # Create time-based features for linear regression
            self.data['days_since_start'] = (self.data['ds'] - self.data['ds'].min()).dt.days
            X = self.data[['days_since_start']].values
            y = self.data['y'].values
            
            self.model = LinearRegression()
            self.model.fit(X, y)
            self.model_type = 'linear'  # Update model type for prediction
        
        self.is_fitted = True
        return self
    
    def predict(self, periods=30):
        """Generate cost forecasts"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if self.model_type == 'prophet':
            future = self.model.make_future_dataframe(periods=periods)
            forecast = self.model.predict(future)
            return forecast
        
        elif self.model_type == 'arima':
            forecast = self.model.forecast(steps=periods)
            return forecast
            
        elif self.model_type == 'linear':
            # Linear regression fallback prediction
            last_day = self.data['days_since_start'].max()
            future_days = np.arange(last_day + 1, last_day + periods + 1).reshape(-1, 1)
            future_values = self.model.predict(future_days)
            
            # Create Prophet-like output format for compatibility
            future_dates = pd.date_range(
                start=self.data['ds'].max() + pd.Timedelta(days=1),
                periods=periods,
                freq='D'
            )
            
            forecast = pd.DataFrame({
                'ds': future_dates,
                'yhat': future_values,
                'yhat_lower': future_values * 0.9,  # Simple confidence interval
                'yhat_upper': future_values * 1.1
            })
            return forecast
    
    def plot_forecast(self, forecast_data, periods=30):
        """Create interactive forecast visualization"""
        if self.model_type == 'prophet':
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=self.data['ds'],
                y=self.data['y'],
                mode='lines',
                name='Historical Costs',
                line=dict(color='blue')
            ))
            
            # Forecast
            future_dates = forecast_data['ds'].tail(periods)
            future_values = forecast_data['yhat'].tail(periods)
            
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=future_values,
                mode='lines',
                name='Forecast',
                line=dict(color='red', dash='dash')
            ))
            
            # Confidence intervals
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=forecast_data['yhat_upper'].tail(periods),
                fill=None,
                mode='lines',
                line_color='rgba(0,0,0,0)',
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=forecast_data['yhat_lower'].tail(periods),
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,0,0,0)',
                name='Confidence Interval',
                fillcolor='rgba(255,0,0,0.2)'
            ))
            
            fig.update_layout(
                title='AWS Cost Forecast - Next 30 Days',
                xaxis_title='Date',
                yaxis_title='Daily Cost (USD)',
                hovermode='x unified'
            )
            
            return fig

class AWSAnomalyDetector:
    """Anomaly Detection for unusual cost patterns"""
    
    def __init__(self, contamination=0.1):
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def prepare_features(self, ec2_df, s3_df):
        """Prepare features for anomaly detection"""
        # EC2 features
        ec2_features = ec2_df.groupby('Region').agg({
            'CostPerHourUSD': ['sum', 'mean', 'std'],
            'CPUUtilization': ['mean', 'std'],
            'MemoryUtilization': ['mean', 'std']
        }).fillna(0)
        
        # S3 features
        s3_features = s3_df.groupby('Region').agg({
            'MonthlyCostUSD': ['sum', 'mean', 'std'],
            'TotalSizeGB': ['sum', 'mean', 'std'],
            'ObjectCount': ['sum', 'mean']
        }).fillna(0)
        
        # Flatten column names
        ec2_features.columns = ['_'.join(col) for col in ec2_features.columns]
        s3_features.columns = ['_'.join(col) for col in s3_features.columns]
        
        # Combine features
        features = pd.concat([ec2_features, s3_features], axis=1).fillna(0)
        return features
    
    def fit(self, ec2_df, s3_df):
        """Train anomaly detection model"""
        self.features = self.prepare_features(ec2_df, s3_df)
        self.features_scaled = self.scaler.fit_transform(self.features)
        self.model.fit(self.features_scaled)
        self.is_fitted = True
        return self
    
    def predict_anomalies(self):
        """Detect anomalies in current data"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        anomaly_scores = self.model.decision_function(self.features_scaled)
        anomalies = self.model.predict(self.features_scaled)
        
        results = pd.DataFrame({
            'Region': self.features.index,
            'Anomaly_Score': anomaly_scores,
            'Is_Anomaly': anomalies == -1
        })
        
        return results.sort_values('Anomaly_Score')
    
    def plot_anomalies(self, anomaly_results):
        """Visualize anomaly detection results"""
        fig = px.scatter(
            anomaly_results,
            x='Region',
            y='Anomaly_Score',
            color='Is_Anomaly',
            title='AWS Cost Anomaly Detection by Region',
            color_discrete_map={True: 'red', False: 'blue'}
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                      annotation_text="Anomaly Threshold")
        
        return fig

class AWSResourceClusterer:
    """Smart clustering for resource optimization"""
    
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters, random_state=42)
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def prepare_features(self, ec2_df, s3_df):
        """Prepare features for clustering"""
        # EC2 clustering features
        ec2_features = ec2_df[['CPUUtilization', 'MemoryUtilization', 'CostPerHourUSD']].copy()
        ec2_features['ResourceType'] = 'EC2'
        ec2_features['Efficiency'] = ec2_features['CPUUtilization'] / (ec2_features['CostPerHourUSD'] + 0.01)
        
        # S3 clustering features (normalize to similar scale)
        s3_features = pd.DataFrame({
            'CPUUtilization': s3_df['TotalSizeGB'] / s3_df['TotalSizeGB'].max() * 100,  # Normalized usage
            'MemoryUtilization': s3_df['ObjectCount'] / s3_df['ObjectCount'].max() * 100,  # Normalized objects
            'CostPerHourUSD': s3_df['MonthlyCostUSD'] / 30 / 24,  # Convert to hourly
            'ResourceType': 'S3',
            'Efficiency': (s3_df['TotalSizeGB'] / (s3_df['MonthlyCostUSD'] + 0.01))
        })
        
        # Combine features
        all_features = pd.concat([ec2_features, s3_features], ignore_index=True)
        return all_features
    
    def fit(self, ec2_df, s3_df):
        """Train clustering model"""
        self.data = self.prepare_features(ec2_df, s3_df)
        self.features = self.data[['CPUUtilization', 'MemoryUtilization', 'CostPerHourUSD', 'Efficiency']]
        self.features_scaled = self.scaler.fit_transform(self.features)
        
        # Find optimal number of clusters
        silhouette_scores = []
        K_range = range(2, min(11, len(self.features)))
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            cluster_labels = kmeans.fit_predict(self.features_scaled)
            silhouette_avg = silhouette_score(self.features_scaled, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        
        # Use best k
        best_k = K_range[np.argmax(silhouette_scores)]
        self.model = KMeans(n_clusters=best_k, random_state=42)
        self.clusters = self.model.fit_predict(self.features_scaled)
        
        self.data['Cluster'] = self.clusters
        self.is_fitted = True
        return self
    
    def get_cluster_insights(self):
        """Generate insights about each cluster"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before generating insights")
        
        insights = []
        for cluster_id in range(self.model.n_clusters):
            cluster_data = self.data[self.data['Cluster'] == cluster_id]
            
            insight = {
                'cluster_id': cluster_id,
                'size': len(cluster_data),
                'avg_cpu': cluster_data['CPUUtilization'].mean(),
                'avg_memory': cluster_data['MemoryUtilization'].mean(),
                'avg_cost': cluster_data['CostPerHourUSD'].mean(),
                'avg_efficiency': cluster_data['Efficiency'].mean(),
                'resource_types': cluster_data['ResourceType'].value_counts().to_dict()
            }
            
            # Generate recommendation
            if insight['avg_efficiency'] < 1.0 and insight['avg_cost'] > 0.1:
                insight['recommendation'] = "High cost, low efficiency - Consider optimization"
            elif insight['avg_efficiency'] > 2.0:
                insight['recommendation'] = "Well optimized - Good performance"
            else:
                insight['recommendation'] = "Moderate efficiency - Monitor usage"
                
            insights.append(insight)
        
        return insights
    
    def plot_clusters(self):
        """Visualize clustering results"""
        fig = px.scatter_3d(
            self.data,
            x='CPUUtilization',
            y='CostPerHourUSD',
            z='Efficiency',
            color='Cluster',
            symbol='ResourceType',
            title='AWS Resource Clustering Analysis',
            labels={
                'CPUUtilization': 'CPU/Usage %',
                'CostPerHourUSD': 'Cost per Hour (USD)',
                'Efficiency': 'Efficiency Score'
            }
        )
        
        return fig

class AWSOptimizationPredictor:
    """ML-powered optimization recommendations"""
    
    def __init__(self):
        self.ec2_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.s3_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        
    def prepare_training_data(self, ec2_df, s3_df):
        """Prepare labeled training data for optimization prediction"""
        # EC2 optimization labels (based on business rules)
        ec2_features = ec2_df[['CPUUtilization', 'MemoryUtilization', 'CostPerHourUSD']].copy()
        ec2_features['optimization_needed'] = (
            (ec2_df['CPUUtilization'] < 20) |  # Low CPU
            (ec2_df['CostPerHourUSD'] > ec2_df['CostPerHourUSD'].quantile(0.8))  # High cost
        ).astype(int)
        
        # S3 optimization labels
        s3_features = s3_df[['TotalSizeGB', 'MonthlyCostUSD', 'ObjectCount']].copy()
        s3_features['optimization_needed'] = (
            (s3_df['StorageClass'] == 'STANDARD') &  # Standard storage
            (s3_df['MonthlyCostUSD'] > s3_df['MonthlyCostUSD'].quantile(0.7))  # High cost
        ).astype(int)
        
        return ec2_features, s3_features
    
    def fit(self, ec2_df, s3_df):
        """Train optimization prediction models"""
        self.ec2_features, self.s3_features = self.prepare_training_data(ec2_df, s3_df)
        
        # Train EC2 model
        X_ec2 = self.ec2_features[['CPUUtilization', 'MemoryUtilization', 'CostPerHourUSD']]
        y_ec2 = self.ec2_features['optimization_needed']
        
        if len(X_ec2) > 0:
            X_ec2_scaled = self.scaler.fit_transform(X_ec2)
            self.ec2_model.fit(X_ec2_scaled, y_ec2)
        
        # Train S3 model
        X_s3 = self.s3_features[['TotalSizeGB', 'MonthlyCostUSD', 'ObjectCount']]
        y_s3 = self.s3_features['optimization_needed']
        
        if len(X_s3) > 0:
            self.s3_model.fit(X_s3, y_s3)
        
        self.is_fitted = True
        return self
    
    def predict_optimizations(self, ec2_df, s3_df):
        """Predict optimization opportunities"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        results = {'ec2': [], 's3': []}
        
        # EC2 predictions
        if len(ec2_df) > 0:
            X_ec2 = ec2_df[['CPUUtilization', 'MemoryUtilization', 'CostPerHourUSD']]
            X_ec2_scaled = self.scaler.transform(X_ec2)
            
            ec2_predictions = self.ec2_model.predict(X_ec2_scaled)
            ec2_probabilities = self.ec2_model.predict_proba(X_ec2_scaled)[:, 1]
            
            ec2_results = ec2_df.copy()
            ec2_results['optimization_needed'] = ec2_predictions
            ec2_results['optimization_probability'] = ec2_probabilities
            
            results['ec2'] = ec2_results
        
        # S3 predictions
        if len(s3_df) > 0:
            X_s3 = s3_df[['TotalSizeGB', 'MonthlyCostUSD', 'ObjectCount']]
            
            s3_predictions = self.s3_model.predict(X_s3)
            s3_probabilities = self.s3_model.predict_proba(X_s3)[:, 1]
            
            s3_results = s3_df.copy()
            s3_results['optimization_needed'] = s3_predictions
            s3_results['optimization_probability'] = s3_probabilities
            
            results['s3'] = s3_results
        
        return results
    
    def generate_smart_recommendations(self, predictions):
        """Generate intelligent optimization recommendations"""
        recommendations = []
        
        # EC2 recommendations
        if len(predictions['ec2']) > 0:
            high_prob_ec2 = predictions['ec2'][
                predictions['ec2']['optimization_probability'] > 0.7
            ].sort_values('optimization_probability', ascending=False)
            
            for _, instance in high_prob_ec2.head(5).iterrows():
                rec = {
                    'type': 'EC2',
                    'resource_id': instance.get('InstanceId', 'Unknown'),
                    'current_cost': instance['CostPerHourUSD'],
                    'cpu_utilization': instance['CPUUtilization'],
                    'confidence': instance['optimization_probability'],
                    'action': self._get_ec2_action(instance),
                    'potential_savings': instance['CostPerHourUSD'] * 0.3 * 24 * 30  # 30% savings estimate
                }
                recommendations.append(rec)
        
        # S3 recommendations
        if len(predictions['s3']) > 0:
            high_prob_s3 = predictions['s3'][
                predictions['s3']['optimization_probability'] > 0.7
            ].sort_values('optimization_probability', ascending=False)
            
            for _, bucket in high_prob_s3.head(5).iterrows():
                rec = {
                    'type': 'S3',
                    'resource_id': bucket.get('BucketName', 'Unknown'),
                    'current_cost': bucket['MonthlyCostUSD'],
                    'storage_size': bucket['TotalSizeGB'],
                    'confidence': bucket['optimization_probability'],
                    'action': self._get_s3_action(bucket),
                    'potential_savings': bucket['MonthlyCostUSD'] * 0.4  # 40% savings estimate
                }
                recommendations.append(rec)
        
        return recommendations
    
    def _get_ec2_action(self, instance):
        """Determine specific EC2 optimization action"""
        if instance['CPUUtilization'] < 10:
            return "Consider terminating - Very low utilization"
        elif instance['CPUUtilization'] < 25:
            return "Downsize instance type - Low utilization"
        elif instance['CostPerHourUSD'] > 1.0:
            return "Consider spot instances - High cost"
        else:
            return "Monitor closely - Moderate optimization potential"
    
    def _get_s3_action(self, bucket):
        """Determine specific S3 optimization action"""
        if bucket.get('StorageClass') == 'STANDARD':
            return "Implement lifecycle policy - Move to IA/Glacier"
        elif bucket['MonthlyCostUSD'] > 100:
            return "Review storage class - High cost bucket"
        else:
            return "Optimize access patterns - Reduce costs"

# Utility functions for model management
def save_models(models, filepath='aws_ml_models.joblib'):
    """Save trained models to disk"""
    joblib.dump(models, filepath)
    return filepath

def load_models(filepath='aws_ml_models.joblib'):
    """Load trained models from disk"""
    return joblib.load(filepath)

def evaluate_model_performance(y_true, y_pred, model_type='classification'):
    """Evaluate model performance"""
    if model_type == 'classification':
        return classification_report(y_true, y_pred)
    elif model_type == 'regression':
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
        }