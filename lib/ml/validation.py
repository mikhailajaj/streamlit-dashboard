"""
Model Validation Framework for AWS ML Models
Implements train/test splits, metrics calculation, and model evaluation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, silhouette_score
)
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ModelValidator:
    """
    Comprehensive model validation for AWS cost optimization models.
    
    CRITICAL FIX: Adds proper validation that was completely missing.
    All models must be validated before deployment to ensure they work correctly.
    """
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42, cv_folds: int = 5):
        """
        Initialize validator.
        
        Args:
            test_size: Proportion of data for testing (default 0.2 = 20%)
            random_state: Random seed for reproducibility
            cv_folds: Number of cross-validation folds
        """
        self.test_size = test_size
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.validation_results = {}
    
    def train_test_split_stratified(self, X: pd.DataFrame, y: pd.Series, 
                                   stratify_col: Optional[pd.Series] = None) -> Tuple:
        """
        Split data into train/test with optional stratification.
        
        Args:
            X: Features DataFrame
            y: Target Series
            stratify_col: Column to stratify on (ensures balanced splits)
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if stratify_col is not None:
            return train_test_split(
                X, y, test_size=self.test_size, 
                random_state=self.random_state,
                stratify=stratify_col
            )
        else:
            return train_test_split(
                X, y, test_size=self.test_size, 
                random_state=self.random_state
            )
    
    def calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                    model_name: str = 'model') -> Dict:
        """
        Calculate regression metrics (for cost prediction models).
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            
        Returns:
            Dictionary with regression metrics
        """
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        # Avoid division by zero
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else np.nan
        
        # Calculate median absolute error
        median_ae = np.median(np.abs(y_true - y_pred))
        
        metrics = {
            'model_name': model_name,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'median_ae': median_ae,
            'mape': mape,
            'r2_score': r2,
            'n_samples': len(y_true)
        }
        
        self.validation_results[model_name] = metrics
        return metrics
    
    def calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                        y_pred_proba: Optional[np.ndarray] = None,
                                        model_name: str = 'model') -> Dict:
        """
        Calculate classification metrics (for optimization prediction models).
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            model_name: Name of the model
            
        Returns:
            Dictionary with classification metrics
        """
        # Calculate basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Handle binary vs multiclass
        average = 'binary' if len(np.unique(y_true)) == 2 else 'weighted'
        
        precision = precision_score(y_true, y_pred, average=average, zero_division=0)
        recall = recall_score(y_true, y_pred, average=average, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'n_samples': len(y_true),
            'n_classes': len(np.unique(y_true))
        }
        
        # Add classification report
        metrics['classification_report'] = classification_report(
            y_true, y_pred, zero_division=0, output_dict=True
        )
        
        self.validation_results[model_name] = metrics
        return metrics
    
    def calculate_clustering_metrics(self, X: np.ndarray, cluster_labels: np.ndarray,
                                    model_name: str = 'clustering') -> Dict:
        """
        Calculate clustering quality metrics.
        
        Args:
            X: Feature matrix
            cluster_labels: Cluster assignments
            model_name: Name of the model
            
        Returns:
            Dictionary with clustering metrics
        """
        # Silhouette score (measures cluster separation, range: -1 to 1, higher is better)
        silhouette_avg = silhouette_score(X, cluster_labels)
        
        # Calculate cluster sizes
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        cluster_sizes = dict(zip(unique_labels, counts))
        
        # Calculate cluster balance (coefficient of variation)
        cv = np.std(counts) / np.mean(counts) if np.mean(counts) > 0 else 0
        
        metrics = {
            'model_name': model_name,
            'silhouette_score': silhouette_avg,
            'n_clusters': len(unique_labels),
            'cluster_sizes': cluster_sizes,
            'min_cluster_size': int(np.min(counts)),
            'max_cluster_size': int(np.max(counts)),
            'avg_cluster_size': float(np.mean(counts)),
            'cluster_balance_cv': cv,
            'n_samples': len(cluster_labels)
        }
        
        self.validation_results[model_name] = metrics
        return metrics
    
    def cross_validate_model(self, model, X: pd.DataFrame, y: pd.Series,
                            metric: str = 'r2', model_name: str = 'model') -> Dict:
        """
        Perform k-fold cross-validation.
        
        Args:
            model: Scikit-learn model instance
            X: Features
            y: Target
            metric: Scoring metric (e.g., 'r2', 'neg_mean_squared_error', 'accuracy')
            model_name: Name of the model
            
        Returns:
            Dictionary with cross-validation results
        """
        # Determine if classification or regression
        is_classification = hasattr(model, 'predict_proba')
        
        if is_classification:
            # Use stratified k-fold for classification
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        else:
            cv = self.cv_folds
        
        # Perform cross-validation
        scores = cross_val_score(model, X, y, cv=cv, scoring=metric)
        
        cv_results = {
            'model_name': model_name,
            'metric': metric,
            'mean_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
            'min_score': float(np.min(scores)),
            'max_score': float(np.max(scores)),
            'all_scores': scores.tolist(),
            'n_folds': self.cv_folds
        }
        
        return cv_results
    
    def baseline_comparison(self, y_true: np.ndarray, y_pred: np.ndarray,
                           baseline_strategy: str = 'mean') -> Dict:
        """
        Compare model predictions to baseline strategies.
        
        Args:
            y_true: True values
            y_pred: Model predictions
            baseline_strategy: 'mean', 'median', or 'last_value'
            
        Returns:
            Dictionary comparing model to baseline
        """
        # Calculate model performance
        model_mae = mean_absolute_error(y_true, y_pred)
        model_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # Calculate baseline predictions
        if baseline_strategy == 'mean':
            baseline_pred = np.full_like(y_true, np.mean(y_true))
        elif baseline_strategy == 'median':
            baseline_pred = np.full_like(y_true, np.median(y_true))
        elif baseline_strategy == 'last_value':
            baseline_pred = np.full_like(y_true, y_true[-1])
        else:
            raise ValueError(f"Unknown baseline strategy: {baseline_strategy}")
        
        # Calculate baseline performance
        baseline_mae = mean_absolute_error(y_true, baseline_pred)
        baseline_rmse = np.sqrt(mean_squared_error(y_true, baseline_pred))
        
        # Calculate improvement
        mae_improvement = ((baseline_mae - model_mae) / baseline_mae) * 100 if baseline_mae > 0 else 0
        rmse_improvement = ((baseline_rmse - model_rmse) / baseline_rmse) * 100 if baseline_rmse > 0 else 0
        
        comparison = {
            'baseline_strategy': baseline_strategy,
            'model_mae': model_mae,
            'baseline_mae': baseline_mae,
            'mae_improvement_pct': mae_improvement,
            'model_rmse': model_rmse,
            'baseline_rmse': baseline_rmse,
            'rmse_improvement_pct': rmse_improvement,
            'is_better_than_baseline': model_mae < baseline_mae
        }
        
        return comparison
    
    def calculate_confidence_intervals(self, y_true: np.ndarray, y_pred: np.ndarray,
                                     confidence_level: float = 0.95) -> Dict:
        """
        Calculate confidence intervals for predictions.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            confidence_level: Confidence level (default 0.95 for 95% CI)
            
        Returns:
            Dictionary with confidence interval bounds
        """
        # Calculate residuals
        residuals = y_true - y_pred
        
        # Calculate standard error
        n = len(residuals)
        std_error = np.std(residuals, ddof=1)
        
        # Calculate confidence interval using t-distribution
        from scipy import stats
        alpha = 1 - confidence_level
        t_value = stats.t.ppf(1 - alpha/2, df=n-1)
        
        margin_of_error = t_value * std_error / np.sqrt(n)
        
        ci = {
            'confidence_level': confidence_level,
            'lower_bound': float(np.mean(y_pred) - margin_of_error),
            'upper_bound': float(np.mean(y_pred) + margin_of_error),
            'margin_of_error': float(margin_of_error),
            'std_error': float(std_error)
        }
        
        return ci
    
    def plot_regression_diagnostics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   model_name: str = 'Model') -> go.Figure:
        """
        Create diagnostic plots for regression models.
        
        Returns:
            Plotly figure with diagnostic plots
        """
        residuals = y_true - y_pred
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Actual vs Predicted',
                'Residual Plot',
                'Residual Distribution',
                'Q-Q Plot'
            )
        )
        
        # 1. Actual vs Predicted
        fig.add_trace(
            go.Scatter(
                x=y_true, y=y_pred,
                mode='markers',
                name='Predictions',
                marker=dict(color='blue', opacity=0.6)
            ),
            row=1, col=1
        )
        # Add perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ),
            row=1, col=1
        )
        
        # 2. Residual Plot
        fig.add_trace(
            go.Scatter(
                x=y_pred, y=residuals,
                mode='markers',
                name='Residuals',
                marker=dict(color='green', opacity=0.6)
            ),
            row=1, col=2
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)
        
        # 3. Residual Distribution
        fig.add_trace(
            go.Histogram(
                x=residuals,
                name='Residuals',
                marker=dict(color='purple'),
                nbinsx=30
            ),
            row=2, col=1
        )
        
        # 4. Q-Q Plot (check normality of residuals)
        from scipy import stats
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
        sample_quantiles = np.sort(residuals)
        
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=sample_quantiles,
                mode='markers',
                name='Q-Q',
                marker=dict(color='orange', opacity=0.6)
            ),
            row=2, col=2
        )
        # Add reference line
        fig.add_trace(
            go.Scatter(
                x=[theoretical_quantiles.min(), theoretical_quantiles.max()],
                y=[sample_quantiles.min(), sample_quantiles.max()],
                mode='lines',
                name='Normal',
                line=dict(color='red', dash='dash')
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_xaxes(title_text="Actual Values", row=1, col=1)
        fig.update_yaxes(title_text="Predicted Values", row=1, col=1)
        fig.update_xaxes(title_text="Predicted Values", row=1, col=2)
        fig.update_yaxes(title_text="Residuals", row=1, col=2)
        fig.update_xaxes(title_text="Residuals", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=2)
        fig.update_yaxes(title_text="Sample Quantiles", row=2, col=2)
        
        fig.update_layout(
            title=f'{model_name} - Regression Diagnostics',
            height=800,
            showlegend=False
        )
        
        return fig
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str] = None,
                            model_name: str = 'Model') -> go.Figure:
        """
        Create interactive confusion matrix heatmap.
        
        Args:
            cm: Confusion matrix
            class_names: Names of classes
            model_name: Name of the model
            
        Returns:
            Plotly figure
        """
        if class_names is None:
            class_names = [f'Class {i}' for i in range(cm.shape[0])]
        
        # Calculate percentages
        cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create annotations
        annotations = []
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annotations.append(
                    dict(
                        x=j, y=i,
                        text=f"{cm[i, j]}<br>({cm_pct[i, j]:.1f}%)",
                        showarrow=False,
                        font=dict(color='white' if cm_pct[i, j] > 50 else 'black')
                    )
                )
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=class_names,
            y=class_names,
            colorscale='Blues',
            showscale=True
        ))
        
        fig.update_layout(
            title=f'{model_name} - Confusion Matrix',
            xaxis_title='Predicted Label',
            yaxis_title='True Label',
            annotations=annotations
        )
        
        return fig
    
    def generate_validation_report(self) -> pd.DataFrame:
        """
        Generate comprehensive validation report for all validated models.
        
        Returns:
            DataFrame with validation metrics for all models
        """
        if not self.validation_results:
            return pd.DataFrame()
        
        report_data = []
        for model_name, metrics in self.validation_results.items():
            row = {'model_name': model_name}
            
            # Add all metrics except nested structures
            for key, value in metrics.items():
                if key not in ['confusion_matrix', 'classification_report', 'cluster_sizes']:
                    row[key] = value
            
            report_data.append(row)
        
        return pd.DataFrame(report_data)
    
    def get_validation_summary(self, model_name: str) -> str:
        """
        Get text summary of validation results for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Formatted text summary
        """
        if model_name not in self.validation_results:
            return f"No validation results found for {model_name}"
        
        metrics = self.validation_results[model_name]
        
        summary = f"Validation Summary: {model_name}\n"
        summary += "=" * 50 + "\n"
        
        # Format based on metric type
        if 'rmse' in metrics:  # Regression
            summary += f"RMSE: {metrics['rmse']:.4f}\n"
            summary += f"MAE: {metrics['mae']:.4f}\n"
            summary += f"MAPE: {metrics['mape']:.2f}%\n"
            summary += f"R² Score: {metrics['r2_score']:.4f}\n"
        elif 'accuracy' in metrics:  # Classification
            summary += f"Accuracy: {metrics['accuracy']:.4f}\n"
            summary += f"Precision: {metrics['precision']:.4f}\n"
            summary += f"Recall: {metrics['recall']:.4f}\n"
            summary += f"F1 Score: {metrics['f1_score']:.4f}\n"
        elif 'silhouette_score' in metrics:  # Clustering
            summary += f"Silhouette Score: {metrics['silhouette_score']:.4f}\n"
            summary += f"Number of Clusters: {metrics['n_clusters']}\n"
            summary += f"Avg Cluster Size: {metrics['avg_cluster_size']:.1f}\n"
        
        summary += f"Sample Size: {metrics['n_samples']}\n"
        
        return summary
