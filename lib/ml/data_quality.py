"""
Data Quality Analysis and Reporting for AWS Cost Data
Identifies and documents data quality issues before ML training
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class DataQualityAnalyzer:
    """
    Comprehensive data quality analysis for AWS cost data.
    
    CRITICAL FIX: Addresses the 40% missing data issue.
    Analyzes patterns in missing data to determine if it's:
    - Pagination issue (nulls at end)
    - API failure
    - Regional/type-specific pattern
    """
    
    def __init__(self):
        self.quality_reports = {}
        
    def analyze_missing_data(self, df: pd.DataFrame, dataset_name: str = 'dataset') -> Dict:
        """
        Comprehensive analysis of missing data patterns.
        
        Args:
            df: DataFrame to analyze
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with missing data analysis
        """
        total_rows = len(df)
        total_cols = len(df.columns)
        
        # Overall statistics
        missing_counts = df.isnull().sum()
        missing_pct = (missing_counts / total_rows * 100).round(2)
        
        # Identify completely NULL rows
        completely_null_rows = df.isnull().all(axis=1).sum()
        completely_null_pct = (completely_null_rows / total_rows * 100).round(2)
        
        # Identify rows with ANY missing values
        rows_with_nulls = df.isnull().any(axis=1).sum()
        rows_with_nulls_pct = (rows_with_nulls / total_rows * 100).round(2)
        
        # Check for pagination pattern (nulls concentrated at end)
        null_row_indices = df[df.isnull().all(axis=1)].index.tolist()
        is_pagination_issue = False
        if null_row_indices:
            # Check if nulls are consecutive and at the end
            is_consecutive = all(
                null_row_indices[i] + 1 == null_row_indices[i + 1] 
                for i in range(len(null_row_indices) - 1)
            )
            is_at_end = null_row_indices[-1] == total_rows - 1
            is_pagination_issue = is_consecutive and is_at_end
        
        # Missing data by column
        missing_by_column = pd.DataFrame({
            'column': df.columns,
            'missing_count': missing_counts.values,
            'missing_percentage': missing_pct.values,
            'data_type': df.dtypes.values
        }).sort_values('missing_percentage', ascending=False)
        
        # Analyze missing data patterns
        patterns = self._identify_missing_patterns(df)
        
        report = {
            'dataset_name': dataset_name,
            'total_rows': total_rows,
            'total_columns': total_cols,
            'completely_null_rows': completely_null_rows,
            'completely_null_percentage': completely_null_pct,
            'rows_with_any_nulls': rows_with_nulls,
            'rows_with_nulls_percentage': rows_with_nulls_pct,
            'is_pagination_issue': is_pagination_issue,
            'null_row_indices_sample': null_row_indices[:10] if null_row_indices else [],
            'missing_by_column': missing_by_column.to_dict('records'),
            'patterns': patterns,
            'data_quality_score': self._calculate_quality_score(df)
        }
        
        self.quality_reports[dataset_name] = report
        return report
    
    def _identify_missing_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Identify patterns in missing data.
        """
        patterns = {
            'random_missing': False,
            'column_specific': False,
            'row_blocks': False,
            'correlated_missing': []
        }
        
        # Check for column-specific missingness
        missing_cols = df.columns[df.isnull().any()].tolist()
        if len(missing_cols) < len(df.columns):
            patterns['column_specific'] = True
        
        # Check for row blocks (consecutive rows with nulls)
        null_rows = df.isnull().any(axis=1)
        if null_rows.any():
            # Find consecutive sequences
            consecutive_nulls = []
            start = None
            for i, is_null in enumerate(null_rows):
                if is_null and start is None:
                    start = i
                elif not is_null and start is not None:
                    consecutive_nulls.append((start, i - 1))
                    start = None
            if start is not None:
                consecutive_nulls.append((start, len(null_rows) - 1))
            
            if consecutive_nulls:
                patterns['row_blocks'] = True
                patterns['row_block_ranges'] = consecutive_nulls
        
        # Check for correlated missing (columns that are missing together)
        if len(missing_cols) > 1:
            for i, col1 in enumerate(missing_cols):
                for col2 in missing_cols[i+1:]:
                    # Check if they're missing in the same rows
                    both_missing = (df[col1].isnull() & df[col2].isnull()).sum()
                    total_missing = max(df[col1].isnull().sum(), df[col2].isnull().sum())
                    
                    if both_missing / total_missing > 0.9:  # 90% overlap
                        patterns['correlated_missing'].append((col1, col2))
        
        return patterns
    
    def _calculate_quality_score(self, df: pd.DataFrame) -> float:
        """
        Calculate overall data quality score (0-100).
        
        Factors:
        - Completeness (70% weight)
        - Validity (20% weight)
        - Consistency (10% weight)
        """
        # Completeness score
        completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        
        # Validity score (check for reasonable values in numeric columns)
        validity_scores = []
        for col in df.select_dtypes(include=[np.number]).columns:
            non_null = df[col].dropna()
            if len(non_null) > 0:
                # Check for negative values where they shouldn't exist (e.g., costs)
                if 'cost' in col.lower():
                    validity = (non_null >= 0).sum() / len(non_null)
                else:
                    validity = 1.0
                validity_scores.append(validity)
        
        validity = np.mean(validity_scores) * 100 if validity_scores else 100
        
        # Consistency score (check for duplicates)
        if 'ResourceId' in df.columns or 'BucketName' in df.columns:
            id_col = 'ResourceId' if 'ResourceId' in df.columns else 'BucketName'
            non_null_ids = df[id_col].dropna()
            consistency = (1 - non_null_ids.duplicated().sum() / len(non_null_ids)) * 100 if len(non_null_ids) > 0 else 100
        else:
            consistency = 100
        
        # Weighted average
        quality_score = (completeness * 0.7) + (validity * 0.2) + (consistency * 0.1)
        
        return round(quality_score, 2)
    
    def check_data_validity(self, df: pd.DataFrame, dataset_type: str = 'ec2') -> Dict:
        """
        Check for invalid data values.
        
        Args:
            df: DataFrame to check
            dataset_type: 'ec2' or 's3'
            
        Returns:
            Dictionary with validity issues
        """
        issues = {
            'negative_costs': 0,
            'invalid_utilization': 0,
            'invalid_regions': 0,
            'future_dates': 0,
            'other_issues': []
        }
        
        # Check for negative costs
        cost_cols = [col for col in df.columns if 'cost' in col.lower()]
        for col in cost_cols:
            if col in df.columns:
                negative_count = (df[col] < 0).sum()
                issues['negative_costs'] += negative_count
        
        # Check for invalid utilization (should be 0-100)
        if dataset_type == 'ec2':
            util_cols = ['CPUUtilization', 'MemoryUtilization']
            for col in util_cols:
                if col in df.columns:
                    invalid = ((df[col] < 0) | (df[col] > 100)).sum()
                    issues['invalid_utilization'] += invalid
        
        # Check for invalid regions
        if 'Region' in df.columns:
            valid_regions = [
                'us-east-1', 'us-east-2', 'us-west-1', 'us-west-2',
                'eu-west-1', 'eu-central-1', 'ap-south-1', 'ap-southeast-1'
            ]
            invalid_regions = ~df['Region'].isin(valid_regions + [np.nan, None])
            issues['invalid_regions'] = invalid_regions.sum()
        
        # Check for future dates
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        for col in date_cols:
            if col in df.columns:
                try:
                    dates = pd.to_datetime(df[col], errors='coerce')
                    future_dates = (dates > pd.Timestamp.now()).sum()
                    issues['future_dates'] += future_dates
                except:
                    pass
        
        return issues
    
    def recommend_cleaning_strategy(self, missing_report: Dict) -> List[Dict]:
        """
        Recommend data cleaning strategies based on missing data analysis.
        
        Args:
            missing_report: Missing data report from analyze_missing_data()
            
        Returns:
            List of recommended cleaning strategies
        """
        recommendations = []
        
        # If pagination issue, simply drop the null rows
        if missing_report['is_pagination_issue']:
            recommendations.append({
                'priority': 'HIGH',
                'issue': 'Pagination Issue Detected',
                'description': f"{missing_report['completely_null_rows']} completely NULL rows at end of dataset",
                'recommendation': 'Drop completely NULL rows - these are pagination artifacts',
                'code': 'df = df.dropna(how="all")',
                'expected_impact': f"Remove {missing_report['completely_null_rows']} rows, retain all valid data"
            })
        
        # If high missing percentage but not pagination
        elif missing_report['completely_null_percentage'] > 30:
            recommendations.append({
                'priority': 'HIGH',
                'issue': 'High Percentage of Missing Data',
                'description': f"{missing_report['completely_null_percentage']:.1f}% of rows have missing data",
                'recommendation': 'Investigate data source - potential API or extraction issue',
                'code': 'Contact data engineering team to verify data pipeline',
                'expected_impact': 'Resolve underlying data collection issue'
            })
        
        # Column-specific missing data
        high_missing_cols = [
            col for col in missing_report['missing_by_column']
            if col['missing_percentage'] > 50 and col['missing_percentage'] < 100
        ]
        
        if high_missing_cols:
            col_names = [col['column'] for col in high_missing_cols]
            recommendations.append({
                'priority': 'MEDIUM',
                'issue': 'Columns with High Missing Rate',
                'description': f"Columns {col_names} have >50% missing values",
                'recommendation': 'Consider dropping these columns or imputing strategically',
                'code': f"df = df.drop(columns={col_names})",
                'expected_impact': 'Improve data quality for ML models'
            })
        
        # Low missing percentage - can impute
        low_missing_cols = [
            col for col in missing_report['missing_by_column']
            if 0 < col['missing_percentage'] <= 10
        ]
        
        if low_missing_cols:
            recommendations.append({
                'priority': 'LOW',
                'issue': 'Sporadic Missing Values',
                'description': f"{len(low_missing_cols)} columns with <10% missing",
                'recommendation': 'Impute with median (numeric) or mode (categorical)',
                'code': 'df.fillna(df.median(numeric_only=True), inplace=True)',
                'expected_impact': 'Preserve maximum data for analysis'
            })
        
        return recommendations
    
    def clean_dataset(self, df: pd.DataFrame, strategy: str = 'auto') -> Tuple[pd.DataFrame, Dict]:
        """
        Clean dataset based on strategy.
        
        Args:
            df: DataFrame to clean
            strategy: 'auto', 'conservative', or 'aggressive'
            
        Returns:
            Tuple of (cleaned_df, cleaning_report)
        """
        df_clean = df.copy()
        report = {
            'original_rows': len(df),
            'original_cols': len(df.columns),
            'actions_taken': []
        }
        
        # 1. Drop completely NULL rows (always safe)
        completely_null = df_clean.isnull().all(axis=1).sum()
        if completely_null > 0:
            df_clean = df_clean.dropna(how='all')
            report['actions_taken'].append(f"Dropped {completely_null} completely NULL rows")
        
        if strategy in ['auto', 'aggressive']:
            # 2. Drop columns with >90% missing data
            missing_pct = df_clean.isnull().sum() / len(df_clean)
            cols_to_drop = missing_pct[missing_pct > 0.9].index.tolist()
            if cols_to_drop:
                df_clean = df_clean.drop(columns=cols_to_drop)
                report['actions_taken'].append(f"Dropped {len(cols_to_drop)} columns with >90% missing")
        
        if strategy in ['auto', 'conservative', 'aggressive']:
            # 3. Impute numeric columns with median
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df_clean[col].isnull().any():
                    median_val = df_clean[col].median()
                    nulls_filled = df_clean[col].isnull().sum()
                    df_clean[col].fillna(median_val, inplace=True)
                    report['actions_taken'].append(f"Imputed {nulls_filled} nulls in {col} with median={median_val:.2f}")
        
        report['final_rows'] = len(df_clean)
        report['final_cols'] = len(df_clean.columns)
        report['rows_removed'] = report['original_rows'] - report['final_rows']
        report['cols_removed'] = report['original_cols'] - report['final_cols']
        
        return df_clean, report
    
    def plot_missing_data_heatmap(self, df: pd.DataFrame, dataset_name: str = 'Dataset') -> go.Figure:
        """
        Create heatmap visualization of missing data patterns.
        
        Args:
            df: DataFrame to visualize
            dataset_name: Name of the dataset
            
        Returns:
            Plotly figure
        """
        # Create binary matrix (1 = missing, 0 = present)
        missing_matrix = df.isnull().astype(int)
        
        # Sample if too many rows
        if len(missing_matrix) > 200:
            missing_matrix = missing_matrix.sample(n=200, random_state=42).sort_index()
        
        fig = go.Figure(data=go.Heatmap(
            z=missing_matrix.T.values,
            x=missing_matrix.index,
            y=missing_matrix.columns,
            colorscale=[[0, 'lightgreen'], [1, 'red']],
            showscale=True,
            colorbar=dict(
                title="Missing",
                tickvals=[0, 1],
                ticktext=['Present', 'Missing']
            )
        ))
        
        fig.update_layout(
            title=f'{dataset_name} - Missing Data Heatmap',
            xaxis_title='Row Index',
            yaxis_title='Column',
            height=600,
            xaxis=dict(showticklabels=False)  # Hide row indices for clarity
        )
        
        return fig
    
    def generate_quality_report(self, ec2_df: pd.DataFrame = None, s3_df: pd.DataFrame = None) -> str:
        """
        Generate comprehensive text report of data quality.
        
        Args:
            ec2_df: EC2 DataFrame (optional)
            s3_df: S3 DataFrame (optional)
            
        Returns:
            Formatted text report
        """
        report = "=" * 70 + "\n"
        report += "AWS DATA QUALITY REPORT\n"
        report += "=" * 70 + "\n\n"
        
        if ec2_df is not None:
            ec2_report = self.analyze_missing_data(ec2_df, 'EC2')
            report += "EC2 COMPUTE RESOURCES\n"
            report += "-" * 70 + "\n"
            report += f"Total Rows: {ec2_report['total_rows']}\n"
            report += f"Total Columns: {ec2_report['total_columns']}\n"
            report += f"Completely NULL Rows: {ec2_report['completely_null_rows']} ({ec2_report['completely_null_percentage']:.1f}%)\n"
            report += f"Data Quality Score: {ec2_report['data_quality_score']:.1f}/100\n"
            
            if ec2_report['is_pagination_issue']:
                report += "\n⚠️  PAGINATION ISSUE DETECTED!\n"
                report += "   NULL rows are consecutive and at the end of the dataset.\n"
                report += "   Recommendation: Drop these rows - they are pagination artifacts.\n"
            
            report += "\nTop Missing Columns:\n"
            for col_info in ec2_report['missing_by_column'][:5]:
                if col_info['missing_percentage'] > 0:
                    report += f"  - {col_info['column']}: {col_info['missing_percentage']:.1f}% missing\n"
            
            report += "\n"
        
        if s3_df is not None:
            s3_report = self.analyze_missing_data(s3_df, 'S3')
            report += "S3 STORAGE BUCKETS\n"
            report += "-" * 70 + "\n"
            report += f"Total Rows: {s3_report['total_rows']}\n"
            report += f"Total Columns: {s3_report['total_columns']}\n"
            report += f"Completely NULL Rows: {s3_report['completely_null_rows']} ({s3_report['completely_null_percentage']:.1f}%)\n"
            report += f"Data Quality Score: {s3_report['data_quality_score']:.1f}/100\n"
            report += "\n"
        
        report += "=" * 70 + "\n"
        
        return report
