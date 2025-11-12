"""
Scenario-Based Cost Modeling for AWS FinOps
Replaces time series forecasting on snapshot data with actionable scenario projections
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class AWSCostScenarioModeler:
    """
    Scenario-based cost modeling for AWS resources.
    
    CRITICAL FIX: Replaces Prophet/ARIMA time series forecasting which was meaningless
    on snapshot data. Instead, provides actionable scenario projections based on:
    - Current cost baseline
    - Growth assumptions
    - Optimization opportunities
    - Business scenarios
    """
    
    def __init__(self):
        self.baseline_costs = {}
        self.optimization_potential = {}
        self.scenarios = {}
        self.is_fitted = False
        
    def analyze_baseline(self, ec2_df: pd.DataFrame, s3_df: pd.DataFrame) -> Dict:
        """
        Analyze current cost baseline from snapshot data.
        
        Args:
            ec2_df: EC2 instances with cost data
            s3_df: S3 buckets with cost data
            
        Returns:
            Dictionary with baseline cost metrics
        """
        # Clean data first
        ec2_clean = ec2_df.dropna(subset=['CostUSD'])
        s3_clean = s3_df.dropna(subset=['MonthlyCostUSD'])
        
        # Calculate baseline costs
        self.baseline_costs = {
            'ec2_monthly': ec2_clean['CostUSD'].sum() if len(ec2_clean) > 0 else 0,
            's3_monthly': s3_clean['MonthlyCostUSD'].sum() if len(s3_clean) > 0 else 0,
            'total_monthly': 0,
            'ec2_count': len(ec2_clean),
            's3_count': len(s3_clean),
            'breakdown_by_region': {},
            'breakdown_by_type': {}
        }
        
        self.baseline_costs['total_monthly'] = (
            self.baseline_costs['ec2_monthly'] + 
            self.baseline_costs['s3_monthly']
        )
        
        # Regional breakdown
        if len(ec2_clean) > 0:
            ec2_by_region = ec2_clean.groupby('Region')['CostUSD'].sum().to_dict()
        else:
            ec2_by_region = {}
            
        if len(s3_clean) > 0:
            s3_by_region = s3_clean.groupby('Region')['MonthlyCostUSD'].sum().to_dict()
        else:
            s3_by_region = {}
        
        # Combine regional costs
        all_regions = set(list(ec2_by_region.keys()) + list(s3_by_region.keys()))
        for region in all_regions:
            self.baseline_costs['breakdown_by_region'][region] = (
                ec2_by_region.get(region, 0) + s3_by_region.get(region, 0)
            )
        
        # Type breakdown
        if len(ec2_clean) > 0:
            self.baseline_costs['breakdown_by_type']['ec2'] = ec2_clean.groupby(
                'InstanceType'
            )['CostUSD'].sum().to_dict()
        else:
            self.baseline_costs['breakdown_by_type']['ec2'] = {}
        
        return self.baseline_costs
    
    def identify_optimization_opportunities(self, ec2_df: pd.DataFrame, s3_df: pd.DataFrame) -> Dict:
        """
        Identify cost optimization opportunities from current snapshot.
        
        Returns:
            Dictionary with optimization potential by category
        """
        ec2_clean = ec2_df.dropna(subset=['CostUSD', 'CPUUtilization'])
        s3_clean = s3_df.dropna(subset=['MonthlyCostUSD'])
        
        opportunities = {
            'idle_instances': {'count': 0, 'monthly_waste': 0},
            'underutilized_instances': {'count': 0, 'potential_savings': 0},
            'stopped_instances': {'count': 0, 'potential_savings': 0},
            's3_lifecycle_candidates': {'count': 0, 'potential_savings': 0},
            'total_potential_monthly_savings': 0
        }
        
        if len(ec2_clean) > 0:
            # Idle instances (CPU < 5%)
            idle_mask = ec2_clean['CPUUtilization'] < 5
            idle_instances = ec2_clean[idle_mask]
            opportunities['idle_instances']['count'] = len(idle_instances)
            opportunities['idle_instances']['monthly_waste'] = idle_instances['CostUSD'].sum()
            
            # Underutilized instances (CPU < 25%)
            underutil_mask = (ec2_clean['CPUUtilization'] >= 5) & (ec2_clean['CPUUtilization'] < 25)
            underutil_instances = ec2_clean[underutil_mask]
            opportunities['underutilized_instances']['count'] = len(underutil_instances)
            # Assume 30% savings by downsizing
            opportunities['underutilized_instances']['potential_savings'] = (
                underutil_instances['CostUSD'].sum() * 0.30
            )
            
            # Stopped instances still incurring storage costs
            stopped_mask = ec2_clean['State'] == 'stopped'
            stopped_instances = ec2_clean[stopped_mask]
            opportunities['stopped_instances']['count'] = len(stopped_instances)
            # Assume 80% savings by removing unused stopped instances
            opportunities['stopped_instances']['potential_savings'] = (
                stopped_instances['CostUSD'].sum() * 0.80
            )
        
        if len(s3_clean) > 0:
            # S3 buckets using STANDARD storage class (candidates for lifecycle policies)
            if 'StorageClass' in s3_clean.columns:
                standard_buckets = s3_clean[s3_clean['StorageClass'] == 'STANDARD']
                # Filter for buckets with significant cost
                lifecycle_candidates = standard_buckets[
                    standard_buckets['MonthlyCostUSD'] > 10
                ]
                opportunities['s3_lifecycle_candidates']['count'] = len(lifecycle_candidates)
                # Assume 40% savings by moving to IA/Glacier
                opportunities['s3_lifecycle_candidates']['potential_savings'] = (
                    lifecycle_candidates['MonthlyCostUSD'].sum() * 0.40
                )
        
        # Calculate total potential savings
        opportunities['total_potential_monthly_savings'] = (
            opportunities['idle_instances']['monthly_waste'] +
            opportunities['underutilized_instances']['potential_savings'] +
            opportunities['stopped_instances']['potential_savings'] +
            opportunities['s3_lifecycle_candidates']['potential_savings']
        )
        
        self.optimization_potential = opportunities
        return opportunities
    
    def generate_scenarios(self, months: int = 12) -> Dict[str, pd.DataFrame]:
        """
        Generate cost projection scenarios based on baseline and optimization potential.
        
        Scenarios:
        1. Baseline: Current trajectory (no changes, modest growth)
        2. Conservative: 5% annual growth, 10% optimization achieved
        3. Aggressive Growth: 15% annual growth, 25% optimization achieved
        4. Optimized: 5% annual growth, 40% optimization achieved
        
        Args:
            months: Number of months to project (default 12)
            
        Returns:
            Dictionary with scenario DataFrames
        """
        if not self.baseline_costs:
            raise ValueError("Must call analyze_baseline() first")
        
        baseline_monthly = self.baseline_costs['total_monthly']
        max_savings = self.optimization_potential.get('total_potential_monthly_savings', 0)
        
        # Generate monthly projections for each scenario
        month_range = np.arange(0, months)
        
        # Scenario 1: Baseline (no optimization, 2% monthly growth)
        baseline_scenario = self._project_costs(
            baseline_monthly, 
            growth_rate=0.02 / 12,  # 2% annual = 0.167% monthly
            optimization_savings=0,
            months=months
        )
        
        # Scenario 2: Conservative (5% annual growth, 10% optimization)
        conservative_scenario = self._project_costs(
            baseline_monthly,
            growth_rate=0.05 / 12,  # 5% annual
            optimization_savings=max_savings * 0.10,
            months=months
        )
        
        # Scenario 3: Aggressive Growth (15% annual growth, 25% optimization)
        aggressive_scenario = self._project_costs(
            baseline_monthly,
            growth_rate=0.15 / 12,  # 15% annual
            optimization_savings=max_savings * 0.25,
            months=months
        )
        
        # Scenario 4: Optimized (5% annual growth, 40% optimization)
        optimized_scenario = self._project_costs(
            baseline_monthly,
            growth_rate=0.05 / 12,
            optimization_savings=max_savings * 0.40,
            months=months
        )
        
        self.scenarios = {
            'baseline': baseline_scenario,
            'conservative': conservative_scenario,
            'aggressive_growth': aggressive_scenario,
            'optimized': optimized_scenario
        }
        
        self.is_fitted = True
        return self.scenarios
    
    def _project_costs(self, initial_cost: float, growth_rate: float, 
                      optimization_savings: float, months: int) -> pd.DataFrame:
        """
        Project costs over time with growth and optimization.
        
        Formula: Cost(t) = (InitialCost - OptimizationSavings) * (1 + growth_rate)^t
        """
        month_indices = np.arange(0, months)
        
        # Apply one-time optimization savings at month 0
        base_cost_after_optimization = initial_cost - optimization_savings
        
        # Apply growth over time
        monthly_costs = base_cost_after_optimization * np.power(1 + growth_rate, month_indices)
        
        # Create DataFrame
        projection = pd.DataFrame({
            'month': month_indices,
            'monthly_cost': monthly_costs,
            'cumulative_cost': np.cumsum(monthly_costs),
            'savings_vs_baseline': 0  # Will calculate later
        })
        
        return projection
    
    def calculate_scenario_comparisons(self) -> pd.DataFrame:
        """
        Compare all scenarios and calculate savings vs baseline.
        
        Returns:
            DataFrame with scenario comparison metrics
        """
        if not self.is_fitted:
            raise ValueError("Must call generate_scenarios() first")
        
        baseline = self.scenarios['baseline']
        
        comparisons = []
        for scenario_name, scenario_data in self.scenarios.items():
            if scenario_name == 'baseline':
                continue
            
            savings_vs_baseline = (
                baseline['cumulative_cost'].iloc[-1] - 
                scenario_data['cumulative_cost'].iloc[-1]
            )
            
            avg_monthly_cost = scenario_data['monthly_cost'].mean()
            total_12_month_cost = scenario_data['cumulative_cost'].iloc[-1]
            
            comparisons.append({
                'scenario': scenario_name,
                'avg_monthly_cost': avg_monthly_cost,
                'total_12_month_cost': total_12_month_cost,
                'savings_vs_baseline_12m': savings_vs_baseline,
                'savings_percentage': (savings_vs_baseline / baseline['cumulative_cost'].iloc[-1]) * 100
            })
        
        return pd.DataFrame(comparisons)
    
    def plot_scenario_comparison(self) -> go.Figure:
        """
        Create interactive visualization comparing all scenarios.
        
        Returns:
            Plotly figure with scenario comparison
        """
        if not self.is_fitted:
            raise ValueError("Must call generate_scenarios() first")
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Monthly Cost Projections', 'Cumulative Cost Comparison'),
            vertical_spacing=0.15,
            row_heights=[0.5, 0.5]
        )
        
        colors = {
            'baseline': '#d62728',  # Red
            'conservative': '#ff7f0e',  # Orange
            'aggressive_growth': '#9467bd',  # Purple
            'optimized': '#2ca02c'  # Green
        }
        
        scenario_labels = {
            'baseline': 'Baseline (No Optimization)',
            'conservative': 'Conservative (10% Opt.)',
            'aggressive_growth': 'Aggressive Growth (25% Opt.)',
            'optimized': 'Optimized (40% Opt.)'
        }
        
        # Plot monthly costs
        for scenario_name, scenario_data in self.scenarios.items():
            fig.add_trace(
                go.Scatter(
                    x=scenario_data['month'],
                    y=scenario_data['monthly_cost'],
                    mode='lines',
                    name=scenario_labels.get(scenario_name, scenario_name),
                    line=dict(color=colors.get(scenario_name, '#1f77b4'), width=2),
                    legendgroup=scenario_name
                ),
                row=1, col=1
            )
        
        # Plot cumulative costs
        for scenario_name, scenario_data in self.scenarios.items():
            fig.add_trace(
                go.Scatter(
                    x=scenario_data['month'],
                    y=scenario_data['cumulative_cost'],
                    mode='lines',
                    name=scenario_labels.get(scenario_name, scenario_name),
                    line=dict(color=colors.get(scenario_name, '#1f77b4'), width=2),
                    showlegend=False,
                    legendgroup=scenario_name
                ),
                row=2, col=1
            )
        
        # Update axes
        fig.update_xaxes(title_text="Month", row=1, col=1)
        fig.update_xaxes(title_text="Month", row=2, col=1)
        fig.update_yaxes(title_text="Monthly Cost (USD)", row=1, col=1)
        fig.update_yaxes(title_text="Cumulative Cost (USD)", row=2, col=1)
        
        fig.update_layout(
            title={
                'text': 'AWS Cost Scenario Analysis - 12 Month Projection',
                'x': 0.5,
                'xanchor': 'center'
            },
            hovermode='x unified',
            height=800,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def get_summary_metrics(self) -> Dict:
        """
        Get summary metrics for all scenarios.
        
        Returns:
            Dictionary with key metrics
        """
        if not self.is_fitted:
            raise ValueError("Must call generate_scenarios() first")
        
        baseline = self.scenarios['baseline']
        optimized = self.scenarios['optimized']
        
        summary = {
            'current_monthly_cost': self.baseline_costs['total_monthly'],
            'baseline_12m_total': baseline['cumulative_cost'].iloc[-1],
            'optimized_12m_total': optimized['cumulative_cost'].iloc[-1],
            'potential_12m_savings': baseline['cumulative_cost'].iloc[-1] - optimized['cumulative_cost'].iloc[-1],
            'monthly_optimization_potential': self.optimization_potential.get('total_potential_monthly_savings', 0),
            'roi_percentage': 0
        }
        
        # Calculate ROI
        if summary['baseline_12m_total'] > 0:
            summary['roi_percentage'] = (
                summary['potential_12m_savings'] / summary['baseline_12m_total']
            ) * 100
        
        return summary
    
    def generate_recommendations(self) -> List[Dict]:
        """
        Generate actionable recommendations based on scenario analysis.
        
        Returns:
            List of recommendation dictionaries
        """
        if not self.optimization_potential:
            raise ValueError("Must call identify_optimization_opportunities() first")
        
        recommendations = []
        
        # Idle instances
        if self.optimization_potential['idle_instances']['count'] > 0:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'EC2 - Idle Resources',
                'issue': f"{self.optimization_potential['idle_instances']['count']} instances with <5% CPU utilization",
                'action': 'Terminate or stop idle instances',
                'potential_monthly_savings': self.optimization_potential['idle_instances']['monthly_waste'],
                'effort': 'Low',
                'impact': 'High'
            })
        
        # Underutilized instances
        if self.optimization_potential['underutilized_instances']['count'] > 0:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'EC2 - Right-Sizing',
                'issue': f"{self.optimization_potential['underutilized_instances']['count']} instances with <25% CPU utilization",
                'action': 'Downsize to smaller instance types',
                'potential_monthly_savings': self.optimization_potential['underutilized_instances']['potential_savings'],
                'effort': 'Medium',
                'impact': 'Medium'
            })
        
        # Stopped instances
        if self.optimization_potential['stopped_instances']['count'] > 0:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'EC2 - Stopped Instances',
                'issue': f"{self.optimization_potential['stopped_instances']['count']} stopped instances incurring storage costs",
                'action': 'Terminate unused stopped instances or create AMIs',
                'potential_monthly_savings': self.optimization_potential['stopped_instances']['potential_savings'],
                'effort': 'Low',
                'impact': 'Medium'
            })
        
        # S3 lifecycle policies
        if self.optimization_potential['s3_lifecycle_candidates']['count'] > 0:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'S3 - Storage Class',
                'issue': f"{self.optimization_potential['s3_lifecycle_candidates']['count']} buckets using STANDARD storage",
                'action': 'Implement lifecycle policies to move to IA/Glacier',
                'potential_monthly_savings': self.optimization_potential['s3_lifecycle_candidates']['potential_savings'],
                'effort': 'Low',
                'impact': 'High'
            })
        
        # Sort by potential savings
        recommendations.sort(key=lambda x: x['potential_monthly_savings'], reverse=True)
        
        return recommendations
