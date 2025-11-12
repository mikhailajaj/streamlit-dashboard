"""
Budget Management and Alert System
Financial Impact: Prevent $12K-$24K/year in budget overruns

This module provides:
- Budget creation by scope (service, region, team, total)
- Real-time threshold monitoring (50%, 80%, 100%, forecasted)
- Alert management and notification system
- Budget variance reporting and burn rate tracking
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import os
import warnings
warnings.filterwarnings('ignore')


class BudgetManager:
    """Manages budgets and spending alerts"""
    
    # Budget storage file
    BUDGET_FILE = 'data/budgets.json'
    ALERT_FILE = 'data/budget_alerts.json'
    
    def __init__(self, budget_file: Optional[str] = None, alert_file: Optional[str] = None):
        self.budget_file = budget_file or self.BUDGET_FILE
        self.alert_file = alert_file or self.ALERT_FILE
        self.budgets = self._load_budgets()
        self.alert_history = self._load_alerts()
    
    def _load_budgets(self) -> List[Dict]:
        """Load budgets from JSON file"""
        if os.path.exists(self.budget_file):
            try:
                with open(self.budget_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def _save_budgets(self):
        """Save budgets to JSON file"""
        os.makedirs(os.path.dirname(self.budget_file), exist_ok=True)
        with open(self.budget_file, 'w') as f:
            json.dump(self.budgets, f, indent=2, default=str)
    
    def _load_alerts(self) -> List[Dict]:
        """Load alert history from JSON file"""
        if os.path.exists(self.alert_file):
            try:
                with open(self.alert_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def _save_alerts(self):
        """Save alert history to JSON file"""
        os.makedirs(os.path.dirname(self.alert_file), exist_ok=True)
        with open(self.alert_file, 'w') as f:
            json.dump(self.alert_history, f, indent=2, default=str)
    
    def create_budget(self, 
                     name: str,
                     scope_type: str,
                     scope_value: str,
                     amount: float,
                     period: str = 'monthly',
                     owner: str = '',
                     alert_thresholds: Optional[List[int]] = None,
                     start_date: Optional[datetime] = None) -> Dict:
        """
        Create a new budget
        
        Args:
            name: Budget name
            scope_type: 'total', 'service' (EC2/S3), 'region', 'team', 'environment'
            scope_value: Value for the scope (e.g., 'us-east-1' for region)
            amount: Budget amount in USD
            period: 'monthly', 'quarterly', 'annually'
            owner: Budget owner email/name
            alert_thresholds: List of percentage thresholds [50, 80, 100]
            start_date: Budget start date (defaults to current month)
            
        Returns:
            Created budget dictionary
        """
        if alert_thresholds is None:
            alert_thresholds = [50, 80, 100]
        
        if start_date is None:
            start_date = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        budget_id = f"budget_{len(self.budgets) + 1}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        budget = {
            'id': budget_id,
            'name': name,
            'scope_type': scope_type,
            'scope_value': scope_value,
            'amount': amount,
            'period': period,
            'owner': owner,
            'alert_thresholds': sorted(alert_thresholds),
            'start_date': start_date.isoformat(),
            'status': 'active',
            'created_at': datetime.now().isoformat(),
            'last_alerted_threshold': 0
        }
        
        self.budgets.append(budget)
        self._save_budgets()
        
        return budget
    
    def get_budget(self, budget_id: str) -> Optional[Dict]:
        """Get budget by ID"""
        for budget in self.budgets:
            if budget['id'] == budget_id:
                return budget
        return None
    
    def list_budgets(self, active_only: bool = True) -> List[Dict]:
        """List all budgets"""
        if active_only:
            return [b for b in self.budgets if b.get('status') == 'active']
        return self.budgets
    
    def update_budget(self, budget_id: str, updates: Dict) -> bool:
        """Update budget parameters"""
        for i, budget in enumerate(self.budgets):
            if budget['id'] == budget_id:
                self.budgets[i].update(updates)
                self.budgets[i]['updated_at'] = datetime.now().isoformat()
                self._save_budgets()
                return True
        return False
    
    def delete_budget(self, budget_id: str) -> bool:
        """Soft delete budget (set status to inactive)"""
        return self.update_budget(budget_id, {'status': 'inactive'})
    
    def calculate_actual_spend(self, 
                              budget: Dict,
                              ec2_df: pd.DataFrame,
                              s3_df: pd.DataFrame,
                              current_date: Optional[datetime] = None) -> Dict:
        """
        Calculate actual spend for a budget based on scope
        
        Returns:
            Dictionary with spend calculations
        """
        if current_date is None:
            current_date = datetime.now()
        
        scope_type = budget['scope_type']
        scope_value = budget['scope_value']
        
        # Filter data based on scope
        if scope_type == 'total':
            ec2_filtered = ec2_df
            s3_filtered = s3_df
        elif scope_type == 'service':
            if scope_value.upper() == 'EC2':
                ec2_filtered = ec2_df
                s3_filtered = pd.DataFrame()
            elif scope_value.upper() == 'S3':
                ec2_filtered = pd.DataFrame()
                s3_filtered = s3_df
            else:
                ec2_filtered = ec2_df
                s3_filtered = s3_df
        elif scope_type == 'region':
            ec2_filtered = ec2_df[ec2_df['Region'] == scope_value]
            s3_filtered = s3_df[s3_df['Region'] == scope_value]
        elif scope_type == 'team':
            # Parse tags to filter by team
            ec2_filtered = ec2_df[ec2_df['Tags'].str.contains(f'Owner={scope_value}', na=False)]
            s3_filtered = s3_df[s3_df['Tags'].str.contains(f'Owner={scope_value}', na=False)]
        elif scope_type == 'environment':
            ec2_filtered = ec2_df[ec2_df['Tags'].str.contains(f'Environment={scope_value}', na=False)]
            s3_filtered = s3_df[s3_df['Tags'].str.contains(f'Environment={scope_value}', na=False)]
        else:
            ec2_filtered = ec2_df
            s3_filtered = s3_df
        
        # Calculate monthly costs
        ec2_monthly = ec2_filtered['CostPerHourUSD'].sum() * 24 * 30 if len(ec2_filtered) > 0 else 0
        s3_monthly = s3_filtered['MonthlyCostUSD'].sum() if len(s3_filtered) > 0 else 0
        total_monthly = ec2_monthly + s3_monthly
        
        # Calculate period-specific costs
        period = budget['period']
        if period == 'monthly':
            period_multiplier = 1
            days_in_period = 30
        elif period == 'quarterly':
            period_multiplier = 3
            days_in_period = 90
        elif period == 'annually':
            period_multiplier = 12
            days_in_period = 365
        else:
            period_multiplier = 1
            days_in_period = 30
        
        total_period_cost = total_monthly * period_multiplier
        
        # Calculate days elapsed in current period
        start_date = datetime.fromisoformat(budget['start_date'])
        days_elapsed = (current_date - start_date).days % days_in_period
        days_remaining = days_in_period - days_elapsed
        
        # Pro-rate actual spend based on days elapsed
        if days_in_period > 0:
            actual_spend = total_period_cost * (days_elapsed / days_in_period)
        else:
            actual_spend = total_period_cost
        
        # Calculate burn rate
        if days_elapsed > 0:
            burn_rate_per_day = actual_spend / days_elapsed
        else:
            burn_rate_per_day = total_monthly / 30
        
        # Project end-of-period spend
        projected_spend = burn_rate_per_day * days_in_period
        
        return {
            'actual_spend': actual_spend,
            'projected_spend': projected_spend,
            'budget_amount': budget['amount'],
            'remaining_budget': budget['amount'] - actual_spend,
            'percent_used': (actual_spend / budget['amount'] * 100) if budget['amount'] > 0 else 0,
            'projected_percent': (projected_spend / budget['amount'] * 100) if budget['amount'] > 0 else 0,
            'burn_rate_per_day': burn_rate_per_day,
            'days_elapsed': days_elapsed,
            'days_remaining': days_remaining,
            'days_in_period': days_in_period,
            'is_over_budget': actual_spend > budget['amount'],
            'will_exceed_budget': projected_spend > budget['amount'],
            'ec2_monthly_cost': ec2_monthly,
            's3_monthly_cost': s3_monthly,
            'total_monthly_cost': total_monthly,
            'resource_counts': {
                'ec2_instances': len(ec2_filtered),
                's3_buckets': len(s3_filtered)
            }
        }
    
    def check_alerts(self, 
                    budget: Dict,
                    spend_data: Dict,
                    suppress_duplicates: bool = True) -> List[Dict]:
        """
        Check if any alert thresholds have been crossed
        
        Args:
            budget: Budget dictionary
            spend_data: Spend calculation from calculate_actual_spend()
            suppress_duplicates: Don't alert on same threshold twice
            
        Returns:
            List of triggered alerts
        """
        alerts = []
        current_threshold = int(spend_data['percent_used'])
        last_alerted = budget.get('last_alerted_threshold', 0)
        
        for threshold in budget['alert_thresholds']:
            # Check if threshold crossed
            if spend_data['percent_used'] >= threshold:
                # Suppress duplicate alerts
                if suppress_duplicates and threshold <= last_alerted:
                    continue
                
                alert = {
                    'budget_id': budget['id'],
                    'budget_name': budget['name'],
                    'threshold': threshold,
                    'percent_used': spend_data['percent_used'],
                    'actual_spend': spend_data['actual_spend'],
                    'budget_amount': budget['amount'],
                    'overage': spend_data['actual_spend'] - budget['amount'] if threshold >= 100 else 0,
                    'alert_type': 'threshold_exceeded',
                    'severity': self._get_alert_severity(threshold),
                    'timestamp': datetime.now().isoformat(),
                    'owner': budget.get('owner', 'N/A'),
                    'message': self._generate_alert_message(budget, spend_data, threshold)
                }
                
                alerts.append(alert)
                
                # Update last alerted threshold
                if threshold > last_alerted:
                    self.update_budget(budget['id'], {'last_alerted_threshold': threshold})
        
        # Check for forecasted overspend
        if spend_data['will_exceed_budget'] and not spend_data['is_over_budget']:
            if suppress_duplicates and 'forecast' in str(last_alerted):
                pass  # Already alerted on forecast
            else:
                alert = {
                    'budget_id': budget['id'],
                    'budget_name': budget['name'],
                    'threshold': 'forecast',
                    'percent_used': spend_data['percent_used'],
                    'projected_spend': spend_data['projected_spend'],
                    'budget_amount': budget['amount'],
                    'projected_overage': spend_data['projected_spend'] - budget['amount'],
                    'alert_type': 'forecast_exceed',
                    'severity': 'warning',
                    'timestamp': datetime.now().isoformat(),
                    'owner': budget.get('owner', 'N/A'),
                    'message': f"Budget '{budget['name']}' is projected to exceed by ${spend_data['projected_spend'] - budget['amount']:,.2f} based on current burn rate."
                }
                alerts.append(alert)
        
        # Save alerts to history
        if alerts:
            self.alert_history.extend(alerts)
            self._save_alerts()
        
        return alerts
    
    def _get_alert_severity(self, threshold: int) -> str:
        """Determine alert severity based on threshold"""
        if threshold >= 100:
            return 'critical'
        elif threshold >= 80:
            return 'high'
        elif threshold >= 50:
            return 'medium'
        else:
            return 'low'
    
    def _generate_alert_message(self, budget: Dict, spend_data: Dict, threshold: int) -> str:
        """Generate human-readable alert message"""
        if threshold >= 100:
            return f"🚨 CRITICAL: Budget '{budget['name']}' has exceeded {threshold}% (${spend_data['actual_spend']:,.2f} / ${budget['amount']:,.2f})"
        else:
            return f"⚠️ WARNING: Budget '{budget['name']}' has reached {threshold}% threshold (${spend_data['actual_spend']:,.2f} / ${budget['amount']:,.2f})"
    
    def get_budget_summary(self, 
                          ec2_df: pd.DataFrame,
                          s3_df: pd.DataFrame,
                          active_only: bool = True) -> List[Dict]:
        """
        Get summary of all budgets with current spend
        
        Returns:
            List of budget summaries
        """
        budgets = self.list_budgets(active_only=active_only)
        summaries = []
        
        for budget in budgets:
            spend_data = self.calculate_actual_spend(budget, ec2_df, s3_df)
            
            summary = {
                'id': budget['id'],
                'name': budget['name'],
                'scope': f"{budget['scope_type']}: {budget['scope_value']}",
                'budget_amount': budget['amount'],
                'actual_spend': spend_data['actual_spend'],
                'projected_spend': spend_data['projected_spend'],
                'percent_used': spend_data['percent_used'],
                'remaining': spend_data['remaining_budget'],
                'status': self._get_budget_status(spend_data),
                'burn_rate': spend_data['burn_rate_per_day'],
                'days_remaining': spend_data['days_remaining'],
                'owner': budget.get('owner', 'N/A'),
                'alerts_enabled': len(budget.get('alert_thresholds', [])) > 0
            }
            
            summaries.append(summary)
        
        return summaries
    
    def _get_budget_status(self, spend_data: Dict) -> str:
        """Determine budget status based on spend data"""
        if spend_data['is_over_budget']:
            return 'exceeded'
        elif spend_data['will_exceed_budget']:
            return 'at_risk'
        elif spend_data['percent_used'] >= 80:
            return 'warning'
        elif spend_data['percent_used'] >= 50:
            return 'caution'
        else:
            return 'healthy'
    
    def create_budget_templates(self) -> List[Dict]:
        """
        Create common budget templates
        
        Returns:
            List of budget template definitions
        """
        templates = [
            {
                'name': 'Total AWS Spend',
                'scope_type': 'total',
                'scope_value': 'all',
                'suggested_amount': 65000,  # Based on current $64,852/month
                'period': 'monthly',
                'alert_thresholds': [50, 80, 90, 100]
            },
            {
                'name': 'EC2 Compute Budget',
                'scope_type': 'service',
                'scope_value': 'EC2',
                'suggested_amount': 52000,  # Based on current $51,918/month
                'period': 'monthly',
                'alert_thresholds': [50, 80, 100]
            },
            {
                'name': 'S3 Storage Budget',
                'scope_type': 'service',
                'scope_value': 'S3',
                'suggested_amount': 13000,  # Based on current $12,934/month
                'period': 'monthly',
                'alert_thresholds': [50, 80, 100]
            },
            {
                'name': 'US-East-1 Regional Budget',
                'scope_type': 'region',
                'scope_value': 'us-east-1',
                'suggested_amount': 20000,
                'period': 'monthly',
                'alert_thresholds': [80, 100]
            },
            {
                'name': 'Production Environment',
                'scope_type': 'environment',
                'scope_value': 'Prod',
                'suggested_amount': 25000,
                'period': 'monthly',
                'alert_thresholds': [80, 90, 100]
            },
            {
                'name': 'Development Environment',
                'scope_type': 'environment',
                'scope_value': 'Dev',
                'suggested_amount': 15000,
                'period': 'monthly',
                'alert_thresholds': [50, 80, 100]
            }
        ]
        
        return templates


def test_budget_manager():
    """Test the budget manager"""
    # Create sample data
    ec2_data = {
        'InstanceId': ['i-001', 'i-002', 'i-003'],
        'Region': ['us-east-1', 'us-east-1', 'us-west-2'],
        'CostPerHourUSD': [0.17, 0.096, 0.126],
        'Tags': ['Owner=Alice,Environment=Dev', 'Owner=Bob,Environment=Prod', 'Owner=Alice,Environment=Dev']
    }
    
    s3_data = {
        'BucketName': ['bucket-1', 'bucket-2'],
        'Region': ['us-east-1', 'us-west-2'],
        'MonthlyCostUSD': [100.0, 150.0],
        'Tags': ['Owner=Alice,Environment=Dev', 'Owner=Bob,Environment=Prod']
    }
    
    ec2_df = pd.DataFrame(ec2_data)
    s3_df = pd.DataFrame(s3_data)
    
    # Initialize manager
    manager = BudgetManager(budget_file='data/test_budgets.json', alert_file='data/test_alerts.json')
    
    # Create test budget
    budget = manager.create_budget(
        name='Test EC2 Budget',
        scope_type='service',
        scope_value='EC2',
        amount=5000,
        period='monthly',
        owner='admin@company.com',
        alert_thresholds=[50, 80, 100]
    )
    
    print(f"Created budget: {budget['name']}")
    print(f"Budget ID: {budget['id']}")
    
    # Calculate spend
    spend_data = manager.calculate_actual_spend(budget, ec2_df, s3_df)
    print(f"\nSpend Data:")
    print(f"  Actual: ${spend_data['actual_spend']:,.2f}")
    print(f"  Projected: ${spend_data['projected_spend']:,.2f}")
    print(f"  Percent Used: {spend_data['percent_used']:.1f}%")
    print(f"  Burn Rate: ${spend_data['burn_rate_per_day']:,.2f}/day")
    
    # Check alerts
    alerts = manager.check_alerts(budget, spend_data)
    if alerts:
        print(f"\n⚠️  {len(alerts)} Alert(s) Triggered:")
        for alert in alerts:
            print(f"  - {alert['message']}")
    else:
        print("\n✅ No alerts triggered")
    
    # Get budget summary
    summaries = manager.get_budget_summary(ec2_df, s3_df)
    print(f"\nBudget Summary ({len(summaries)} budgets):")
    for summary in summaries:
        print(f"  {summary['name']}: ${summary['actual_spend']:,.2f} / ${summary['budget_amount']:,.2f} ({summary['status']})")


if __name__ == "__main__":
    test_budget_manager()
