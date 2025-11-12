"""
Tagging Strategy and Chargeback Module
Financial Impact: $94K-$140K/year through team accountability

This module provides:
- Tag compliance monitoring and reporting
- Mandatory tag schema enforcement
- Team-based cost allocation (showback/chargeback)
- Untagged resource identification
- Tag coverage trend tracking
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class TaggingPolicy:
    """Defines and enforces tagging policies"""
    
    # Mandatory tag schema
    MANDATORY_TAGS = {
        'Owner': {
            'description': 'Resource owner (email or team name)',
            'required': True,
            'validation': 'non_empty',
            'examples': ['Alice', 'Bob', 'DataTeam', 'alice@company.com']
        },
        'Environment': {
            'description': 'Environment type',
            'required': True,
            'validation': 'allowed_values',
            'allowed_values': ['Dev', 'Test', 'Staging', 'Prod', 'QA'],
            'examples': ['Dev', 'Prod', 'Test']
        },
        'Team': {
            'description': 'Business unit or team',
            'required': True,
            'validation': 'non_empty',
            'examples': ['Engineering', 'Data', 'Product', 'Security', 'Finance']
        },
        'CostCenter': {
            'description': 'Finance cost center or GL code',
            'required': True,
            'validation': 'non_empty',
            'examples': ['CC-1001', 'CC-2045', 'GL-ENG-001']
        },
        'Project': {
            'description': 'Project or application name',
            'required': True,
            'validation': 'non_empty',
            'examples': ['WebApp', 'DataPipeline', 'MLPlatform', 'CustomerPortal']
        }
    }
    
    # Optional but recommended tags
    OPTIONAL_TAGS = {
        'ExpirationDate': {
            'description': 'Resource expiration date',
            'validation': 'date_format',
            'examples': ['2025-12-31', '2026-06-30']
        },
        'BackupPolicy': {
            'description': 'Backup frequency',
            'validation': 'allowed_values',
            'allowed_values': ['daily', 'weekly', 'none'],
            'examples': ['daily', 'weekly']
        },
        'Compliance': {
            'description': 'Compliance requirements',
            'validation': 'allowed_values',
            'allowed_values': ['pci', 'hipaa', 'sox', 'none'],
            'examples': ['pci', 'hipaa']
        },
        'Application': {
            'description': 'Application name',
            'validation': 'non_empty',
            'examples': ['web-frontend', 'api-backend', 'data-processor']
        }
    }
    
    @classmethod
    def get_all_tags(cls) -> List[str]:
        """Get list of all tag names"""
        return list(cls.MANDATORY_TAGS.keys()) + list(cls.OPTIONAL_TAGS.keys())
    
    @classmethod
    def get_mandatory_tags(cls) -> List[str]:
        """Get list of mandatory tag names"""
        return list(cls.MANDATORY_TAGS.keys())
    
    @classmethod
    def validate_tag_value(cls, tag_name: str, tag_value: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a tag value against policy
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if tag is defined
        tag_policy = cls.MANDATORY_TAGS.get(tag_name) or cls.OPTIONAL_TAGS.get(tag_name)
        if not tag_policy:
            return True, None  # Unknown tags are allowed
        
        validation = tag_policy.get('validation')
        
        if validation == 'non_empty':
            if not tag_value or tag_value.strip() == '':
                return False, f"{tag_name} cannot be empty"
        
        elif validation == 'allowed_values':
            allowed = tag_policy.get('allowed_values', [])
            if tag_value not in allowed:
                return False, f"{tag_name} must be one of: {', '.join(allowed)}"
        
        elif validation == 'date_format':
            try:
                datetime.strptime(tag_value, '%Y-%m-%d')
            except:
                return False, f"{tag_name} must be in YYYY-MM-DD format"
        
        return True, None


class TagParser:
    """Parses and analyzes resource tags"""
    
    @staticmethod
    def parse_tags(tag_string: str) -> Dict[str, str]:
        """
        Parse tag string into dictionary
        
        Format: 'Key1=Value1,Key2=Value2'
        
        Returns:
            Dictionary of tags
        """
        if pd.isna(tag_string) or not tag_string:
            return {}
        
        tags = {}
        try:
            pairs = tag_string.split(',')
            for pair in pairs:
                if '=' in pair:
                    key, value = pair.split('=', 1)
                    tags[key.strip()] = value.strip()
        except Exception as e:
            return {}
        
        return tags
    
    @staticmethod
    def has_tag(tag_string: str, tag_name: str) -> bool:
        """Check if resource has specific tag"""
        tags = TagParser.parse_tags(tag_string)
        return tag_name in tags
    
    @staticmethod
    def get_tag_value(tag_string: str, tag_name: str, default: str = '') -> str:
        """Get value of specific tag"""
        tags = TagParser.parse_tags(tag_string)
        return tags.get(tag_name, default)


class TagComplianceAnalyzer:
    """Analyzes tag compliance across resources"""
    
    def __init__(self, policy: Optional[TaggingPolicy] = None):
        self.policy = policy or TaggingPolicy()
        self.parser = TagParser()
    
    def analyze_compliance(self, ec2_df: pd.DataFrame, s3_df: pd.DataFrame) -> Dict:
        """
        Analyze tag compliance across all resources
        
        Returns:
            Comprehensive compliance report
        """
        # Combine resources
        all_resources = self._combine_resources(ec2_df, s3_df)
        
        if len(all_resources) == 0:
            return {
                'overall_compliance': 0.0,
                'total_resources': 0,
                'compliant_resources': 0,
                'non_compliant_resources': 0
            }
        
        # Check compliance for each resource
        all_resources['parsed_tags'] = all_resources['Tags'].apply(self.parser.parse_tags)
        all_resources['is_compliant'] = all_resources['parsed_tags'].apply(
            lambda tags: self._check_resource_compliance(tags)
        )
        
        # Calculate compliance by tag
        tag_compliance = {}
        for tag_name in self.policy.get_mandatory_tags():
            has_tag = all_resources['parsed_tags'].apply(lambda tags: tag_name in tags)
            tag_compliance[tag_name] = {
                'present': has_tag.sum(),
                'missing': (~has_tag).sum(),
                'compliance_percentage': (has_tag.sum() / len(all_resources)) * 100
            }
        
        # Calculate compliance by service
        service_compliance = all_resources.groupby('ResourceType')['is_compliant'].agg([
            ('total', 'count'),
            ('compliant', 'sum')
        ]).reset_index()
        service_compliance['compliance_percentage'] = (
            service_compliance['compliant'] / service_compliance['total'] * 100
        )
        
        # Calculate compliance by region
        region_compliance = all_resources.groupby('Region')['is_compliant'].agg([
            ('total', 'count'),
            ('compliant', 'sum')
        ]).reset_index()
        region_compliance['compliance_percentage'] = (
            region_compliance['compliant'] / region_compliance['total'] * 100
        )
        
        # Identify non-compliant resources
        non_compliant = all_resources[~all_resources['is_compliant']].copy()
        non_compliant['missing_tags'] = non_compliant['parsed_tags'].apply(
            lambda tags: self._get_missing_tags(tags)
        )
        
        # Calculate overall metrics
        total_resources = len(all_resources)
        compliant_resources = all_resources['is_compliant'].sum()
        compliance_percentage = (compliant_resources / total_resources * 100) if total_resources > 0 else 0
        
        return {
            'overall_compliance': compliance_percentage,
            'total_resources': total_resources,
            'compliant_resources': int(compliant_resources),
            'non_compliant_resources': total_resources - int(compliant_resources),
            'target_compliance': 95.0,
            'compliance_gap': max(0, 95.0 - compliance_percentage),
            'tag_compliance': tag_compliance,
            'service_compliance': service_compliance.to_dict('records'),
            'region_compliance': region_compliance.to_dict('records'),
            'non_compliant_resources': non_compliant[
                ['ResourceId', 'ResourceType', 'Region', 'missing_tags']
            ].to_dict('records')[:50]  # Limit to 50 for display
        }
    
    def _combine_resources(self, ec2_df: pd.DataFrame, s3_df: pd.DataFrame) -> pd.DataFrame:
        """Combine EC2 and S3 resources into unified dataframe"""
        resources = []
        
        # Add EC2 instances
        if len(ec2_df) > 0:
            ec2_resources = ec2_df[['InstanceId', 'Region', 'Tags']].copy()
            ec2_resources.columns = ['ResourceId', 'Region', 'Tags']
            ec2_resources['ResourceType'] = 'EC2'
            resources.append(ec2_resources)
        
        # Add S3 buckets
        if len(s3_df) > 0:
            s3_resources = s3_df[['BucketName', 'Region', 'Tags']].copy()
            s3_resources.columns = ['ResourceId', 'Region', 'Tags']
            s3_resources['ResourceType'] = 'S3'
            resources.append(s3_resources)
        
        if resources:
            return pd.concat(resources, ignore_index=True)
        return pd.DataFrame(columns=['ResourceId', 'Region', 'Tags', 'ResourceType'])
    
    def _check_resource_compliance(self, tags: Dict[str, str]) -> bool:
        """Check if resource has all mandatory tags"""
        for tag_name in self.policy.get_mandatory_tags():
            if tag_name not in tags:
                return False
        return True
    
    def _get_missing_tags(self, tags: Dict[str, str]) -> List[str]:
        """Get list of missing mandatory tags"""
        missing = []
        for tag_name in self.policy.get_mandatory_tags():
            if tag_name not in tags:
                missing.append(tag_name)
        return missing
    
    def generate_compliance_report(self, ec2_df: pd.DataFrame, s3_df: pd.DataFrame) -> str:
        """Generate human-readable compliance report"""
        analysis = self.analyze_compliance(ec2_df, s3_df)
        
        report = f"""
TAG COMPLIANCE REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}

OVERALL COMPLIANCE
------------------
Compliance Rate: {analysis['overall_compliance']:.1f}%
Target: {analysis['target_compliance']:.1f}%
Gap: {analysis['compliance_gap']:.1f}%

Resources:
  Total: {analysis['total_resources']}
  Compliant: {analysis['compliant_resources']}
  Non-Compliant: {analysis['non_compliant_resources']}

MANDATORY TAG COMPLIANCE
-------------------------"""
        
        for tag_name, data in analysis['tag_compliance'].items():
            report += f"\n{tag_name}:"
            report += f"\n  Present: {data['present']} ({data['compliance_percentage']:.1f}%)"
            report += f"\n  Missing: {data['missing']}"
        
        report += f"\n\nCOMPLIANCE BY SERVICE"
        report += f"\n---------------------"
        for service in analysis['service_compliance']:
            report += f"\n{service['ResourceType']}: {service['compliance_percentage']:.1f}% ({service['compliant']}/{service['total']})"
        
        report += f"\n\nTOP NON-COMPLIANT RESOURCES"
        report += f"\n---------------------------"
        for i, resource in enumerate(analysis['non_compliant_resources'][:10], 1):
            report += f"\n{i}. {resource['ResourceId']} ({resource['ResourceType']})"
            report += f"\n   Missing: {', '.join(resource['missing_tags'])}"
        
        return report


class ChargebackEngine:
    """Allocates costs to teams and generates chargeback reports"""
    
    def __init__(self):
        self.parser = TagParser()
    
    def allocate_costs(self, ec2_df: pd.DataFrame, s3_df: pd.DataFrame,
                      allocation_tag: str = 'Team',
                      period: str = 'monthly') -> pd.DataFrame:
        """
        Allocate costs by specified tag (Team, CostCenter, Project, etc.)
        
        Returns:
            DataFrame with cost allocation by tag value
        """
        # Parse tags and extract allocation dimension
        ec2_allocated = ec2_df.copy()
        s3_allocated = s3_df.copy()
        
        ec2_allocated['AllocatedTo'] = ec2_allocated['Tags'].apply(
            lambda x: self.parser.get_tag_value(x, allocation_tag, 'Untagged')
        )
        s3_allocated['AllocatedTo'] = s3_allocated['Tags'].apply(
            lambda x: self.parser.get_tag_value(x, allocation_tag, 'Untagged')
        )
        
        # Calculate monthly costs for EC2 (handle both CostPerHourUSD and CostUSD columns)
        if 'CostPerHourUSD' in ec2_allocated.columns:
            ec2_allocated['MonthlyCost'] = ec2_allocated['CostPerHourUSD'] * 24 * 30
        elif 'CostUSD' in ec2_allocated.columns:
            # CostUSD is already monthly or needs conversion
            ec2_allocated['MonthlyCost'] = ec2_allocated['CostUSD']
        else:
            ec2_allocated['MonthlyCost'] = 0
        
        # Determine ID column (could be InstanceId or ResourceId)
        ec2_id_col = 'InstanceId' if 'InstanceId' in ec2_allocated.columns else 'ResourceId'
        
        # Aggregate by allocation dimension for EC2
        ec2_costs = ec2_allocated.groupby('AllocatedTo').agg({
            ec2_id_col: 'count',
            'MonthlyCost': 'sum',
            'CPUUtilization': 'mean'
        }).reset_index()
        ec2_costs.columns = ['AllocatedTo', 'EC2_Count', 'EC2_MonthlyCost', 'EC2_AvgCPU']
        
        # Calculate S3 monthly costs (handle both MonthlyCostUSD and CostUSD columns)
        if 'MonthlyCostUSD' in s3_allocated.columns:
            s3_cost_col = 'MonthlyCostUSD'
        elif 'CostUSD' in s3_allocated.columns:
            s3_cost_col = 'CostUSD'
        else:
            s3_allocated['CostUSD'] = 0
            s3_cost_col = 'CostUSD'
        
        # Aggregate by allocation dimension for S3
        s3_costs = s3_allocated.groupby('AllocatedTo').agg({
            'BucketName': 'count',
            s3_cost_col: 'sum',
            'TotalSizeGB': 'sum'
        }).reset_index()
        s3_costs.columns = ['AllocatedTo', 'S3_Count', 'S3_MonthlyCost', 'S3_TotalGB']
        
        # Merge EC2 and S3 costs
        allocation = pd.merge(ec2_costs, s3_costs, on='AllocatedTo', how='outer').fillna(0)
        
        # Calculate totals
        allocation['TotalMonthlyCost'] = allocation['EC2_MonthlyCost'] + allocation['S3_MonthlyCost']
        allocation['TotalResourceCount'] = allocation['EC2_Count'] + allocation['S3_Count']
        
        # Calculate percentages
        total_cost = allocation['TotalMonthlyCost'].sum()
        allocation['PercentageOfTotal'] = (allocation['TotalMonthlyCost'] / total_cost * 100) if total_cost > 0 else 0
        
        # Sort by cost descending
        allocation = allocation.sort_values('TotalMonthlyCost', ascending=False)
        
        return allocation
    
    def generate_chargeback_report(self, ec2_df: pd.DataFrame, s3_df: pd.DataFrame,
                                   allocation_tag: str = 'Team',
                                   report_period: Optional[str] = None) -> Dict:
        """
        Generate comprehensive chargeback report
        
        Returns:
            Dictionary containing report data and summary
        """
        if report_period is None:
            report_period = datetime.now().strftime('%Y-%m')
        
        # Get cost allocation
        allocation = self.allocate_costs(ec2_df, s3_df, allocation_tag)
        
        # Calculate summary metrics
        total_cost = allocation['TotalMonthlyCost'].sum()
        untagged_cost = allocation[allocation['AllocatedTo'] == 'Untagged']['TotalMonthlyCost'].sum()
        tagged_cost = total_cost - untagged_cost
        allocation_coverage = (tagged_cost / total_cost * 100) if total_cost > 0 else 0
        
        # Identify top cost centers
        top_5 = allocation.head(5).to_dict('records')
        
        # Calculate cost per resource
        allocation['CostPerResource'] = allocation['TotalMonthlyCost'] / allocation['TotalResourceCount']
        allocation['CostPerResource'] = allocation['CostPerResource'].replace([np.inf, -np.inf], 0)
        
        report = {
            'period': report_period,
            'allocation_dimension': allocation_tag,
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'total_monthly_cost': total_cost,
                'tagged_cost': tagged_cost,
                'untagged_cost': untagged_cost,
                'allocation_coverage_percentage': allocation_coverage,
                'total_teams': len(allocation),
                'teams_with_costs': len(allocation[allocation['TotalMonthlyCost'] > 0])
            },
            'top_cost_centers': top_5,
            'detailed_allocation': allocation.to_dict('records'),
            'recommendations': self._generate_chargeback_recommendations(allocation, allocation_coverage)
        }
        
        return report
    
    def _generate_chargeback_recommendations(self, allocation: pd.DataFrame, 
                                            coverage: float) -> List[str]:
        """Generate recommendations based on allocation analysis"""
        recommendations = []
        
        # Check allocation coverage
        if coverage < 90:
            untagged = allocation[allocation['AllocatedTo'] == 'Untagged']['TotalMonthlyCost'].sum()
            recommendations.append(
                f"Improve tag coverage: ${untagged:,.2f}/month ({100-coverage:.1f}%) is unallocated. "
                f"Tag these resources to enable proper chargeback."
            )
        
        # Check for dominant cost centers
        if len(allocation) > 0:
            top_team = allocation.iloc[0]
            if top_team['PercentageOfTotal'] > 50:
                recommendations.append(
                    f"{top_team['AllocatedTo']} accounts for {top_team['PercentageOfTotal']:.1f}% of costs. "
                    f"Consider breaking down into sub-teams or projects for better visibility."
                )
        
        # Check for teams with high cost per resource
        if 'CostPerResource' in allocation.columns:
            high_cost_teams = allocation[allocation['CostPerResource'] > allocation['CostPerResource'].median() * 2]
            if len(high_cost_teams) > 0:
                for _, team in high_cost_teams.head(3).iterrows():
                    if team['AllocatedTo'] != 'Untagged':
                        recommendations.append(
                            f"{team['AllocatedTo']} has high cost per resource (${team['CostPerResource']:,.2f}). "
                            f"Review resource efficiency and right-sizing opportunities."
                        )
        
        return recommendations
    
    def export_chargeback_csv(self, allocation: pd.DataFrame, filepath: str):
        """Export chargeback report to CSV for finance teams"""
        export_df = allocation[[
            'AllocatedTo', 'TotalMonthlyCost', 'EC2_MonthlyCost', 'S3_MonthlyCost',
            'EC2_Count', 'S3_Count', 'TotalResourceCount', 'PercentageOfTotal'
        ]].copy()
        
        export_df.columns = [
            'Team/CostCenter', 'Total Monthly Cost (USD)', 'EC2 Cost (USD)', 'S3 Cost (USD)',
            'EC2 Instances', 'S3 Buckets', 'Total Resources', 'Percentage of Total (%)'
        ]
        
        export_df.to_csv(filepath, index=False)
        return filepath


def test_tagging_chargeback():
    """Test the tagging and chargeback engine"""
    # Create sample data with realistic tags
    ec2_data = {
        'InstanceId': ['i-001', 'i-002', 'i-003', 'i-004', 'i-005'],
        'Region': ['us-east-1', 'us-east-1', 'us-west-2', 'us-west-2', 'eu-west-1'],
        'CostPerHourUSD': [0.17, 0.096, 0.126, 0.21, 0.15],
        'CPUUtilization': [75, 65, 80, 45, 90],
        'Tags': [
            'Owner=Alice,Environment=Dev,Team=Engineering,CostCenter=CC-1001,Project=WebApp',
            'Owner=Bob,Environment=Prod,Team=DataScience,CostCenter=CC-2001,Project=MLPlatform',
            'Owner=Charlie,Environment=Test',  # Missing tags
            'Owner=Alice,Environment=Dev,Team=Engineering,CostCenter=CC-1001,Project=API',
            ''  # No tags
        ]
    }
    
    s3_data = {
        'BucketName': ['bucket-1', 'bucket-2', 'bucket-3'],
        'Region': ['us-east-1', 'us-west-2', 'us-east-1'],
        'MonthlyCostUSD': [100.0, 150.0, 200.0],
        'TotalSizeGB': [500, 750, 1000],
        'Tags': [
            'Owner=Bob,Environment=Prod,Team=DataScience,CostCenter=CC-2001,Project=DataLake',
            'Owner=Alice,Environment=Dev,Team=Engineering,CostCenter=CC-1001,Project=Storage',
            'Owner=David'  # Missing tags
        ]
    }
    
    ec2_df = pd.DataFrame(ec2_data)
    s3_df = pd.DataFrame(s3_data)
    
    # Test compliance analysis
    print("=" * 80)
    print("TAG COMPLIANCE ANALYSIS")
    print("=" * 80)
    
    analyzer = TagComplianceAnalyzer()
    compliance = analyzer.analyze_compliance(ec2_df, s3_df)
    
    print(f"\nOverall Compliance: {compliance['overall_compliance']:.1f}%")
    print(f"Compliant Resources: {compliance['compliant_resources']}/{compliance['total_resources']}")
    print(f"Target: {compliance['target_compliance']:.1f}%")
    print(f"Gap: {compliance['compliance_gap']:.1f}%")
    
    print("\nTag-by-Tag Compliance:")
    for tag_name, data in compliance['tag_compliance'].items():
        print(f"  {tag_name}: {data['compliance_percentage']:.1f}% ({data['present']}/{data['present'] + data['missing']})")
    
    # Test chargeback
    print("\n" + "=" * 80)
    print("CHARGEBACK REPORT")
    print("=" * 80)
    
    chargeback = ChargebackEngine()
    report = chargeback.generate_chargeback_report(ec2_df, s3_df, allocation_tag='Team')
    
    print(f"\nPeriod: {report['period']}")
    print(f"Allocation Dimension: {report['allocation_dimension']}")
    print(f"\nSummary:")
    print(f"  Total Monthly Cost: ${report['summary']['total_monthly_cost']:,.2f}")
    print(f"  Tagged Cost: ${report['summary']['tagged_cost']:,.2f}")
    print(f"  Untagged Cost: ${report['summary']['untagged_cost']:,.2f}")
    print(f"  Allocation Coverage: {report['summary']['allocation_coverage_percentage']:.1f}%")
    
    print(f"\nTop Cost Centers:")
    for i, center in enumerate(report['top_cost_centers'], 1):
        print(f"  {i}. {center['AllocatedTo']}: ${center['TotalMonthlyCost']:,.2f} ({center['PercentageOfTotal']:.1f}%)")
    
    if report['recommendations']:
        print(f"\nRecommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")


if __name__ == "__main__":
    test_tagging_chargeback()
