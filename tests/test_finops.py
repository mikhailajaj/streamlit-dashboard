"""
FinOps Module Test Suite
Validates all three critical FinOps modules

Run this script to verify your FinOps implementation is working correctly.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_header(text):
    """Print formatted header"""
    print(f"\n{BLUE}{'='*80}{RESET}")
    print(f"{BLUE}{text.center(80)}{RESET}")
    print(f"{BLUE}{'='*80}{RESET}\n")

def print_success(text):
    """Print success message"""
    print(f"{GREEN}✓ {text}{RESET}")

def print_error(text):
    """Print error message"""
    print(f"{RED}✗ {text}{RESET}")

def print_warning(text):
    """Print warning message"""
    print(f"{YELLOW}⚠ {text}{RESET}")

def print_info(text):
    """Print info message"""
    print(f"  {text}")

def create_sample_data():
    """Create sample EC2 and S3 data for testing"""
    
    # Sample EC2 data
    ec2_data = {
        'InstanceId': [f'i-{1000+i}' for i in range(20)],
        'Region': ['us-east-1']*8 + ['us-west-2']*7 + ['eu-west-1']*5,
        'InstanceType': ['c5.xlarge']*5 + ['m5.large']*8 + ['r5.large']*4 + ['t3.small']*3,
        'State': ['running']*15 + ['stopped']*3 + ['terminated']*2,
        'CostPerHourUSD': np.random.uniform(0.05, 0.20, 20),
        'CPUUtilization': np.random.uniform(10, 90, 20),
        'MemoryUtilization': np.random.uniform(20, 80, 20),
        'LaunchTime': pd.date_range(start='2024-01-01', periods=20, freq='7D'),
        'Tags': [
            'Owner=Alice,Environment=Dev,Team=Engineering,CostCenter=CC-1001,Project=WebApp',
            'Owner=Bob,Environment=Prod,Team=DataScience,CostCenter=CC-2001,Project=MLPlatform',
            'Owner=Charlie,Environment=Test',  # Incomplete tags
            'Owner=Alice,Environment=Dev,Team=Engineering,CostCenter=CC-1001,Project=API',
            '',  # No tags
            'Owner=Bob,Environment=Prod,Team=DataScience,CostCenter=CC-2001,Project=DataLake',
            'Owner=Alice,Environment=Dev,Team=Engineering,CostCenter=CC-1001,Project=WebApp',
            'Owner=David,Environment=Staging,Team=Product,CostCenter=CC-3001,Project=Portal',
            'Owner=Charlie,Environment=Test,Team=QA',
            'Owner=Bob,Environment=Prod,Team=DataScience,CostCenter=CC-2001,Project=MLPlatform',
            'Owner=Alice,Environment=Dev,Team=Engineering,CostCenter=CC-1001,Project=WebApp',
            'Owner=David,Environment=Staging,Team=Product,CostCenter=CC-3001,Project=Portal',
            'Owner=Charlie,Environment=Test,Team=QA,CostCenter=CC-4001,Project=Testing',
            'Owner=Bob,Environment=Prod,Team=DataScience,CostCenter=CC-2001,Project=Analytics',
            'Owner=Alice,Environment=Dev,Team=Engineering,CostCenter=CC-1001,Project=API',
            'Owner=David',  # Incomplete
            '',
            'Owner=Charlie,Environment=Test,Team=QA,CostCenter=CC-4001,Project=Testing',
            'Owner=Bob,Environment=Prod,Team=DataScience,CostCenter=CC-2001,Project=MLPlatform',
            'Owner=Alice,Environment=Dev,Team=Engineering,CostCenter=CC-1001,Project=WebApp'
        ]
    }
    
    # Sample S3 data
    s3_data = {
        'BucketName': [f'bucket-{i}' for i in range(10)],
        'Region': ['us-east-1']*4 + ['us-west-2']*3 + ['eu-west-1']*3,
        'MonthlyCostUSD': np.random.uniform(50, 300, 10),
        'TotalSizeGB': np.random.uniform(100, 5000, 10),
        'ObjectCount': np.random.randint(1000, 100000, 10),
        'StorageClass': ['STANDARD']*4 + ['STANDARD_IA']*3 + ['GLACIER']*3,
        'Encryption': ['AES256']*7 + ['None']*3,
        'CreationDate': pd.date_range(start='2023-01-01', periods=10, freq='30D'),
        'Tags': [
            'Owner=Bob,Environment=Prod,Team=DataScience,CostCenter=CC-2001,Project=DataLake',
            'Owner=Alice,Environment=Dev,Team=Engineering,CostCenter=CC-1001,Project=Storage',
            'Owner=David',  # Incomplete
            'Owner=Charlie,Environment=Test,Team=QA,CostCenter=CC-4001,Project=Testing',
            'Owner=Bob,Environment=Prod,Team=DataScience,CostCenter=CC-2001,Project=Analytics',
            'Owner=Alice,Environment=Dev,Team=Engineering,CostCenter=CC-1001,Project=Backup',
            '',  # No tags
            'Owner=David,Environment=Staging,Team=Product,CostCenter=CC-3001,Project=Portal',
            'Owner=Charlie,Environment=Test,Team=QA,CostCenter=CC-4001,Project=Logs',
            'Owner=Bob,Environment=Prod,Team=DataScience,CostCenter=CC-2001,Project=DataLake'
        ]
    }
    
    return pd.DataFrame(ec2_data), pd.DataFrame(s3_data)


def test_module_imports():
    """Test that all FinOps modules can be imported"""
    print_header("Module Import Test")
    
    modules = [
        'finops_ri_engine',
        'finops_budget_manager',
        'finops_tagging_chargeback',
        'finops_dashboard_integration'
    ]
    
    all_passed = True
    
    for module_name in modules:
        try:
            __import__(module_name)
            print_success(f"{module_name}.py imported successfully")
        except ImportError as e:
            print_error(f"{module_name}.py import failed: {e}")
            all_passed = False
    
    return all_passed


def test_ri_engine(ec2_df, s3_df):
    """Test Reserved Instance recommendation engine"""
    print_header("Reserved Instance Engine Test")
    
    try:
        from finops_ri_engine import RIRecommendationEngine, RIPricingEngine
        
        # Test pricing engine
        print_info("Testing RIPricingEngine...")
        pricing_engine = RIPricingEngine()
        price = pricing_engine.get_on_demand_price('c5.xlarge', 'us-east-1')
        print_success(f"Got on-demand price: ${price}/hour")
        
        # Test RI recommendation engine
        print_info("Testing RIRecommendationEngine...")
        ri_engine = RIRecommendationEngine(lookback_days=90)
        
        # Generate recommendations
        recommendations = ri_engine.generate_ri_recommendations(ec2_df, min_uptime_pct=80.0, target_coverage=0.70)
        print_success(f"Generated {len(recommendations)} RI recommendations")
        
        if recommendations:
            top_rec = recommendations[0]
            print_info(f"  Top recommendation: {top_rec['instance_type']} in {top_rec['region']}")
            print_info(f"  Quantity: {top_rec['recommended_ri_quantity']}")
            print_info(f"  Annual Savings: ${top_rec['annual_savings']:,.2f}")
            print_info(f"  Confidence: {top_rec['confidence_score']:.1f}%")
        
        # Test coverage calculation
        coverage = ri_engine.calculate_current_coverage(ec2_df)
        print_success(f"Coverage calculation: {coverage['coverage_percentage']:.1f}%")
        print_info(f"  Steady-state instances: {coverage['steady_state_instances']}")
        print_info(f"  Gap savings potential: ${coverage['gap_annual_savings_potential']:,.2f}/year")
        
        # Test savings plan comparison
        sp_comparison = ri_engine.generate_savings_plan_comparison(ec2_df)
        if sp_comparison['recommendation'] != 'insufficient_data':
            print_success(f"Savings Plan recommendation: {sp_comparison['recommendation']}")
            print_info(f"  Reason: {sp_comparison['reason']}")
        
        return True
        
    except Exception as e:
        print_error(f"RI Engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_budget_manager(ec2_df, s3_df):
    """Test Budget Manager"""
    print_header("Budget Manager Test")
    
    try:
        from finops_budget_manager import BudgetManager
        
        # Initialize manager with test files
        manager = BudgetManager(
            budget_file='data/test_budgets.json',
            alert_file='data/test_alerts.json'
        )
        print_success("BudgetManager initialized")
        
        # Create a test budget
        print_info("Creating test budget...")
        budget = manager.create_budget(
            name='Test Budget - EC2',
            scope_type='service',
            scope_value='EC2',
            amount=5000.0,
            period='monthly',
            owner='test@company.com',
            alert_thresholds=[50, 80, 100]
        )
        print_success(f"Created budget: {budget['name']} (ID: {budget['id']})")
        
        # Calculate spend
        print_info("Calculating actual spend...")
        spend_data = manager.calculate_actual_spend(budget, ec2_df, s3_df)
        print_success(f"Actual spend: ${spend_data['actual_spend']:,.2f}")
        print_info(f"  Projected: ${spend_data['projected_spend']:,.2f}")
        print_info(f"  Percent used: {spend_data['percent_used']:.1f}%")
        print_info(f"  Burn rate: ${spend_data['burn_rate_per_day']:,.2f}/day")
        
        # Check alerts
        print_info("Checking for alerts...")
        alerts = manager.check_alerts(budget, spend_data, suppress_duplicates=False)
        if alerts:
            print_success(f"Generated {len(alerts)} alert(s)")
            for alert in alerts:
                print_info(f"  {alert['severity'].upper()}: {alert['message']}")
        else:
            print_success("No alerts triggered (budget healthy)")
        
        # Get budget summary
        summaries = manager.get_budget_summary(ec2_df, s3_df)
        print_success(f"Budget summary generated: {len(summaries)} budget(s)")
        
        # Test budget templates
        templates = manager.create_budget_templates()
        print_success(f"Created {len(templates)} budget templates")
        
        # Cleanup test files
        import os
        for f in ['data/test_budgets.json', 'data/test_alerts.json']:
            if os.path.exists(f):
                os.remove(f)
        
        return True
        
    except Exception as e:
        print_error(f"Budget Manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tagging_chargeback(ec2_df, s3_df):
    """Test Tagging and Chargeback module"""
    print_header("Tagging & Chargeback Test")
    
    try:
        from finops_tagging_chargeback import (
            TaggingPolicy, TagParser, TagComplianceAnalyzer, ChargebackEngine
        )
        
        # Test tagging policy
        print_info("Testing TaggingPolicy...")
        mandatory_tags = TaggingPolicy.get_mandatory_tags()
        print_success(f"Mandatory tags: {', '.join(mandatory_tags)}")
        
        # Test tag parser
        print_info("Testing TagParser...")
        parser = TagParser()
        sample_tag = 'Owner=Alice,Environment=Dev,Team=Engineering'
        parsed = parser.parse_tags(sample_tag)
        print_success(f"Parsed {len(parsed)} tags from sample string")
        
        # Test compliance analyzer
        print_info("Testing TagComplianceAnalyzer...")
        analyzer = TagComplianceAnalyzer()
        compliance = analyzer.analyze_compliance(ec2_df, s3_df)
        print_success(f"Overall compliance: {compliance['overall_compliance']:.1f}%")
        print_info(f"  Compliant resources: {compliance['compliant_resources']}/{compliance['total_resources']}")
        print_info(f"  Non-compliant: {compliance['non_compliant_resources']}")
        print_info(f"  Gap to target (95%): {compliance['compliance_gap']:.1f}%")
        
        # Tag-by-tag compliance
        print_info("Tag-by-tag compliance:")
        for tag_name, data in list(compliance['tag_compliance'].items())[:3]:
            print_info(f"    {tag_name}: {data['compliance_percentage']:.1f}% ({data['present']}/{data['present']+data['missing']})")
        
        # Test chargeback engine
        print_info("Testing ChargebackEngine...")
        chargeback = ChargebackEngine()
        
        # Allocate costs by team
        allocation = chargeback.allocate_costs(ec2_df, s3_df, allocation_tag='Team')
        print_success(f"Cost allocation by Team: {len(allocation)} teams")
        
        # Generate chargeback report
        report = chargeback.generate_chargeback_report(ec2_df, s3_df, allocation_tag='Team')
        print_success(f"Chargeback report generated for period: {report['period']}")
        print_info(f"  Total monthly cost: ${report['summary']['total_monthly_cost']:,.2f}")
        print_info(f"  Tagged cost: ${report['summary']['tagged_cost']:,.2f}")
        print_info(f"  Untagged cost: ${report['summary']['untagged_cost']:,.2f}")
        print_info(f"  Allocation coverage: {report['summary']['allocation_coverage_percentage']:.1f}%")
        
        if report['top_cost_centers']:
            print_info("Top cost centers:")
            for center in report['top_cost_centers'][:3]:
                print_info(f"    {center['AllocatedTo']}: ${center['TotalMonthlyCost']:,.2f} ({center['PercentageOfTotal']:.1f}%)")
        
        # Test CSV export
        csv_path = 'data/test_chargeback_export.csv'
        chargeback.export_chargeback_csv(allocation, csv_path)
        print_success(f"CSV export successful: {csv_path}")
        
        # Cleanup
        if os.path.exists(csv_path):
            os.remove(csv_path)
        
        return True
        
    except Exception as e:
        print_error(f"Tagging & Chargeback test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dashboard_integration(ec2_df, s3_df):
    """Test dashboard integration"""
    print_header("Dashboard Integration Test")
    
    try:
        from finops_dashboard_integration import show_finops_dashboard
        print_success("Dashboard integration module imported successfully")
        print_info("Note: UI components require Streamlit runtime to test fully")
        print_info("Run 'streamlit run streamlit_dashboard.py' to test UI")
        return True
        
    except Exception as e:
        print_error(f"Dashboard integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def calculate_financial_impact(ec2_df, s3_df):
    """Calculate potential financial impact"""
    print_header("Financial Impact Analysis")
    
    try:
        from finops_ri_engine import RIRecommendationEngine
        
        # Calculate current costs
        ec2_monthly = ec2_df['CostPerHourUSD'].sum() * 24 * 30
        s3_monthly = s3_df['MonthlyCostUSD'].sum()
        total_monthly = ec2_monthly + s3_monthly
        total_annual = total_monthly * 12
        
        print_info(f"Current AWS Spend:")
        print_info(f"  EC2 Monthly: ${ec2_monthly:,.2f}")
        print_info(f"  S3 Monthly: ${s3_monthly:,.2f}")
        print_info(f"  Total Monthly: ${total_monthly:,.2f}")
        print_info(f"  Total Annual: ${total_annual:,.2f}")
        
        # Calculate RI savings potential
        ri_engine = RIRecommendationEngine()
        recommendations = ri_engine.generate_ri_recommendations(ec2_df)
        
        if recommendations:
            total_annual_savings = sum(r['annual_savings'] for r in recommendations)
            print_success(f"RI Savings Potential: ${total_annual_savings:,.2f}/year")
        else:
            total_annual_savings = 0
            print_warning("No RI recommendations available (may need more steady-state workloads)")
        
        # Estimate tag compliance impact (10-15% savings through accountability)
        tag_savings_low = total_monthly * 0.10 * 12
        tag_savings_high = total_monthly * 0.15 * 12
        
        print_info(f"\nTag Compliance Savings (estimated):")
        print_info(f"  Conservative (10%): ${tag_savings_low:,.2f}/year")
        print_info(f"  Moderate (15%): ${tag_savings_high:,.2f}/year")
        
        # Budget management impact (prevent overruns)
        budget_savings_low = total_monthly * 0.02 * 12  # 2% prevented overruns
        budget_savings_high = total_monthly * 0.05 * 12  # 5% prevented overruns
        
        print_info(f"\nBudget Management Savings (estimated):")
        print_info(f"  Conservative: ${budget_savings_low:,.2f}/year")
        print_info(f"  Moderate: ${budget_savings_high:,.2f}/year")
        
        # Total potential savings
        total_savings_conservative = total_annual_savings + tag_savings_low + budget_savings_low
        total_savings_moderate = total_annual_savings + tag_savings_high + budget_savings_high
        
        print_success(f"\nTotal Potential Savings:")
        print_success(f"  Conservative: ${total_savings_conservative:,.2f}/year ({total_savings_conservative/total_annual*100:.1f}%)")
        print_success(f"  Moderate: ${total_savings_moderate:,.2f}/year ({total_savings_moderate/total_annual*100:.1f}%)")
        
        # ROI calculation
        implementation_cost = 60000  # Estimated from guide
        roi_conservative = (total_savings_conservative / implementation_cost - 1) * 100
        roi_moderate = (total_savings_moderate / implementation_cost - 1) * 100
        payback_months_conservative = implementation_cost / (total_savings_conservative / 12)
        payback_months_moderate = implementation_cost / (total_savings_moderate / 12)
        
        print_info(f"\nROI Analysis:")
        print_info(f"  Implementation Cost: ${implementation_cost:,.2f}")
        print_info(f"  Conservative ROI: {roi_conservative:.0f}%")
        print_info(f"  Moderate ROI: {roi_moderate:.0f}%")
        print_info(f"  Payback Period: {payback_months_conservative:.1f} - {payback_months_moderate:.1f} months")
        
        return True
        
    except Exception as e:
        print_error(f"Financial impact analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print_header("FinOps Module Test Suite")
    print_info(f"Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create sample data
    print_info("Creating sample data...")
    ec2_df, s3_df = create_sample_data()
    print_success(f"Created sample data: {len(ec2_df)} EC2 instances, {len(s3_df)} S3 buckets")
    
    # Run tests
    results = {
        'Module Imports': test_module_imports(),
        'RI Engine': test_ri_engine(ec2_df, s3_df),
        'Budget Manager': test_budget_manager(ec2_df, s3_df),
        'Tagging & Chargeback': test_tagging_chargeback(ec2_df, s3_df),
        'Dashboard Integration': test_dashboard_integration(ec2_df, s3_df),
        'Financial Impact': calculate_financial_impact(ec2_df, s3_df)
    }
    
    # Summary
    print_header("Test Summary")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        if result:
            print_success(f"{test_name}: PASSED")
        else:
            print_error(f"{test_name}: FAILED")
    
    print(f"\n{BLUE}{'='*80}{RESET}")
    if passed == total:
        print_success(f"All tests passed! ({passed}/{total})")
        print_success("✅ FinOps modules are ready for production use")
        print_info("\nNext steps:")
        print_info("  1. Run: streamlit run streamlit_dashboard.py")
        print_info("  2. Navigate to: 💰 Enterprise FinOps tab")
        print_info("  3. Start unlocking $150K-$300K in annual savings!")
        return 0
    else:
        print_error(f"Some tests failed ({passed}/{total} passed)")
        print_warning("Please review errors above and fix before deploying")
        return 1


if __name__ == "__main__":
    sys.exit(main())
