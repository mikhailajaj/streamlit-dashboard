"""
FinOps Dashboard Integration Module
Integrates RI, Budget, and Tagging modules into Streamlit dashboard

This module provides UI components for:
- Reserved Instance recommendations and tracking
- Budget management and alerting
- Tag compliance and chargeback reporting
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Import FinOps engines
try:
    from finops.ri_engine import RIRecommendationEngine, RIPricingEngine
    from finops.budget_manager import BudgetManager
    from finops.tagging import TagComplianceAnalyzer, ChargebackEngine, TaggingPolicy
    FINOPS_AVAILABLE = True
except ImportError:
    FINOPS_AVAILABLE = False


# WCAG AA Compliant Colors (from main dashboard)
WCAG_COLORS = {
    'aws_orange': '#D86613',
    'aws_dark': '#232F3E',
    'success': '#0F8C4F',
    'warning': '#B7791F',
    'error': '#C52A1E',
    'info': '#0972D3',
    'chart_blue': '#0972D3',
    'chart_orange': '#D86613',
    'chart_green': '#0F8C4F',
    'chart_purple': '#8B3FD9',
    'chart_teal': '#067F88',
    'chart_pink': '#C7407B',
}


def show_finops_dashboard(ec2_df: pd.DataFrame, s3_df: pd.DataFrame):
    """Main FinOps dashboard interface"""
    
    if not FINOPS_AVAILABLE:
        st.error("⚠️ FinOps modules not available. Please ensure finops_*.py files are in the correct directory.")
        return
    
    st.header("💰 Enterprise FinOps Platform")
    st.markdown("""
    **Unlock $150K-$300K annual savings** through Reserved Instances, Budget Management, and Tag Compliance.
    """)
    
    # High-level KPI metrics
    show_finops_kpis(ec2_df, s3_df)
    
    # Tabbed interface for each FinOps capability
    finops_tab1, finops_tab2, finops_tab3, finops_tab4 = st.tabs([
        "🎯 RI & Savings Plans",
        "📊 Budget Management", 
        "🏷️ Tag Compliance",
        "💳 Chargeback Reports"
    ])
    
    with finops_tab1:
        show_ri_recommendations(ec2_df, s3_df)
    
    with finops_tab2:
        show_budget_management(ec2_df, s3_df)
    
    with finops_tab3:
        show_tag_compliance(ec2_df, s3_df)
    
    with finops_tab4:
        show_chargeback_reports(ec2_df, s3_df)


def show_finops_kpis(ec2_df: pd.DataFrame, s3_df: pd.DataFrame):
    """Display high-level FinOps KPIs"""
    
    # Calculate current spend
    ec2_monthly = ec2_df['CostPerHourUSD'].sum() * 24 * 30
    s3_monthly = s3_df['MonthlyCostUSD'].sum()
    total_monthly = ec2_monthly + s3_monthly
    
    # Initialize engines
    ri_engine = RIRecommendationEngine()
    coverage = ri_engine.calculate_current_coverage(ec2_df)
    
    # Calculate potential savings (simplified)
    potential_ri_savings = coverage['gap_annual_savings_potential'] / 12  # Monthly
    potential_total_savings = potential_ri_savings * 1.2  # Add buffer for other optimizations
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="💰 Monthly Spend",
            value=f"${total_monthly:,.0f}",
            delta=f"${total_monthly * 12:,.0f}/year",
            help="Total monthly AWS spend across EC2 and S3"
        )
    
    with col2:
        st.metric(
            label="🎯 RI Coverage",
            value=f"{coverage['coverage_percentage']:.1f}%",
            delta=f"Target: 70%",
            delta_color="inverse",
            help="Percentage of steady-state workloads covered by Reserved Instances"
        )
    
    with col3:
        st.metric(
            label="💡 Savings Potential",
            value=f"${potential_total_savings:,.0f}/mo",
            delta=f"${potential_total_savings * 12:,.0f}/year",
            help="Estimated monthly savings from RI purchases and optimizations"
        )
    
    with col4:
        # Tag compliance
        analyzer = TagComplianceAnalyzer()
        compliance = analyzer.analyze_compliance(ec2_df, s3_df)
        
        st.metric(
            label="🏷️ Tag Compliance",
            value=f"{compliance['overall_compliance']:.1f}%",
            delta=f"Target: 95%",
            delta_color="inverse",
            help="Percentage of resources with all mandatory tags"
        )
    
    with col5:
        # Steady-state instances
        st.metric(
            label="⚡ Steady Instances",
            value=coverage['steady_state_instances'],
            delta=f"{coverage['steady_state_instances']/len(ec2_df)*100:.0f}% of total",
            help="Instances suitable for Reserved Instance purchase (>80% uptime)"
        )


def show_ri_recommendations(ec2_df: pd.DataFrame, s3_df: pd.DataFrame):
    """Display Reserved Instance recommendations"""
    
    st.subheader("🎯 Reserved Instance & Savings Plan Recommendations")
    st.markdown("""
    **Financial Impact:** Up to $125,000/year in savings
    
    Reserved Instances (RIs) offer significant discounts (40-72%) for steady-state workloads.
    This analysis identifies optimal RI purchase recommendations.
    """)
    
    # Initialize engine
    ri_engine = RIRecommendationEngine()
    
    # Configuration options
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        min_uptime = st.slider(
            "Minimum Uptime Threshold (%)",
            60, 100, 80, 5,
            help="Only recommend RIs for instances with uptime above this threshold"
        )
    
    with col2:
        target_coverage = st.slider(
            "Target RI Coverage (%)",
            50, 100, 70, 5,
            help="Target percentage of steady-state workloads to cover with RIs"
        ) / 100
    
    with col3:
        if st.button("🔄 Refresh Analysis", key="refresh_ri"):
            st.rerun()
    
    # Generate recommendations
    with st.spinner("Analyzing usage patterns and generating recommendations..."):
        recommendations = ri_engine.generate_ri_recommendations(
            ec2_df, 
            min_uptime_pct=min_uptime,
            target_coverage=target_coverage
        )
        
        coverage_stats = ri_engine.calculate_current_coverage(ec2_df)
        sp_comparison = ri_engine.generate_savings_plan_comparison(ec2_df)
    
    # Display coverage metrics
    st.markdown("---")
    st.subheader("📊 Current RI Coverage Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Instances", coverage_stats['total_instances'])
    with col2:
        st.metric("Steady-State", coverage_stats['steady_state_instances'])
    with col3:
        st.metric("Current Coverage", f"{coverage_stats['coverage_percentage']:.1f}%")
    with col4:
        st.metric("Coverage Gap", int(coverage_stats['coverage_gap']))
    
    # Potential savings
    st.info(f"""
    💰 **Closing the coverage gap could save ${coverage_stats['gap_monthly_savings_potential']:,.2f}/month** 
    (${coverage_stats['gap_annual_savings_potential']:,.2f}/year)
    """)
    
    # RI Recommendations Table
    st.markdown("---")
    st.subheader("🎯 Top RI Purchase Recommendations")
    
    if len(recommendations) == 0:
        st.warning("No RI recommendations available. Try lowering the minimum uptime threshold or check that you have steady-state workloads.")
    else:
        # Display top recommendations
        rec_df = pd.DataFrame(recommendations)
        
        # Format for display
        display_df = rec_df[[
            'instance_type', 'region', 'recommended_ri_quantity',
            'recommended_term', 'recommended_payment', 'discount_percentage',
            'monthly_savings', 'annual_savings', 'total_upfront_cost',
            'payback_months', 'confidence_score', 'recommendation_reason'
        ]].copy()
        
        display_df.columns = [
            'Instance Type', 'Region', 'Qty', 'Term', 'Payment',
            'Discount %', 'Monthly Savings', 'Annual Savings', 'Upfront Cost',
            'Payback (mo)', 'Confidence %', 'Reason'
        ]
        
        # Format currency columns
        for col in ['Monthly Savings', 'Annual Savings', 'Upfront Cost']:
            display_df[col] = display_df[col].apply(lambda x: f"${x:,.2f}")
        
        # Format percentage columns
        for col in ['Discount %', 'Confidence %']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}%")
        
        display_df['Payback (mo)'] = display_df['Payback (mo)'].apply(lambda x: f"{x:.1f}")
        
        st.dataframe(display_df, use_container_width=True, height=400)
        
        # Total savings summary
        total_monthly = rec_df['monthly_savings'].sum()
        total_annual = rec_df['annual_savings'].sum()
        total_upfront = rec_df['total_upfront_cost'].sum()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Monthly Savings", f"${total_monthly:,.2f}")
        with col2:
            st.metric("Total Annual Savings", f"${total_annual:,.2f}")
        with col3:
            st.metric("Total Upfront Investment", f"${total_upfront:,.2f}")
        
        # Visualization: Savings by instance family
        st.markdown("---")
        st.subheader("📈 Savings Potential by Instance Type")
        
        fig = px.bar(
            rec_df.head(10),
            x='instance_type',
            y='annual_savings',
            color='region',
            title="Top 10 Instance Types by Annual Savings",
            labels={'annual_savings': 'Annual Savings (USD)', 'instance_type': 'Instance Type'},
            color_discrete_sequence=[WCAG_COLORS['chart_blue'], WCAG_COLORS['chart_orange'], 
                                    WCAG_COLORS['chart_green'], WCAG_COLORS['chart_purple']]
        )
        fig.update_yaxes(tickformat='$,.0f')
        st.plotly_chart(fig, use_container_width=True)
    
    # Savings Plans comparison
    st.markdown("---")
    st.subheader("💡 Reserved Instances vs Savings Plans")
    
    if sp_comparison['recommendation'] != 'insufficient_data':
        st.info(f"**Recommendation:** {sp_comparison['recommendation'].replace('_', ' ').title()}")
        st.markdown(f"**Reason:** {sp_comparison['reason']}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Compute Savings Plans** (Most Flexible)")
            st.write(f"- 1-Year: {sp_comparison['compute_savings_plan']['1yr_discount_pct']:.0f}% discount = ${sp_comparison['compute_savings_plan']['1yr_annual_savings']:,.0f}/year")
            st.write(f"- 3-Year: {sp_comparison['compute_savings_plan']['3yr_discount_pct']:.0f}% discount = ${sp_comparison['compute_savings_plan']['3yr_annual_savings']:,.0f}/year")
        
        with col2:
            st.markdown("**EC2 Instance Savings Plans** (Higher Discount)")
            st.write(f"- 1-Year: {sp_comparison['ec2_instance_savings_plan']['1yr_discount_pct']:.0f}% discount = ${sp_comparison['ec2_instance_savings_plan']['1yr_annual_savings']:,.0f}/year")
            st.write(f"- 3-Year: {sp_comparison['ec2_instance_savings_plan']['3yr_discount_pct']:.0f}% discount = ${sp_comparison['ec2_instance_savings_plan']['3yr_annual_savings']:,.0f}/year")


def show_budget_management(ec2_df: pd.DataFrame, s3_df: pd.DataFrame):
    """Display budget management interface"""
    
    st.subheader("📊 Budget Management & Alerts")
    st.markdown("""
    **Financial Impact:** Prevent $12K-$24K/year in budget overruns
    
    Create budgets, monitor spending, and receive alerts before overspending occurs.
    """)
    
    # Initialize budget manager
    manager = BudgetManager()
    
    # Budget management tabs
    budget_tab1, budget_tab2, budget_tab3 = st.tabs([
        "📋 Budget Overview",
        "➕ Create Budget",
        "🚨 Active Alerts"
    ])
    
    with budget_tab1:
        show_budget_overview(manager, ec2_df, s3_df)
    
    with budget_tab2:
        show_budget_creation(manager, ec2_df, s3_df)
    
    with budget_tab3:
        show_budget_alerts(manager, ec2_df, s3_df)


def show_budget_overview(manager: BudgetManager, ec2_df: pd.DataFrame, s3_df: pd.DataFrame):
    """Display budget overview"""
    
    summaries = manager.get_budget_summary(ec2_df, s3_df, active_only=True)
    
    if len(summaries) == 0:
        st.info("📝 No budgets created yet. Use the 'Create Budget' tab to set up your first budget.")
        
        # Show templates
        st.markdown("### 📋 Budget Templates")
        st.markdown("Here are some recommended budget templates based on your current spend:")
        
        templates = manager.create_budget_templates()
        template_df = pd.DataFrame(templates)
        st.dataframe(template_df, use_container_width=True)
        
        return
    
    # Display budget cards
    st.markdown("### Active Budgets")
    
    for summary in summaries:
        status_color = {
            'healthy': WCAG_COLORS['success'],
            'caution': WCAG_COLORS['info'],
            'warning': WCAG_COLORS['warning'],
            'at_risk': WCAG_COLORS['warning'],
            'exceeded': WCAG_COLORS['error']
        }.get(summary['status'], WCAG_COLORS['info'])
        
        status_icon = {
            'healthy': '✅',
            'caution': '📊',
            'warning': '⚠️',
            'at_risk': '⚠️',
            'exceeded': '🚨'
        }.get(summary['status'], '📊')
        
        with st.expander(f"{status_icon} {summary['name']} - {summary['percent_used']:.1f}% Used", expanded=(summary['status'] in ['at_risk', 'exceeded'])):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Budget", f"${summary['budget_amount']:,.2f}")
            with col2:
                st.metric("Actual Spend", f"${summary['actual_spend']:,.2f}")
            with col3:
                st.metric("Projected", f"${summary['projected_spend']:,.2f}")
            with col4:
                st.metric("Remaining", f"${summary['remaining']:,.2f}")
            
            # Progress bar
            progress_pct = min(summary['percent_used'] / 100, 1.0)
            st.progress(progress_pct)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Scope:** {summary['scope']}")
                st.write(f"**Owner:** {summary['owner']}")
            with col2:
                st.write(f"**Burn Rate:** ${summary['burn_rate']:,.2f}/day")
                st.write(f"**Days Remaining:** {summary['days_remaining']}")


def show_budget_creation(manager: BudgetManager, ec2_df: pd.DataFrame, s3_df: pd.DataFrame):
    """Display budget creation form"""
    
    st.markdown("### Create New Budget")
    
    col1, col2 = st.columns(2)
    
    with col1:
        budget_name = st.text_input("Budget Name", placeholder="e.g., Q1 2025 AWS Spend")
        
        scope_type = st.selectbox(
            "Scope Type",
            ['total', 'service', 'region', 'team', 'environment'],
            format_func=lambda x: x.title()
        )
        
        if scope_type == 'service':
            scope_value = st.selectbox("Service", ['EC2', 'S3'])
        elif scope_type == 'region':
            regions = sorted(set(ec2_df['Region'].unique()) | set(s3_df['Region'].unique()))
            scope_value = st.selectbox("Region", regions)
        elif scope_type == 'team':
            scope_value = st.text_input("Team Name", placeholder="e.g., Engineering")
        elif scope_type == 'environment':
            scope_value = st.selectbox("Environment", ['Dev', 'Test', 'Staging', 'Prod'])
        else:
            scope_value = 'all'
    
    with col2:
        budget_amount = st.number_input("Budget Amount (USD)", min_value=0.0, value=10000.0, step=100.0)
        
        period = st.selectbox("Period", ['monthly', 'quarterly', 'annually'])
        
        owner = st.text_input("Budget Owner", placeholder="email@company.com")
        
        thresholds = st.multiselect(
            "Alert Thresholds (%)",
            [50, 60, 70, 80, 90, 100],
            default=[50, 80, 100]
        )
    
    if st.button("💾 Create Budget", type="primary"):
        if budget_name and budget_amount > 0:
            budget = manager.create_budget(
                name=budget_name,
                scope_type=scope_type,
                scope_value=scope_value,
                amount=budget_amount,
                period=period,
                owner=owner,
                alert_thresholds=thresholds
            )
            st.success(f"✅ Budget '{budget_name}' created successfully!")
            st.balloons()
        else:
            st.error("Please fill in all required fields")


def show_budget_alerts(manager: BudgetManager, ec2_df: pd.DataFrame, s3_df: pd.DataFrame):
    """Display budget alerts"""
    
    st.markdown("### 🚨 Budget Alerts")
    
    budgets = manager.list_budgets(active_only=True)
    all_alerts = []
    
    for budget in budgets:
        spend_data = manager.calculate_actual_spend(budget, ec2_df, s3_df)
        alerts = manager.check_alerts(budget, spend_data, suppress_duplicates=False)
        all_alerts.extend(alerts)
    
    if len(all_alerts) == 0:
        st.success("✅ No active budget alerts. All budgets are within thresholds.")
        return
    
    # Display alerts
    for alert in sorted(all_alerts, key=lambda x: x['severity'], reverse=True):
        severity_color = {
            'critical': 'error',
            'high': 'warning',
            'medium': 'warning',
            'low': 'info'
        }.get(alert['severity'], 'info')
        
        getattr(st, severity_color)(alert['message'])


def show_tag_compliance(ec2_df: pd.DataFrame, s3_df: pd.DataFrame):
    """Display tag compliance dashboard"""
    
    st.subheader("🏷️ Tag Compliance & Governance")
    st.markdown("""
    **Financial Impact:** $94K-$140K/year through team accountability
    
    Track tag compliance and enforce tagging policies across all resources.
    """)
    
    # Initialize analyzer
    analyzer = TagComplianceAnalyzer()
    compliance = analyzer.analyze_compliance(ec2_df, s3_df)
    
    # High-level metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Overall Compliance", f"{compliance['overall_compliance']:.1f}%")
    with col2:
        st.metric("Compliant Resources", f"{compliance['compliant_resources']}/{compliance['total_resources']}")
    with col3:
        st.metric("Target", "95.0%")
    with col4:
        gap_color = "inverse" if compliance['compliance_gap'] > 0 else "off"
        st.metric("Gap to Target", f"{compliance['compliance_gap']:.1f}%", delta_color=gap_color)
    
    # Progress to target
    st.progress(min(compliance['overall_compliance'] / 100, 1.0))
    
    # Tag-by-tag compliance
    st.markdown("---")
    st.subheader("📊 Compliance by Tag")
    
    tag_data = []
    for tag_name, data in compliance['tag_compliance'].items():
        tag_data.append({
            'Tag': tag_name,
            'Present': data['present'],
            'Missing': data['missing'],
            'Compliance %': data['compliance_percentage']
        })
    
    tag_df = pd.DataFrame(tag_data)
    
    fig = px.bar(
        tag_df,
        x='Tag',
        y='Compliance %',
        title="Tag Compliance by Type",
        color='Compliance %',
        color_continuous_scale=[[0, WCAG_COLORS['error']], [0.5, WCAG_COLORS['warning']], [1, WCAG_COLORS['success']]],
        labels={'Compliance %': 'Compliance (%)'}
    )
    fig.update_yaxes(range=[0, 100])
    fig.add_hline(y=95, line_dash="dash", line_color="gray", annotation_text="Target: 95%")
    st.plotly_chart(fig, use_container_width=True)
    
    # Service and region compliance
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Compliance by Service")
        service_df = pd.DataFrame(compliance['service_compliance'])
        st.dataframe(service_df, use_container_width=True)
    
    with col2:
        st.markdown("### Compliance by Region")
        region_df = pd.DataFrame(compliance['region_compliance'])
        st.dataframe(region_df, use_container_width=True)
    
    # Non-compliant resources
    st.markdown("---")
    st.subheader("⚠️ Non-Compliant Resources (Top 50)")
    
    if len(compliance['non_compliant_resources']) > 0:
        non_compliant_df = pd.DataFrame(compliance['non_compliant_resources'])
        non_compliant_df['missing_tags'] = non_compliant_df['missing_tags'].apply(lambda x: ', '.join(x))
        st.dataframe(non_compliant_df, use_container_width=True, height=300)
    else:
        st.success("🎉 All resources are compliant!")


def show_chargeback_reports(ec2_df: pd.DataFrame, s3_df: pd.DataFrame):
    """Display chargeback reports"""
    
    st.subheader("💳 Showback & Chargeback Reports")
    st.markdown("""
    **Drive accountability** by allocating costs to teams, cost centers, and projects.
    """)
    
    # Initialize chargeback engine
    chargeback = ChargebackEngine()
    
    # Allocation dimension selector
    allocation_tag = st.selectbox(
        "Allocate Costs By",
        ['Team', 'Owner', 'Environment', 'CostCenter', 'Project'],
        help="Select the tag dimension for cost allocation"
    )
    
    # Generate report
    with st.spinner("Generating chargeback report..."):
        report = chargeback.generate_chargeback_report(ec2_df, s3_df, allocation_tag=allocation_tag)
    
    # Summary metrics
    st.markdown("### 📊 Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Monthly Cost", f"${report['summary']['total_monthly_cost']:,.2f}")
    with col2:
        st.metric("Tagged Cost", f"${report['summary']['tagged_cost']:,.2f}")
    with col3:
        st.metric("Untagged Cost", f"${report['summary']['untagged_cost']:,.2f}")
    with col4:
        st.metric("Allocation Coverage", f"{report['summary']['allocation_coverage_percentage']:.1f}%")
    
    # Top cost centers
    st.markdown("---")
    st.subheader(f"💰 Top 5 Cost Centers by {allocation_tag}")
    
    top_5_df = pd.DataFrame(report['top_cost_centers'])
    if len(top_5_df) > 0:
        fig = px.pie(
            top_5_df,
            values='TotalMonthlyCost',
            names='AllocatedTo',
            title=f"Cost Distribution by {allocation_tag}",
            color_discrete_sequence=[WCAG_COLORS['chart_blue'], WCAG_COLORS['chart_orange'],
                                    WCAG_COLORS['chart_green'], WCAG_COLORS['chart_purple'],
                                    WCAG_COLORS['chart_teal']]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed allocation table
    st.markdown("---")
    st.subheader("📋 Detailed Cost Allocation")
    
    allocation_df = pd.DataFrame(report['detailed_allocation'])
    
    # Format for display
    display_cols = ['AllocatedTo', 'TotalMonthlyCost', 'EC2_MonthlyCost', 'S3_MonthlyCost',
                   'EC2_Count', 'S3_Count', 'TotalResourceCount', 'PercentageOfTotal']
    
    if all(col in allocation_df.columns for col in display_cols):
        display_df = allocation_df[display_cols].copy()
        display_df.columns = [col.replace('_', ' ') for col in display_cols]
        
        # Format currency
        for col in [c for c in display_df.columns if 'Cost' in c]:
            display_df[col] = display_df[col].apply(lambda x: f"${x:,.2f}")
        
        # Format percentage
        display_df['PercentageOfTotal'] = display_df['PercentageOfTotal'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(display_df, use_container_width=True, height=400)
        
        # Export button
        if st.button("📥 Export to CSV"):
            csv_path = f"data/chargeback_report_{allocation_tag}_{datetime.now().strftime('%Y%m%d')}.csv"
            chargeback.export_chargeback_csv(allocation_df, csv_path)
            st.success(f"✅ Report exported to {csv_path}")
    
    # Recommendations
    if report['recommendations']:
        st.markdown("---")
        st.subheader("💡 Recommendations")
        for i, rec in enumerate(report['recommendations'], 1):
            st.info(f"{i}. {rec}")


if __name__ == "__main__":
    st.write("This module is meant to be imported into the main dashboard.")
