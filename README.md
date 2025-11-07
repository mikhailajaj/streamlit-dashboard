# Week 9 Activity: Exploratory Analysis of EC2 & S3 Usage

## Overview
This project performs comprehensive exploratory data analysis (EDA) on AWS EC2 and S3 datasets to understand cost distribution, usage efficiency, and optimization opportunities across AWS regions.

## Project Structure
```
activity-nov-5/
├── Week 9 - EDA/
│   ├── aws_resources_compute.csv    # EC2 dataset
│   └── aws_resources_S3.csv         # S3 dataset
├── eda_analysis.py                  # Main analysis script
├── eda_lib.py                       # Reusable analysis library
├── streamlit_dashboard.py           # Interactive dashboard
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── ec2_analysis.png                 # EC2 visualizations
└── s3_analysis.png                  # S3 visualizations
```

## Datasets

### EC2 Dataset (200 instances, 120 valid after cleaning)
- **InstanceId**: Unique EC2 instance identifier
- **InstanceType**: Instance type (t3.small, m5.large, etc.)
- **Region**: AWS region
- **State**: Instance state (running/stopped/terminated)
- **CPUUtilization**: Average CPU usage percentage
- **MemoryUtilization**: Average memory usage percentage
- **NetworkIn_Bps**: Average incoming network bytes/sec
- **NetworkOut_Bps**: Average outgoing network bytes/sec
- **CostPerHourUSD**: On-demand cost per hour
- **Tags**: Key=value pairs for Owner and Environment
- **LaunchTime**: Instance launch datetime

### S3 Dataset (80 buckets, all valid)
- **BucketName**: S3 bucket name
- **Region**: AWS region
- **StorageClass**: STANDARD, STANDARD_IA, GLACIER
- **ObjectCount**: Total objects in bucket
- **TotalSizeGB**: Total bucket size in GB
- **MonthlyCostUSD**: Estimated monthly storage cost
- **VersioningEnabled**: True/False
- **Encryption**: AES256 or None
- **CreatedDate**: Bucket creation date
- **Tags**: Owner and Purpose tags

## Analysis Tasks Completed ✅

### 1. Data Loading and Exploration
- ✅ Loaded both datasets into pandas
- ✅ Displayed dataset info, shape, and summary statistics
- ✅ Identified data types and structure

### 2. Data Cleaning and Quality Assessment
- ✅ Handled missing data (removed 80 incomplete EC2 records)
- ✅ Detected outliers using IQR method
- ✅ Cleaned and standardized data formats

### 3. Visualizations Created
- ✅ **EC2 CPU Utilization Histogram**: Distribution of CPU usage
- ✅ **EC2 CPU vs Cost Scatter Plot**: Relationship between utilization and cost
- ✅ **S3 Storage by Region Bar Chart**: Total storage distribution
- ✅ **S3 Cost vs Storage Scatter Plot**: Cost efficiency analysis

### 4. Key Insights Identified
- ✅ **Top 5 Most Expensive EC2 Instances**: Highest cost instances
- ✅ **Top 5 Largest S3 Buckets**: Biggest storage consumers
- ✅ **Average EC2 Cost per Region**: Regional cost analysis
- ✅ **Total S3 Storage per Region**: Regional storage distribution

### 5. Optimization Recommendations
- ✅ **EC2 Optimizations**: Rightsizing underutilized instances
- ✅ **S3 Optimizations**: Lifecycle policies and encryption improvements

### 6. Interactive Dashboard
- ✅ **Streamlit Dashboard**: Comprehensive interactive analysis tool

## How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run Main Analysis
```bash
python eda_analysis.py
```

### Launch Interactive Dashboard
```bash
streamlit run streamlit_dashboard.py
```

## Key Findings

### EC2 Analysis
- **Total Instances**: 120 (after cleaning)
- **Monthly Cost Estimate**: $1,643.94
- **Average CPU Utilization**: 49.8%
- **Most Expensive Region**: us-west-2
- **Optimization Potential**: 15 instances with CPU < 10%

### S3 Analysis
- **Total Buckets**: 80
- **Total Storage**: 183,926.05 GB
- **Monthly Cost**: $13,171.59
- **Storage Distribution**: us-east-1 leads with 67,513.89 GB
- **Security**: 19 buckets lack encryption

### Regional Distribution
- **us-east-1**: Highest S3 storage, moderate EC2 costs
- **us-west-2**: Highest EC2 costs, significant S3 presence
- **eu-west-1**: Balanced EC2/S3 distribution
- **ap-south-1**: Lower costs but growing usage

## Optimization Recommendations

### EC2 Optimizations
1. **Rightsize Underutilized Instances**: 15 instances with <10% CPU could save $324.79/month
2. **Review Expensive Low-Utilization**: High-cost instances with <25% CPU need attention

### S3 Optimizations
1. **Implement Lifecycle Policies**: Move STANDARD storage to IA/Glacier for 30-70% savings
2. **Enable Encryption**: 19 unencrypted buckets need security improvements

### Potential Monthly Savings
- **EC2**: $324.79 from rightsizing
- **S3**: $985.37 from lifecycle policies
- **Total**: $1,310.16/month ($15,721.92/year)

## Technology Stack
- **Python**: Data analysis and processing
- **Pandas**: Data manipulation and analysis
- **Matplotlib/Seaborn**: Static visualizations
- **Plotly**: Interactive visualizations
- **Streamlit**: Web dashboard framework

## Files Generated
- `ec2_analysis.png`: EC2 analysis visualizations
- `s3_analysis.png`: S3 analysis visualizations
- Analysis outputs and recommendations in console

## Dashboard Features
- **Overview**: Key metrics and regional comparisons
- **EC2 Analysis**: Instance types, utilization, efficiency
- **S3 Analysis**: Storage classes, encryption, costs
- **Comparative Analysis**: Side-by-side regional analysis
- **Optimization**: Actionable recommendations and savings potential

## Next Steps
1. Implement automated monitoring for low-utilization instances
2. Set up S3 lifecycle policies for cost optimization
3. Enable encryption for all S3 buckets
4. Regular review of instance rightsizing opportunities
5. Monitor regional cost trends for strategic planning