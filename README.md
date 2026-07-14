# Online Retail CRM Analytics and Customer Segmentation

An end-to-end CRM analytics project built on the UCI-style `OnlineRetail.csv` transactional dataset.  
The project cleans raw sales logs, performs business-focused exploratory analysis, engineers customer-level CRM features, and applies both RFM segmentation and unsupervised machine learning to identify high-value customer groups.

## Project Highlights

- Cleaned raw transaction data for missing customer IDs, returns, and invalid pricing
- Performed EDA on revenue trends, country performance, and product contribution
- Built customer-level RFM features for CRM campaign planning
- Implemented K-means clustering with `numpy` for portfolio-friendly machine learning
- Generated an executive-style HTML report with visuals and exportable CSV outputs

## Business Problem

Retail businesses often have transaction data but lack a structured customer view.  
This project converts raw orders into CRM-ready insights so a business can:

- identify high-value customers
- spot inactive or at-risk buyers
- understand revenue concentration
- design retention, reactivation, and upsell campaigns

## Tech Stack

- Python
- pandas
- numpy

## Repository Structure

```text
online_retail_crm_project/
|-- data/
|   |-- README.md
|-- output/
|   |-- analysis_report.html
|   |-- executive_summary.md
|   |-- customer_rfm_segments.csv
|   |-- cluster_profile.csv
|   |-- kmeans_diagnostics.csv
|   |-- monthly_revenue.csv
|   |-- rfm_segment_summary.csv
|   |-- top_products.csv
|   |-- figures/
|-- src/
|   |-- analyze_online_retail.py
|-- .gitignore
|-- README.md
|-- requirements.txt
```

## Dataset Setup

1. Download or copy `OnlineRetail.csv`
2. Place it inside the `data/` folder
3. Keep the filename as `OnlineRetail.csv`

The dataset itself is not committed so the repository stays lightweight and GitHub-ready.

## How to Run

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the full analysis:

```powershell
python src/analyze_online_retail.py
```

If your dataset is in another location, run:

```powershell
python src/analyze_online_retail.py --input "path\to\OnlineRetail.csv" --output-dir output
```

## Outputs

The script generates:

- `output/analysis_report.html` for a polished portfolio-ready report
- `output/executive_summary.md` for quick recruiter or resume talking points
- `output/customer_rfm_segments.csv` with customer-level features and labels
- `output/cluster_profile.csv` with cluster summaries
- `output/kmeans_diagnostics.csv` with clustering diagnostics
- `output/figures/` with SVG charts

## Key Insights from This Run

- CRM-ready cleaned revenue: about `$8.91M`
- Identified customers used for CRM analysis: `4,338`
- Revenue concentration in the UK: `82.01%`
- Top cluster: `Champions`
- Strong seasonality with peak revenue in late 2011

## Resume-Ready Skills Demonstrated

- Exploratory Data Analysis
- Customer Segmentation
- RFM Analysis
- Unsupervised Machine Learning
- Business Intelligence Storytelling
- Reproducible Data Science Projects

## Suggested GitHub Repository Name

- `online-retail-crm-analysis`
- `crm-customer-segmentation-project`
- `retail-customer-analytics`
