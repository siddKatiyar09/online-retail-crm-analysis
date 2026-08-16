# Portfolio and Interview Guide

## One-Sentence Project Description

I analyzed more than 500,000 online retail transactions, created RFM customer segments, applied
K-means clustering, and translated the results into targeted CRM recommendations.

## Resume Bullet Points

- Cleaned and analyzed more than 500,000 retail transaction records using Python and pandas.
- Engineered Recency, Frequency, and Monetary features for 4,338 identifiable customers.
- Built RFM segments and K-means clusters to identify valuable, loyal, inactive, and at-risk customers.
- Created seven notebook visualizations and an HTML report explaining revenue trends and customer behavior.
- Converted analytical findings into retention, cross-sell, reactivation, and win-back strategies.

## Simple Interview Explanation

The business had transaction data but no clear customer groups. I first cleaned the data by removing
duplicates, anonymous customers, returns, and invalid prices. I then explored revenue by month,
country, product, and weekday. Next, I created RFM features for every customer and assigned readable
segments such as Champions and At Risk. Finally, I used K-means clustering as a second method for
finding customers with similar behavior and suggested marketing actions for each RFM segment.

## Why RFM Was Used

RFM is easy for business teams to understand:

- Recency measures how recently a customer purchased.
- Frequency measures how often the customer purchased.
- Monetary measures how much the customer spent.

Together, these values provide a practical view of customer engagement and value.

## Why K-means Was Used

K-means provides an unsupervised machine-learning view of the customer base. It groups customers
using patterns in their RFM values without requiring predefined labels. The RFM values were
log-transformed and standardized before clustering because the original values had different scales
and were strongly skewed.

## Important Results to Remember

- Clean revenue was approximately `$8.91M`.
- The project analyzed `4,338` identifiable customers.
- The United Kingdom generated `82.01%` of clean revenue.
- November 2011 was the highest-revenue month.
- Champions generated approximately `$6.82M` of customer revenue.

## Possible Interview Questions

### Why remove rows without CustomerID?

Customer segmentation requires a customer identifier. Anonymous transactions can support overall
sales reporting but cannot be connected to an individual customer.

### Why remove negative quantities?

Negative quantities normally represent returns or cancellations. Mixing them with successful sales
would distort purchase frequency and customer spending.

### Why standardize data before K-means?

K-means uses distance. Without standardization, Monetary values would dominate Recency and Frequency
because spending has a much larger numeric scale.

### What would you improve next?

I would add customer lifetime value, product recommendations, a Streamlit dashboard, and a scheduled
pipeline that updates customer segments as new transactions arrive.
