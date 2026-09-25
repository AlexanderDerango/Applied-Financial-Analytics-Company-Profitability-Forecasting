# Applied Financial Analytics: Company Profitability & Forecasting

An applied financial analytics project that uses company financial statements, data science, and statistical modeling to analyze profitability trends and forecast future financial performance.

## Project Overview

The goal of this project is to use financial data to:

* Identify variables associated with company profitability
* Analyze long-term financial trends
* Compare financial performance across companies
* Engineer meaningful financial ratios
* Build regression models to estimate future trends
* Develop an interactive dashboard for exploring financial data and model results

The project combines **data cleaning, financial analysis, exploratory data analysis, statistical modeling, and interactive visualization**.

## Research Questions

* Which financial variables are most closely associated with company profitability?
* What long-term trends can be identified in company financial statements?
* How can regression models be used to estimate future financial trends?
* How do profitability, liquidity, cash flow, and capital structure change over time?

## Dataset

Financial statement data was collected using the **Alpha Vantage API**.

The dataset consists of three primary financial statements:

### Balance Sheet

* Total Assets
* Investments
* Inventory
* Liabilities
* Debt
* Equity

### Income Statement

* Gross Profit
* Total Revenue
* Taxes
* Total Costs and Expenses
* Depreciation
* Net Income

### Cash Flow Statement

* Operating Cash Flow
* Capital Expenditures
* Cash Flow from Investing

The project analyzes six publicly traded companies across multiple years.

## Feature Engineering

Financial ratios were created from the cleaned financial statements to provide more interpretable measures of company performance.

Examples include:

* Gross Margin
* Operating Margin
* Net Margin
* Return on Assets (ROA)
* Return on Equity (ROE)
* Current Ratio
* Debt-to-Equity
* Free Cash Flow
* Interest Coverage
* Cash Flow Ratios

These features allow financial performance to be compared across companies and over time.

## Exploratory Data Analysis

The analysis examined long-term trends in:

* Revenue
* Net Income
* Profit Margins
* ROA and ROE
* Operating Cash Flow
* Capital Expenditures
* Free Cash Flow
* Assets and Liabilities
* Equity
* Liquidity
* Retained Earnings

The project also examined company-specific financial trends for firms including:

* Apple (AAPL)
* Alphabet (GOOG)
* Keysight Technologies (KEYS)
* Meta Platforms (META)
* NVIDIA (NVDA)
* Taiwan Semiconductor Manufacturing Company (TSM)

## Statistical Modeling

The primary model used in the project is **linear regression**.

The model is used to identify long-term trends in financial variables and generate forecasts.

### Model Evaluation

Several statistical measures are used to evaluate the regression results:

* **R²** — measures the proportion of variation explained by the time trend
* **t-statistic** — evaluates the statistical significance of the estimated trend
* **p-value** — evaluates evidence against the null hypothesis of no trend
* **90% prediction intervals** — provide uncertainty ranges around forecasts
* **CAGR** — summarizes compound annual growth over time

The model is intended to identify and extrapolate historical trends rather than provide investment recommendations.

## Interactive Dashboard

The project includes an interactive financial analytics dashboard that allows users to explore company financial data and model outputs.

The dashboard includes:

* Company selection
* Financial KPI summaries
* Historical trend charts
* Regression trend lines
* Forecasts
* 90% prediction intervals
* CAGR calculations
* Financial ratio analysis
* Support for uploading financial statement CSV files

The upload functionality allows users to provide income statement, balance sheet, and cash flow data for additional companies and run the same analysis workflow.

## Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Requests
* HTML
* JavaScript
* Alpha Vantage API
* Google Colab

## Project Structure

```text
├── Presentation.pdf
├── README.md
├── data_cleaning.py
├── get_data.py
└── website.html
```

## Future Work

Potential extensions include:

* Adding macroeconomic variables such as inflation, interest rates, and GDP
* Expanding the number of companies and industries
* Adding additional financial predictors
* Comparing multiple regression approaches
* Incorporating more advanced time-series models
* Testing model performance using out-of-sample data
* Expanding the dashboard's financial analysis capabilities
