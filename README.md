# Apartment Rent Price Prediction: Project Overview
* Created a tool that predicts apartment rent price in 4 US metropolitan areas (Dallas, Seattle, Los Angeles, San Francisco) to help young professionals to make better life decisions
* Scraped over 2000 rental apartment postings from Zillow using Python and scrapy
* Engineered features addressing issues including outliers removal, missing values imputation, rare labels grouping, categorical variable encoding, numeric variable transformation and feature scaling
* Optimized Linear, Lasso, Decision Tree, Support Vector and Random Forest Regressors using GridSearchCV to reach the best model

## Code and Resources Used
**Python Version:** 3.8
**Packages:** scrapy, sklearn, pandas, numpy, scipy, matplotlib, seaborn

## Data Overview
* 2000 apartment postings are scraped from Zillow from 4 large US cities (500 each): Dallas, Seattle, Los Angeles, San Francisco
* After merging, the 2000-record dataset includes 8 variables
* price is the target variable, while others are predictors

![](images/df_head.png)

## Data Preprocessing
* 
