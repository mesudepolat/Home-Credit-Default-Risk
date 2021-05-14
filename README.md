# Home-Credit-Default-Risk

This repo includes the scripts of a classification project created to predict the default risks of customers in the Home Credit Default Risk competition in Kaggle.
https://www.kaggle.com/c/home-credit-default-risk

Reference notebook : https://www.kaggle.com/jsaguiar/lightgbm-with-simple-features

## 1.Loading Packages
* Pandas
* Numpy
* Matplotlib
* Seaborn
* Scikit-learn
* Lightgbm

## 2.Data
* application_{train|test}.csv

  * This is the main table, broken into two files for Train (with TARGET) and Test (without TARGET).
  * Static data for all applications. One row represents one loan in our data sample.

* bureau.csv

  * All client's previous credits provided by other financial institutions that were reported to Credit Bureau (for clients who have a loan in our sample).
  * For every loan in our sample, there are as many rows as number of credits the client had in Credit Bureau before the application date.

* bureau_balance.csv

  * Monthly balances of previous credits in Credit Bureau.
  * This table has one row for each month of history of every previous credit reported to Credit Bureau – i.e the table has (#loans in sample * # of relative previous credits * #   of months where we have some history observable for the previous credits) rows.

* POS_CASH_balance.csv

  * Monthly balance snapshots of previous POS (point of sales) and cash loans that the applicant had with Home Credit.
  * This table has one row for each month of history of every previous credit in Home Credit (consumer credit and cash loans) related to loans in our sample – i.e. the table has (#loans in sample * # of relative previous credits * # of months in which we have some history observable for the previous credits) rows.












## 3.Understanding The Data
## 4.Exploratory Data Analysis
## Model Building : Part I
## Model Building: Part II
