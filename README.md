# Home-Credit-Default-Risk

This repo includes the scripts of a classification project created to predict the default risks of customers in the Home Credit Default Risk competition in Kaggle.
https://www.kaggle.com/c/home-credit-default-risk

Reference notebook : https://www.kaggle.com/jsaguiar/lightgbm-with-simple-features

## 1.Loading Packages
* certifi==2020.12.5
* chardet==4.0.0
* cycler==0.10.0
* idna==2.10
* joblib==1.0.1
* kaggle==1.5.12
* kiwisolver==1.3.1
* lightgbm==2.3.0
* matplotlib==3.4.1
* numpy==1.20.2
* pandas==1.2.4
* Pillow==8.2.0
* pyparsing==2.4.7
* python-dateutil==2.8.1
* python-slugify==4.0.1
* pytz==2021.1
* requests==2.25.1
* scikit-learn==0.24.1
* scipy==1.6.2
* seaborn==0.11.1
* six==1.15.0
* text-unidecode==1.3
* threadpoolctl==2.1.0
* tqdm==4.60.0
* urllib3==1.26.4
* wincertstore==0.2

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

* credit_card_balance.csv

  * Monthly balance snapshots of previous credit cards that the applicant has with Home Credit.
  * This table has one row for each month of history of every previous credit in Home Credit (consumer credit and cash loans) related to loans in our sample – i.e. the table has (#loans in sample * # of relative previous credit cards * # of months where we have some history observable for the previous credit card) rows.

* previous_application.csv

  * All previous applications for Home Credit loans of clients who have loans in our sample.
  * There is one row for each previous application related to loans in our data sample.

* installments_payments.csv

  * Repayment history for the previously disbursed credits in Home Credit related to the loans in our sample.
  * There is a) one row for every payment that was made plus b) one row each for missed payment. One row is equivalent to one payment of one installment OR one installment corresponding to one payment of one previous Home Credit credit related to loans in our sample.

* HomeCredit_columns_description.csv

  * This file contains descriptions for the columns in the various data files.


## 3.Understanding The Data

![home_credit](https://user-images.githubusercontent.com/61362079/118275281-e8630c00-b4ce-11eb-93bb-4b9b115a5ded.png)

