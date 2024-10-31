# Neural Network - Predicting Customer Churn in Telecom Industry
## Project Description
In this project, I train a Neural Network in Pytorch to predict churn of customers. 
* Import Data
* Explorative Data Analysis
* Data Preparation
  * Join Tables
  * Transform Datatype
  * One Hot Encoding
  * Train Test Split
  * Standardize Numerical Features
  * Convert df to tensors
* Model Training
  * Define Neural Network Structure
  * Define Loss and Optimizer
  * Training Loop
* Hyperparameter Tuning
  * Log loss during training loop and visualize using matplotlib.
  * Adjust hyperparameters like batch_size, epochs, learning rate, and momentum.
* Testing
  * 2024-10-29: 78% 
* Next Step:
  * Add a validation set.
  * Considering early stoppage.
  * Considering other optimizer.
  * Considering more feature engineering:
    * How to properly deal with timestamp (seansonality?)
    * How to include category features like state that has ~40 distinct values. 

## Context
The telecommunications (telecom) sector in India is rapidly changing, with more and more telecom businesses being created and many customers deciding to switch between providers. "Churn" refers to the process where customers or subscribers stop using a company's services or products. Understanding the factors that influence keeping a customer as a client in predicting churn is crucial for telecom companies to enhance their service quality and customer satisfaction. As the data scientist on this project, you aim to explore the intricate dynamics of customer behavior and demographics in the Indian telecom sector in predicting customer churn, utilizing two comprehensive datasets from four major telecom partners: Airtel, Reliance Jio, Vodafone, and BSNL:

## Data
- `telecom_demographics.csv` contains information related to Indian customer demographics:

| Variable             | Description                                      |
|----------------------|--------------------------------------------------|
| `customer_id `         | Unique identifier for each customer.             |
| `telecom_partner `     | The telecom partner associated with the customer.|
| `gender `              | The gender of the customer.                      |
| `age `                 | The age of the customer.                         |
| `state`                | The Indian state in which the customer is located.|
| `city`                 | The city in which the customer is located.       |
| `pincode`              | The pincode of the customer's location.          |
| `registration_event` | When the customer registered with the telecom partner.|
| `num_dependents`      | The number of dependents (e.g., children) the customer has.|
| `estimated_salary`     | The customer's estimated salary.                 |

- `telecom_usage` contains information about the usage patterns of Indian customers:

| Variable   | Description                                                  |
|------------|--------------------------------------------------------------|
| `customer_id` | Unique identifier for each customer.                         |
| `calls_made` | The number of calls made by the customer.                    |
| `sms_sent`   | The number of SMS messages sent by the customer.             |
| `data_used`  | The amount of data used by the customer.                     |
| `churn`    | Binary variable indicating whether the customer has churned or not (1 = churned, 0 = not churned).|

