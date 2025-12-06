import numpy as np 
import pandas as pd 
from datetime import datetime 
import os 




def fetch_data(csv_file_path):
    """Fetch data from csv file"""
    try:
        #check if file exists
        if not os.path.exists(csv_file_path):
            print(f"csv file not found: {csv_file_path}")
            return None

        #Read csv file
        data = pd.read_csv(csv_file_path)
        print(f"Successfully loaded {len(data)} transactions from {csv_file_path}.")
        return data 

    except Exception as e:
        print(f"Error loading from csv: {str(e)}")
        return None





def preprocess_data(data):
    """Preprocess the transaction data with consistency checks."""
    if data is None:
        print("No data available. Please fetch data first.")
        return None

    print("Starting Data Processing...")
    df = data.copy()

    # Checking for duplicates
    total_duplicated = df.duplicated().sum()
    transaction_duplicates = df['TransactionID'].duplicated().sum()
    print(f"The total number of duplicates is : {total_duplicated}")
    print(f"The total transaction duplicates is : {transaction_duplicates}")
    
    # Drop duplicated transaction ID
    df = df.drop_duplicates(subset=['TransactionID'])

    # No of customers
    unique_customers = df['CustomerID'].nunique()
    print(f"Unique customers: {unique_customers}")

    # Handling missing values
    df = df.dropna()

    # Ensure proper datetime format
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'], errors = 'coerce')
    return df





def calculate_rfm_metrics(data):
    """Calculate RFM metrics"""
    if data is None:
        print("No data available. Please fetch and process data first.")
        return None

    reference_date = data['TransactionDate'].max() + pd.Timedelta(days=1)

    # Calculate RFM metrics for each customer
    rfm_df = data.groupby('CustomerID').agg({
        'TransactionDate': lambda x: (reference_date - x.max()).days,       #Recency
        'TransactionID': 'count',                                           #Frequency
        'TransactionAmount': 'sum'                                          #Monetary
    }).reset_index()

    rfm_df.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

    # Add customer Demographics
    customer_demographics = data.groupby('CustomerID').agg({
        'CustomerDOB': 'first',
        'CustGender': 'first',
        'CustLocation': 'first',
        'CustAccountBalance': 'last'
    }).reset_index()

    #Merging customer demographics to rfm_data
    rfm_df = rfm_df.merge(customer_demographics, on='CustomerID', how='left')

    return rfm_df