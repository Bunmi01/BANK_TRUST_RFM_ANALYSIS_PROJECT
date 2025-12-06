import numpy as np 
import pandas as pd 
import logging 
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()


def calculate_rfm_scores(rfm_data):

    if rfm_data is None:
        print("No RFM data available for scoring")
        return None

    print("Calculating RFM Scores and Segmentation...")

    #Create RFM scores (1-5 scale)
    rfm_data['R_Score'] = pd.qcut(rfm_data['Recency'], q=5, labels=[5, 4, 3, 2, 1])         #Recency: lower recency = better (more recent)
    rfm_data['F_Score'] = pd.qcut(rfm_data['Frequency'], q=5, labels=[1, 2, 3, 4, 5])       # Frequency: higher frequency = better
    rfm_data['M_Score'] = pd.qcut(rfm_data['Monetary'], q=5, labels=[1, 2, 3, 4, 5])        # Monetary: higher monetary = better

    #Converting the RFM to integers
    rfm_data[['R_Score', 'F_Score', 'M_Score']] = rfm_data[['R_Score', 'F_Score', 'M_Score']].astype(int)

    #Calculate RFM score
    rfm_data['RFM_Score'] = rfm_data['R_Score'] + rfm_data['F_Score'] + rfm_data['M_Score']
    rfm_data['RFM_Group'] = (
    rfm_data['R_Score'].astype(str) +
    rfm_data['F_Score'].astype(str) +
    rfm_data['M_Score'].astype(str)
    )

    print("RFM scoring and segmentation successfully completed. ")
    print(f"Average RFM Score: {rfm_data['RFM_Score'].mean():.2f}")

    return rfm_data






def prepare_for_clustering(rfm_data):
    """Preparing RFM Data for Clustering"""
    if rfm_data is None:
        print("No data is available for clustering preparation")
        return None

    print("Preparing RFM data for clustering")

    # Extract key columns
    rfm_clustering = rfm_data[['Recency', 'Frequency', 'Monetary']].copy()

    #Handle skewness with log transformation
    rfm_clustering['Recency'] = np.log1p(rfm_clustering['Recency'])
    rfm_clustering['Frequency'] = np.log1p(rfm_clustering['Frequency'])
    rfm_clustering['Monetary'] = np.log1p(rfm_clustering['Monetary'])

    #Standardize the features
    rfm_scaled = scaler.fit_transform(rfm_clustering)
    rfm_scaled_df = pd.DataFrame(rfm_scaled, columns=['Recency', 'Frequency', 'Monetary'])

    print("RFM data successfully scaled and transformed for clustering. ")
    return rfm_scaled_df
