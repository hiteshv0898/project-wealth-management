import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
import logging

# Detect Anomalies using Z-Score
def detect_zscore_anomalies(actual, predicted, threshold=3.0):
    """
    Detect anomalies using Z-Score method. Anomalies are points with Z-Score > threshold.
    """
    residuals = actual - predicted
    scaler = StandardScaler()
    residuals_scaled = scaler.fit_transform(residuals.reshape(-1, 1))

    anomalies = np.where(np.abs(residuals_scaled) > threshold)[0]

    if len(anomalies) > 0:
        logging.info(f"⚠️ Z-Score detected {len(anomalies)} anomalies.")
    else:
        logging.info("⚠️ Z-Score detected 0 anomalies.")

    return anomalies

# Detect Anomalies using Isolation Forest
def detect_isolation_forest_anomalies(actual, predicted):
    """
    Detect anomalies using Isolation Forest algorithm.
    """
    residuals = actual - predicted
    isolation_forest = IsolationForest(contamination=0.05)  # Contamination is the fraction of outliers
    anomalies = isolation_forest.fit_predict(residuals.reshape(-1, 1))

    anomalies = np.where(anomalies == -1)[0]

    if len(anomalies) > 0:
        logging.info(f"⚠️ Isolation Forest detected {len(anomalies)} anomalies.")
    else:
        logging.info("⚠️ Isolation Forest detected 0 anomalies.")

    return anomalies

# Detect Anomalies using DBSCAN
def detect_dbscan_anomalies(actual, predicted, eps=0.5, min_samples=5):
    """
    Detect anomalies using DBSCAN (Density-Based Spatial Clustering of Applications with Noise).
    """
    residuals = actual - predicted
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(residuals.reshape(-1, 1))

    # -1 means the point is considered as noise
    anomalies = np.where(labels == -1)[0]

    if len(anomalies) > 0:
        logging.info(f"⚠️ DBSCAN detected {len(anomalies)} anomalies.")
    else:
        logging.info("⚠️ DBSCAN detected 0 anomalies.")

    return anomalies