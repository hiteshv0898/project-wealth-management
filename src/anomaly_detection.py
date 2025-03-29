import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore

def detect_anomalies():
    print("🔎 Performing Anomaly Detection...")

    np.random.seed(42)
    normal_data = np.random.normal(loc=100, scale=5, size=300)
    anomalies = np.random.normal(loc=130, scale=5, size=10)
    data_with_anomalies = np.concatenate([normal_data, anomalies])

    df = pd.DataFrame({'Price': data_with_anomalies})
    df['Z_Score'] = zscore(df['Price'])
    df['Z_Anomaly'] = (np.abs(df['Z_Score']) > 3).astype(int)

    isolation_forest = IsolationForest(contamination=0.02, random_state=42)
    df['IF_Anomaly'] = isolation_forest.fit_predict(df[['Price']])
    df['IF_Anomaly'] = df['IF_Anomaly'].apply(lambda x: 1 if x == -1 else 0)

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Price'], label='Price', color='blue')
    plt.scatter(df.index[df['Z_Anomaly'] == 1], df['Price'][df['Z_Anomaly'] == 1], color='red', label='Z-Score Anomaly', s=80)
    plt.scatter(df.index[df['IF_Anomaly'] == 1], df['Price'][df['IF_Anomaly'] == 1], color='orange', label='Isolation Forest Anomaly', s=80)
    plt.title('Anomaly Detection using Z-Score and Isolation Forest')
    plt.xlabel('Index')
    plt.ylabel('Price')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()