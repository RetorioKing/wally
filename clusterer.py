from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd

def run_kmeans(X, n_clusters):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(X_scaled)
    return labels, kmeans.inertia_, kmeans
