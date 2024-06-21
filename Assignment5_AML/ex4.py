import requests
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO
import requests
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


# بارگیری دیتاست اعداد دست‌نویس
digits = load_digits()

# نمایش ساختار دیتاست و یک تصویر از هر عدد
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(digits.images[i], cmap='gray')
    plt.title(f"Digit {i}")
    plt.axis('off')
plt.show()

# پیش‌پردازش دیتاست
scaler = StandardScaler()
data_scaled = scaler.fit_transform(digits.data)

# Initialize KMeans model
kmeans = KMeans(n_clusters=10, random_state=42)

# Fit KMeans model to the data
kmeans.fit(digits.data)

# Get cluster labels
kmeans_labels = kmeans.labels_

# Plot one example image for each digit with assigned KMeans cluster label
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    # Find an image with the current KMeans label i
    idx = np.where(kmeans_labels == i)[0][0]
    ax.imshow(digits.images[idx], cmap='binary')
    ax.set_title(f"Cluster {i}")
    ax.axis('off')

plt.tight_layout()
plt.show()


# part d
# Initialize Hierarchical clustering model
hierarchical = AgglomerativeClustering(n_clusters=10)

# Fit Hierarchical clustering model to the data
hierarchical.fit(digits.data)

# Get cluster labels
hierarchical_labels = hierarchical.labels_

# Plot one example image for each digit with assigned Hierarchical clustering label
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    # Find an image with the current Hierarchical label i
    idx = np.where(hierarchical_labels == i)[0][0]
    ax.imshow(digits.images[idx], cmap='binary')
    ax.set_title(f"Cluster {i}")
    ax.axis('off')

plt.tight_layout()
plt.show()

# part e

# Evaluate KMeans clustering
kmeans_silhouette = silhouette_score(digits.data, kmeans_labels)
print(f"Silhouette Score for KMeans: {kmeans_silhouette}")

# Evaluate Hierarchical clustering
hierarchical_silhouette = silhouette_score(digits.data, hierarchical_labels)
print(f"Silhouette Score for Hierarchical Clustering: {hierarchical_silhouette}")

# part f

# Initialize arrays to hold predicted labels for clusters
kmeans_predicted_labels = np.zeros_like(kmeans_labels)
hierarchical_predicted_labels = np.zeros_like(hierarchical_labels)

# Assign label to each cluster for KMeans
for i in range(10):  
    mask = (kmeans_labels == i)
    kmeans_predicted_labels[mask] = np.bincount(digits.target[mask]).argmax()

# Assign label to each cluster for Hierarchical clustering
for i in range(10):  
    mask = (hierarchical_labels == i)
    hierarchical_predicted_labels[mask] = np.bincount(digits.target[mask]).argmax()

# Print predicted labels
print(f"Predicted labels for KMeans clusters: {kmeans_predicted_labels}")
print(f"Predicted labels for Hierarchical clusters: {hierarchical_predicted_labels}")
