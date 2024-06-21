import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('D:/pyton1/Assignment5_AML/Assignment5_AML/Country-data.csv')

# Drop the 'country' column and normalize the data
features = df.drop(columns=['country'])
scaler = MinMaxScaler()
normalized_features = scaler.fit_transform(features)

# Convert back to DataFrame
normalized_df = pd.DataFrame(normalized_features, columns=features.columns)

# Save normalized data for later use
df_normalized = df.copy()
df_normalized[features.columns] = normalized_features

# Use the Elbow method to determine the optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(normalized_df)
    wcss.append(kmeans.inertia_)

# Plot the WCSS values
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Determine the optimal number of clusters using the silhouette score
silhouette_scores = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    labels = kmeans.fit_predict(normalized_df)
    score = silhouette_score(normalized_df, labels)
    silhouette_scores.append(score)

# Plot the silhouette scores
plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Silhouette Score Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()

# Determine the optimal number of clusters
optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
print(f'Optimal number of clusters: {optimal_clusters}')
# Apply K-means with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(normalized_df)

# Save the final dataset with clusters
df.to_csv('final_dataset_with_clusters.csv', index=False)

# Display the cluster assignment
print(df[['country', 'Cluster']])


# Apply PCA
pca = PCA()
pca_features = pca.fit_transform(normalized_df)

# Convert to DataFrame
pca_df = pd.DataFrame(pca_features, columns=[f'PC{i+1}' for i in range(pca_features.shape[1])])

# Plot the explained variance ratio
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker='o')
plt.title('Explained Variance by Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.show()

# Calculate cumulative explained variance
cumulative_variance = pca.explained_variance_ratio_.cumsum()
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
plt.title('Cumulative Explained Variance by Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.axhline(y=0.95, color='r', linestyle='--')
plt.show()

# Retain components that explain at least 95% of the variance
n_components = np.argmax(cumulative_variance >= 0.95) + 1
reduced_pca_df = pca_df.iloc[:, :n_components]

# Apply K-means on the reduced data
kmeans_pca = KMeans(n_clusters=optimal_clusters, random_state=42)
df['PCA_Cluster'] = kmeans_pca.fit_predict(reduced_pca_df)

# Save the final dataset with PCA clusters
df.to_csv('final_dataset_with_pca_clusters.csv', index=False)

# Compare with previous clustering results
print(df[['country', 'Cluster', 'PCA_Cluster']])
# /////////////////////////////////////////////////////////////////////////////////////
# Load the final dataset from Problem 1
df = pd.read_csv('final_dataset_with_clusters.csv')

# Select features and target
X = df[['child_mort', 'income', 'life_expec']].values
y = df['Cluster'].values

# Convert the target to binary classes for simplicity (e.g., Cluster 0 and 1)
# If there are more than two clusters, consider merging or selecting two clusters
# Here we assume clusters 0 and 1 for binary classification
binary_mask = (y == 0) | (y == 1)
X_binary = X[binary_mask]
y_binary = y[binary_mask]

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_binary, y_binary, test_size=0.2, random_state=42)

# Training function for perceptron and adaline
def train_and_plot_errors(model, X_train, y_train, X_val, y_val, epochs=100):
    train_errors = []
    val_errors = []

    for epoch in range(epochs):
        model.partial_fit(X_train, y_train, classes=np.unique(y_train))
        train_errors.append(1 - model.score(X_train, y_train))
        val_errors.append(1 - model.score(X_val, y_val))

    return train_errors, val_errors

# Train perceptron
perceptron = Perceptron(max_iter=1, warm_start=True, random_state=42)
perceptron_train_errors, perceptron_val_errors = train_and_plot_errors(perceptron, X_train, y_train, X_val, y_val)

# Train adaline
adaline = SGDClassifier(loss='squared_error', max_iter=1, warm_start=True, random_state=42)
adaline_train_errors, adaline_val_errors = train_and_plot_errors(adaline, X_train, y_train, X_val, y_val)

# Plot training and validation errors
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(perceptron_train_errors, label='Training Error')
plt.plot(perceptron_val_errors, label='Validation Error')
plt.title('Perceptron Training and Validation Errors')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(adaline_train_errors, label='Training Error')
plt.plot(adaline_val_errors, label='Validation Error')
plt.title('Adaline Training and Validation Errors')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.legend()

plt.show()

# Final accuracy
print(f'Perceptron Final Accuracy: {1 - perceptron_val_errors[-1]:.2f}')
print(f'Adaline Final Accuracy: {1 - adaline_val_errors[-1]:.2f}')
