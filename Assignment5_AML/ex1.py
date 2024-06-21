import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import numpy as np
# بارگذاری داده‌ها
data = pd.read_csv('D:/pyton1/Assignment5_AML/Assignment5_AML/Country-data.csv')

# محاسبه ماتریس همبستگی
correlation_matrix = data.corr()

# رسم ماتریس همبستگی
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix of Features")
plt.show()


# ویژگی‌هایی که نیاز به نرمال‌سازی دارند (بدون ستون 'country')
features = data.columns[1:]

# مقدار دهی اولیه اسکیلر
scaler = StandardScaler()

# نرمال‌سازی ویژگی‌ها
data_scaled = data.copy()
data_scaled[features] = scaler.fit_transform(data[features])

# نمایش چند سطر اول داده‌های نرمال شده
print(data_scaled.head())


# روش Elbow
sse = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled[features])
    sse.append(kmeans.inertia_)

# رسم نتایج روش Elbow
plt.figure(figsize=(10, 6))
plt.plot(k_range, sse, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('SSE')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.show()

# روش امتیاز Silhouette
sil_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(data_scaled[features])
    sil_score = silhouette_score(data_scaled[features], labels)
    sil_scores.append(sil_score)

# رسم نتایج امتیاز Silhouette
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), sil_scores, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Optimal Number of Clusters')
plt.show()

# تعیین تعداد بهینه خوشه‌ها براساس روش Elbow و Silhouette
optimal_clusters_elbow = 4  # مقدار مثال از نمودار Elbow
optimal_clusters_silhouette = sil_scores.index(max(sil_scores)) + 2
optimal_clusters = optimal_clusters_elbow if optimal_clusters_elbow == optimal_clusters_silhouette else optimal_clusters_silhouette

print(f'Optimal number of clusters (Elbow): {optimal_clusters_elbow}')
print(f'Optimal number of clusters (Silhouette): {optimal_clusters_silhouette}')
print(f'Final optimal number of clusters: {optimal_clusters}')
# اجرای k-means با تعداد بهینه خوشه‌ها
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
data_scaled['Cluster'] = kmeans.fit_predict(data_scaled[features])

# نمایش چند سطر اول با برچسب خوشه
data_scaled[['country', 'Cluster']].head()
# انتخاب سه ویژگی برای بصری‌سازی
selected_features = ['child_mort', 'income', 'gdpp']

# رسم نمودارهای پراکندگی برای ویژگی‌های انتخاب شده
sns.pairplot(data_scaled, vars=selected_features, hue='Cluster', palette='Set1', diag_kind='kde')
plt.suptitle('Clusters Visualization with Selected Features', y=1.02)
plt.show()


# مقداردهی اولیه PCA
pca = PCA()

# اعمال PCA بر روی داده‌های نرمال شده
pca.fit(data_scaled[features])

# محاسبه واریانس توضیح داده شده
explained_variance = pca.explained_variance_ratio_

# رسم واریانس توضیح داده شده
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance.cumsum(), marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Principal Components')
plt.show()

# تعیین تعداد مؤلفه‌های اصلی که حداقل ۹۰٪ واریانس را توضیح می‌دهند
n_components = next(i for i, total_variance in enumerate(explained_variance.cumsum()) if total_variance >= 0.9) + 1
print(f'Number of principal components to retain: {n_components}')
# کاهش ابعاد داده‌ها
pca = PCA(n_components=n_components)
reduced_data = pca.fit_transform(data_scaled[features])

# اجرای خوشه‌بندی k-means روی داده‌های کاهش یافته
kmeans_reduced = KMeans(n_clusters=optimal_clusters, random_state=42)
data_scaled['Cluster_PCA'] = kmeans_reduced.fit_predict(reduced_data)

# مقایسه نتایج خوشه‌بندی اصلی و مبتنی بر PCA
comparison = data_scaled[['country', 'Cluster', 'Cluster_PCA']]
print(comparison.head())

# بصری‌سازی خوشه‌ها در ابعاد کاهش یافته
plt.figure(figsize=(10, 6))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=data_scaled['Cluster_PCA'], cmap='Set1', marker='o')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Clusters Visualization in Reduced Dimensions')
plt.show()
np.savetxt('D:/pyton1/Assignment5_AML/Assignment5_AML/output.csv', reduced_data, delimiter=',', fmt='%d')
