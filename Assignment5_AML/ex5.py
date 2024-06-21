import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
# Load datasets
dataset1 = np.genfromtxt('D:/pyton1/Assignment5_AML/Assignment5_AML/Datasets/Dataset1.txt', delimiter=',', skip_header=1)
dataset2 = np.genfromtxt('D:/pyton1/Assignment5_AML/Assignment5_AML/Datasets/Dataset2.txt', delimiter=',', skip_header=1)

# Plot Dataset1
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.scatter(dataset1[:, 0], dataset1[:, 1])
plt.title('Dataset 1')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Plot Dataset2
plt.subplot(1, 2, 2)
plt.scatter(dataset2[:, 0], dataset2[:, 1])
plt.title('Dataset 2')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()


# part b

# آموزش KMeans بر روی Dataset1
kmeans1 = KMeans(n_clusters=3, random_state=42)
kmeans1.fit(dataset1)
labels1 = kmeans1.labels_

# آموزش KMeans بر روی Dataset2
kmeans2 = KMeans(n_clusters=2, random_state=42)
kmeans2.fit(dataset2)
labels2 = kmeans2.labels_

# ترسیم نتایج KMeans برای Dataset1
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.scatter(dataset1[:, 0], dataset1[:, 1], c=labels1)
plt.title('کلاسترینگ KMeans بر روی Dataset 1')

# ترسیم نتایج KMeans برای Dataset2
plt.subplot(1, 2, 2)
plt.scatter(dataset2[:, 0], dataset2[:, 1], c=labels2)
plt.title('کلاسترینگ KMeans بر روی Dataset 2')

plt.tight_layout()
plt.show()

# part c

# ارزیابی نمره سیلوئت
silhouette_score1 = silhouette_score(dataset1, labels1)
silhouette_score2 = silhouette_score(dataset2, labels2)

print(f'نمره سیلوئت برای Dataset 1: {silhouette_score1}')
print(f'نمره سیلوئت برای Dataset 2: {silhouette_score2}')

# روش Elbow برای Dataset1
sse1 = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(dataset1)
    sse1.append(kmeans.inertia_)

# روش Elbow برای Dataset2
sse2 = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(dataset2)
    sse2.append(kmeans.inertia_)

# ترسیم نتایج روش Elbow
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, 10), sse1, marker='o')
plt.title('روش Elbow برای Dataset 1')
plt.xlabel('تعداد کلاسترها')
plt.ylabel('SSE')

plt.subplot(1, 2, 2)
plt.plot(range(1, 10), sse2, marker='o')
plt.title('روش Elbow برای Dataset 2')
plt.xlabel('تعداد کلاسترها')
plt.ylabel('SSE')

plt.tight_layout()
plt.show()
# در این روش هم مشخصه دیتاست یک با سه کلاستر نتیجه خوبی میدهد اما برای دیتاست دو نمیتوان مشخص کرد.
#part d

# آموزش DBSCAN بر روی Dataset1
dbscan1 = DBSCAN(eps=0.5, min_samples=5)
dbscan1.fit(dataset1)
labels_dbscan1 = dbscan1.labels_

# آموزش DBSCAN بر روی Dataset2
dbscan2 = DBSCAN(eps=0.3, min_samples=5)
dbscan2.fit(dataset2)
labels_dbscan2 = dbscan2.labels_

# ترسیم نتایج DBSCAN برای Dataset1
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.scatter(dataset1[:, 0], dataset1[:, 1], c=labels_dbscan1)
plt.title('کلاسترینگ DBSCAN بر روی Dataset 1')

# ترسیم نتایج DBSCAN برای Dataset2
plt.subplot(1, 2, 2)
plt.scatter(dataset2[:, 0], dataset2[:, 1], c=labels_dbscan2)
plt.title('کلاسترینگ DBSCAN بر روی Dataset 2')

plt.tight_layout()
plt.show()

# part e
# ارزیابی نمره سیلوئت برای DBSCAN
silhouette_score_dbscan1 = silhouette_score(dataset1, labels_dbscan1)
silhouette_score_dbscan2 = silhouette_score(dataset2, labels_dbscan2)

print(f'نمره سیلوئت برای DBSCAN بر روی Dataset 1: {silhouette_score_dbscan1}')
print(f'نمره سیلوئت برای DBSCAN بر روی Dataset 2: {silhouette_score_dbscan2}')


# part f
""" مراحل:

پیش‌بینی برچسب‌های کلاستر: از مدل آموزش دیده برای پیش‌بینی برچسب‌های کلاستر برای هر نقطه داده در دیتاست خود استفاده کنید.

محاسبه فاصله: فاصله بین هر نقطه داده و مرکز کلاستری که به آن تخصیص یافته است را محاسبه کنید. این فاصله می‌تواند فاصله اقلیدوسی، فاصله منهتنی یا هر فاصله‌سنجی مناسب دیگری باشد.

تعیین آستانه برای نقاط دورافتاده: یک مقدار آستانه برای فواصل تعیین کنید. نقاط داده که فاصله آن‌ها از آستانه بیشتر باشد می‌توانند به عنوان نقاط دورافتاده در نظر گرفته شوند.

شناسایی نقاط دورافتاده: نقاط داده که فاصله آن‌ها از آستانه بیشتر است را شناسایی کنید. این نقاط داده به عنوان نقاط دورافتاده در نظر گرفته می‌شوند.
"""
# بارگذاری Dataset3
dataset3 = np.genfromtxt('D:/pyton1/Assignment5_AML/Assignment5_AML/Datasets/Dataset3.txt', delimiter=',', skip_header=1)

# ترسیم Dataset3
plt.scatter(dataset3[:, 0], dataset3[:, 1])
plt.title('Dataset 3')
plt.xlabel('ویژگی 1')
plt.ylabel('ویژگی 2')
plt.show()

# شناسایی نقاط دورافتاده با استفاده از KMeans
kmeans3 = KMeans(n_clusters=3, random_state=42)
kmeans3.fit(dataset3)
labels3 = kmeans3.labels_

# محاسبه فاصله‌ها تا نزدیکترین مرکز کلاستر
distances = np.min(kmeans3.transform(dataset3), axis=1)

# تعریف نقاط دورافتاده به عنوان نقاطی که فاصله‌شان بیشتر از یک آستانه است
threshold = np.percentile(distances, 97)
outliers = distances > threshold

# ترسیم نتایج با نشان دادن نقاط دورافتاده با رنگی متفاوت
plt.scatter(dataset3[:, 0], dataset3[:, 1], c='b', label='نقاط داده')
plt.scatter(dataset3[outliers, 0], dataset3[outliers, 1], c='r', label='نقاط دورافتاده')
plt.title('شناسایی نقاط دورافتاده با KMeans بر روی Dataset 3')
plt.xlabel('ویژگی 1')
plt.ylabel('ویژگی 2')
plt.legend()
plt.show()