# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 13:25:45 2020

@author: Rafi2401
"""
#library
import pandas as pd #read/write spreadsheet
# =============================================================================
# import numpy as np #counting
# import seaborn as sns #plot
# import matplotlib.pyplot as plt #visualisasi
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import MinMaxScaler
# =============================================================================

#membaca file csv
air = pd.read_csv ('air_bnb.csv')
print(air)
air.head()
#air.info()

# =============================================================================
# #clustering K-means
# #menentukan variabel
# air_x = air.iloc[:, 6:8]
# air_x.head()
# 
# #ploting grafik
# sns.scatterplot(x="longitude", y="latitude", data=air, s=100, color="red", alpha = 0.5)
# 
# #array
# x_array = np.array(air_x)
# print(x_array)
# 
# #standarisasi
# scaler = MinMaxScaler()
# x_scaled = scaler.fit_transform(x_array)
# x_scaled
# 
# # Menentukan dan mengkonfigurasi fungsi kmeans
# kmeans = KMeans(n_clusters = 5, random_state=123)
# # Menentukan cluster dari data
# kmeans.fit(x_scaled)
# print(kmeans.cluster_centers_)
# 
# # Menampilkan hasil cluster
# print(kmeans.labels_)
# # Menambahkan kolom "cluster" dalam data frame air
# air["cluster"] = kmeans.labels_
# air.head()
# 
# fig, rf = plt.subplots()
# sct = rf.scatter(x_scaled[:,1], x_scaled[:,0], s = 100, 
#                  c = air.cluster, marker = "o", alpha = 0.5)
# centers = kmeans.cluster_centers_
# rf.scatter(centers[:,1], centers[:,0], c='red', s=200, alpha=0.5);
# plt.title("Hasil Klustering K-Means")
# plt.xlabel("Scaled Longitude")
# plt.ylabel("Scaled Latitude")
# plt.show()
# =============================================================================

#classification
#.....


# =============================================================================
# #import data ke python
# x = air.iloc[:, :-1].values
# y = air.iloc[:, 3].values
# =============================================================================

# =============================================================================
# #memproses data missing
# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer(missing_values= np.nan, strategy = 'most_frequent')
# imputer = imputer.fit(x[:, 1:3])
# x[:, 1:3] = imputer.transform(x[:, 1:3])
# =============================================================================

# =============================================================================
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# transformer = ColumnTransformer(
#         [('name', OneHotEncoder(), [0])],
#         remainder='estimator')
# x = np.array(transformer.fit_transform(x), dtype=np.float)
# 
# labelencoder_y = LabelEncoder()
# y = labelencoder_y.fit_transform(y)
# =============================================================================
