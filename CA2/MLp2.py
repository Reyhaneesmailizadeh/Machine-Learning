import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler
#part a

#Function to calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))

# Function to initialize centroids randomly
def initialize_centroids(data, k):
    indices = np.random.choice(len(data), k, replace=False)
    return data[indices]

# Function to assign data points to clusters
def assign_clusters(data, centroids):
    distances = np.array([np.array([euclidean_distance(point, centroid) for centroid in centroids]) for point in data])
    return np.argmin(distances, axis=1)

# Function to update centroids based on mean of assigned data points
def update_centroids(data, clusters, k):
    new_centroids = np.zeros((k, data.shape[1]))
    for i in range(k):
        cluster_points = data[clusters == i]
        if len(cluster_points) > 0:
            new_centroids[i] = np.mean(cluster_points, axis=0)
        else:
            new_centroids[i] = data[np.random.choice(len(data))]
    return new_centroids

# Function to perform k-means clustering
def kmeans(data, k, max_iterations=100):
    centroids = initialize_centroids(data, k)
    for _ in range(max_iterations):
        clusters = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, clusters, k)
        if np.array_equal(centroids, new_centroids):
            break
        centroids = new_centroids
    return clusters, centroids

# Load the dataset
data = pd.read_csv('/Users/reyhane/Downloads/driver-data.csv')

# Assuming your dataset has features like 'feature1', 'feature2', etc.
# Adjust the feature names according to your actual dataset
features = data[['mean_dist_day', 'mean_over_speed_perc']].values

# Standardize the features (optional, but can be helpful)
features = (features - features.mean(axis=0)) / features.std(axis=0)

# Apply KMeans clustering
k = 4
clusters, centroids = kmeans(features, k)

# Plot the clusters
plt.scatter(features[:, 0], features[:, 1], c=clusters, cmap='viridis', marker='o', edgecolor='black')
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X', label='Centroids')
plt.title('K-Means Clustering (k=4)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
#part b

def gaussian_kernel(data, gamma=1):
    return rbf_kernel(data, gamma=gamma)

# Standardize the features (optional, but can be helpful)
features_standardized = StandardScaler().fit_transform(features)

# Apply Gaussian kernel to the standardized features
gamma = 0.15  # You can adjust the gamma parameter based on your data
feature_space = gaussian_kernel(features_standardized, gamma)

# Apply KMeans clustering on the feature space
k = 4
kernel_clusters, kernel_centroids = kmeans(feature_space, k)

# Plot the clusters in the original feature space
plt.scatter(features_standardized[:, 0], features_standardized[:, 1], c=kernel_clusters, cmap='viridis', marker='o', edgecolor='black')
plt.scatter(kernel_centroids[:, 0], kernel_centroids[:, 1], s=300, c='red', marker='X', label='Centroids')
plt.title('K-Means Clustering in Gaussian Feature Space (k=4)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

#part c
# from scipy.stats import multivariate_normal
# import numpy as np



# # Extract only the desired features (columns [1] and [2])
# mydataa = data.iloc[:, [1, 2]].values
# print(mydataa)
# # Assuming 'data' is your input data
# from sklearn.preprocessing import MinMaxScaler

# # Create MinMaxScaler
# scaler = MinMaxScaler()

# # Fit and transform the data
# mydata = scaler.fit_transform(mydataa)

# # Print the normalized data
# print(mydata)
# myk = 4
# eps = 0.001
# n_features = mydata.shape[1]
# n_samples = mydata.shape[0]
# print(n_features)
# print(n_samples)
# def initialize_parameters(data, k):
#     n_features = data.shape[1]
#     cluster_prob = np.ones(k) / k
#     # Initialize variances as identity matrices
#     variances = np.tile(np.eye(n_features), (k, 1, 1))
#     average = np.random.randn(k, n_features)
#     return variances, average, cluster_prob


# def sumofdistances(averageold, averagecurrent, k):
#     sum = 0
#     for i in range(k):
#         distance = np.linalg.norm(averagecurrent[i] - averageold[i])
#         #print(distance)
#         sum += distance
#     return sum

# def cal_prob_data(datapoint, variance, average, cluster_prob, k):
#     total_sum = 0
#     for i in range(k):
#         multivariate_dist = multivariate_normal(mean=average[i], cov=variance[i])
#         pdf_value = multivariate_dist.pdf(datapoint)
#         total_sum += pdf_value * cluster_prob[i]
#     return total_sum

# def expectation_step(data, k, n_samples, average, cluster_prob, variance):
#     post_prob = np.zeros((k, n_samples))
#     for i in range(k):
#         # Create a multivariate normal distribution
#         multivariate_dist = multivariate_normal(mean=average[i], cov=variance[i])
#         print("average for cur_cluster:",average[i])
#         for j in range(n_samples):
#             # Extract the data point for the j-th sample
#             datapoint = data[j, :]  # Assuming data is a 2D array
#             #print("datapoint:", datapoint)
#             # Calculate the probability density function (PDF) for the data point
#             pdf_value = multivariate_dist.pdf(datapoint)
#             #print(f"PDF for cur_datapoint: {pdf_value}")
#             post_prob[i][j] = (pdf_value * cluster_prob[i]) / cal_prob_data(datapoint, variance, average, cluster_prob, k)
#     return post_prob

# def Maximization_step(data, post_prob, n_samples,k):
#     newvariance = np.zeros_like(old_variance)
#     newaverage = np.zeros_like(old_average)
#     newprob_cluster = np.zeros_like(old_cluster_prob)
#     for i in range(k):
#         sum1 = np.sum(post_prob[i][:, np.newaxis] * data, axis=0)
#         sum2 = np.sum(post_prob[i])
#         newaverage[i] = sum1 / sum2

#     for i in range(k):
#         sum2 = np.sum(post_prob[i])
#         sum3 = np.sum(post_prob[i] * np.linalg.norm(data - newaverage[i], axis=1))
#         newvariance[i] = sum3 / sum2
#         newprob_cluster[i] = sum2 / n_samples
        
#     print("new_prob_cluster:",newprob_cluster)
#     return newvariance, newaverage, newprob_cluster

# old_variance, old_average, old_cluster_prob = initialize_parameters(mydata, myk)  
# print("old_variance :", old_variance)
# print("old_average:", old_average)
# print("old_cluster_prob:", old_cluster_prob)
# post_prob = expectation_step(mydata, myk, n_samples, old_average, old_cluster_prob, old_variance)
# print("post_prob:",post_prob)
# cur_variance, cur_average, cur_cluster_prob = Maximization_step(mydata, post_prob, n_samples,myk)
# print("cur_variance:",cur_variance)

# sumofdist = sumofdistances(old_average, cur_average, myk)
# count = 0
# print("cur_average:",cur_average)
# print("sumofdist:",sumofdist)
# print("cur_cluster_prob:", cur_cluster_prob)
# while sumofdist >= eps:
#     count = count + 1
#     old_variance, old_average, old_cluster_prob = cur_variance, cur_average, cur_cluster_prob
#     post_prob = expectation_step(mydata, myk, n_samples, old_average, old_cluster_prob, old_variance)
#     cur_variance, cur_average, cur_cluster_prob = Maximization_step(old_average, mydata, post_prob, n_samples,myk)
#     sumofdist = sumofdistances(old_average, cur_average, myk)
#     print(count)

#q1 part d
    
from sklearn.metrics.pairwise import euclidean_distances

# Function to calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Function to compute the betacv index
def betacv(features, clusters, centroids):
    k = len(centroids)
    cluster_distances = np.zeros(k)
    for i in range(k):
        cluster_points = features[clusters == i]
        cluster_center = centroids[i]
        intra_cluster_distances = euclidean_distances(cluster_points, [cluster_center])
        avg_intra_cluster_distance = np.mean(intra_cluster_distances)
        cluster_distances[i] = avg_intra_cluster_distance

    betacv_index = 0
    for i in range(k):
        max_similarity = 0
        for j in range(k):
            if i != j:
                similarity = (cluster_distances[i] + cluster_distances[j]) / euclidean_distance(centroids[i], centroids[j])
                max_similarity = max(max_similarity, similarity)

        betacv_index += max_similarity

    betacv_index /= k
    return betacv_index


# Calculate betacv for KMeans
kmeans_betacv = betacv(features, clusters, centroids)
print(f'KMeans Betacv: {kmeans_betacv}')
 
#betacv for kernel kmeans

# Assuming 'feature_space' is the data after applying the Gaussian kernel
kernel_kmeans_betacv = betacv(feature_space, kernel_clusters, kernel_centroids)
print(f'Kernel KMeans Betacv: {kernel_kmeans_betacv}')




#q2 part a
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
def euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

def find_neighbors(data, target_point, eps):
    neighbors = []
    for i, point in enumerate(data):
        if euclidean_distance(target_point, point) < eps:
            neighbors.append(i)
    return neighbors

def dbscan(data, eps, min_samples):
    labels = np.full(len(data), -1)  # -1 represents noise
    cluster_id = 0

    for i, point in enumerate(data):
        if labels[i] != -1:  # Skip if already visited
            continue

        neighbors = find_neighbors(data, point, eps)

        if len(neighbors) < min_samples:
            labels[i] = -1  # Mark as noise
        else:
            cluster_id += 1
            labels[i] = cluster_id

            for neighbor in neighbors:
                if labels[neighbor] == -1:
                    labels[neighbor] = cluster_id
                    new_neighbors = find_neighbors(data, data[neighbor], eps)
                    if len(new_neighbors) >= min_samples:
                        neighbors.extend(new_neighbors)

    return labels

# Generate a dataset with two circles
n_samples = 100
circles_data, _ = make_circles(n_samples=n_samples, factor=0.3, noise=0.02, random_state=42)

# Apply our DBSCAN implementation
eps = 0.3
min_samples = 5
dbscan_labels = dbscan(circles_data, eps, min_samples)

# Plot the results
plt.scatter(circles_data[:, 0], circles_data[:, 1], c=dbscan_labels, cmap='viridis')
plt.title('DBSCAN Clustering (Custom Implementation)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
#q2 part b
import numpy as np
import matplotlib.pyplot as plt

def initialize_centers(data, k):
    indices = np.random.choice(len(data), k, replace=False)
    return data[indices]

def assign_to_clusters(data, centers):
    distances = np.linalg.norm(data[:, np.newaxis, :] - centers, axis=2)
    return np.argmin(distances, axis=1)

def update_centers(data, labels, k):
    new_centers = np.zeros((k, data.shape[1]))
    for i in range(k):
        cluster_points = data[labels == i]
        if len(cluster_points) > 0:
            new_centers[i] = np.mean(cluster_points, axis=0)
    return new_centers

def kmeans(data, k, max_iterations=100, tol=1e-6):
    centers = initialize_centers(data, k)

    for _ in range(max_iterations):
        # Assign points to clusters
        labels = assign_to_clusters(data, centers)

        # Update cluster centers
        new_centers = update_centers(data, labels, k)

        # Check for convergence
        if np.all(np.abs(new_centers - centers) < tol):
            break

        centers = new_centers

    return labels, centers

# Apply k-means clustering
k = 2
kmeans_labels, kmeans_centers = kmeans(circles_data, k)

# Plot the results
plt.scatter(circles_data[:, 0], circles_data[:, 1], c=kmeans_labels, cmap='viridis')
plt.scatter(kmeans_centers[:, 0], kmeans_centers[:, 1], c='red', marker='X', label='Centroids')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
# DBSCAN might be considered more suitable because it can
# handle non-convex shapes and automatically find clusters based
# on density. The synthetic dataset generated using make_circles has a circular structure,
# and DBSCAN is able to capture this structure without assuming a specific cluster shape. Furthermore, 
# k-means is considered a linear clustering algorithm. The reason for this is that 
# k-means operates based on the concept of centroids and minimization of the sum of 
# squared distances, which leads to the formation of spherical or convex clusters.


#q3

def I(x):
    if abs(x) <= 1:
        result = 1
    else: 
        result = 0
    return result

def kernel1(x):
    result = 0.5 * I(x)
    return result

def kernel2(x):
    result = (1 / np.sqrt(2 * np.pi)) * np.exp((-1 / (x**2)))
    return result

def kernel3(x):
    result = (3/4) * (1 - x**2) * I(x)
    return result

def estimateddensity(data,kernelfunction,datapoint,n,h):
    sum = 0
    for i in range(n):
        curdata = data.iloc[i,1]
        z = (datapoint - curdata)/h
        sum = sum + kernelfunction(z)
    result = (1/(n * h)) * sum
    return result


datanerve = pd.read_csv('/Users/reyhane/Downloads/nerve.csv')
print(datanerve)
datapoint = datanerve.iloc[1,1]
print( " datapoint:",datapoint)
n_samples3 = datanerve.shape[0]

for h in np.arange(0.1,0.6,0.1):
    print("current h ",h)
    estimatedk1 = estimateddensity(datanerve,kernel1,datapoint,n_samples3,h)
    estimatedk2 = estimateddensity(datanerve,kernel2,datapoint,n_samples3,h)
    estimatedk3 = estimateddensity(datanerve,kernel3,datapoint,n_samples3,h)
    print("estimated density kernel 1", estimatedk1)
    print("estimated density kernel 2", estimatedk2)
    print("estimated density kernel 3", estimatedk3)



