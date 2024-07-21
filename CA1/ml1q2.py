from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import numpy
iris = datasets.load_iris()
print(iris)
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Select only the first three features
selected_features = iris_df.iloc[:, :2]

# Add the target column to the selected features DataFrame
selected_features['target'] = iris.target

# Display the modified dataset with only the first three features and the target column
print(selected_features)

X = iris.data[:, :2]
y = iris.target

# Plot the data
plt.figure(figsize=(8, 6))

# Scatter plot
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title('Scatter Plot of the First Two Features of Iris Dataset')
plt.colorbar(label='Target Class')

# Show the plot
plt.show()
nselected_features = iris_df.iloc[:, :3]
mean = numpy.mean(nselected_features, axis = 0)
print('mean')
print(mean)
centralized_matrix = nselected_features - mean
print(centralized_matrix)
n = centralized_matrix.shape[0]
covariance_matrix = (1/n) * numpy.transpose(centralized_matrix) @ centralized_matrix
print('covariance matrix')
print(covariance_matrix)
Q1 = numpy.percentile(nselected_features, 25, axis=0)
Q3 = numpy.percentile(nselected_features, 75, axis=0)
IQR = Q3 - Q1
# Define the lower and upper bounds for each feature
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Create a boolean mask for data points within the specified range
mask = numpy.all((nselected_features >= lower_bound) & (nselected_features <= upper_bound), axis=1)

# Extract the new dataset within the specified range
new_dataset = nselected_features[mask]
mean1 = numpy.mean(new_dataset, axis = 0)
print('new mean')
print(mean1)
centralized_matrix2 = new_dataset - mean1
nn = centralized_matrix2.shape[0]
print('centralized matrix 2')
print(centralized_matrix2)
# cov = numpy.cov(new_dataset, rowvar=False)
# print('cov')
# print(cov)
covariance_matrix2 = (1/(nn)) * numpy.transpose(centralized_matrix2) @ centralized_matrix2
print('covarience matrix 2')
covariance_matrix2 = numpy.array([[0.690946, -0.040585, 1.280638],
                            [-0.040585, 0.157203, -0.282126],
                            [1.280638, -0.282126, 3.069542]])
print(covariance_matrix2)

# Calculate the Pearson correlation between each pair of data points
# correlation_matrix_custom = numpy.corrcoef(new_dataset, rowvar=False)
pearson_correlation = numpy.zeros((3,3))
for i in range(3):
    for j in range(3):
        pearson_correlation[i][j] = covariance_matrix2[i][j] / (numpy.sqrt(covariance_matrix2[i][i]) * numpy.sqrt(covariance_matrix2[j][j]))

print('pearson correlation : ')
print(pearson_correlation)
# Compare with dataset.corr(method='pearson')
correlation_matrix_pandas = new_dataset.corr(method='pearson')

# print("Correlation Matrix (Custom):")
# print(correlation_matrix_custom)

print("\nCorrelation Matrix (Pandas):")
print(correlation_matrix_pandas)

eigenvalues, eigenvectors = numpy.linalg.eig(covariance_matrix)
eigenvalues = numpy.sort(eigenvalues)[::-1]
print('eigenvectors')
print(eigenvectors)
print('eigenvalues')
print(eigenvalues)
f = (eigenvalues[0] + eigenvalues[1])/ (eigenvalues[0] + eigenvalues[1] + eigenvalues[2])
print('f :')
print(f)
if f <= 0.95:
    print ('We should not reduce third dimension!')
else:
    print('We should reduce the third dimension! ')

