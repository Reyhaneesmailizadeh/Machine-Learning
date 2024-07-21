import numpy as np
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import numpy

iris = datasets.load_iris()
print(iris)
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Select only the first four features
selected_features = iris_df.iloc[:, :4]
print('data')
print(selected_features)


def scaler_square_kernel(x_i, x_j):
    result = numpy.dot(numpy.array(x_i), numpy.array(x_j).T)
    return result ** 2


def matrix_square_kernel(data):
    n = data.shape[0]
    kernel_matrix = numpy.zeros((n, n))
    for i in range(n):
        for j in range(n):
            kernel_matrix[i, j] = scaler_square_kernel(np.array(data.iloc[i, :]), np.array(data.iloc[j, :]))

    return kernel_matrix


def phi_cal(landa, eigvec):
    phi = numpy.dot(numpy.sqrt(landa), eigvec)
    return phi


def approximate_zero(matrix, threshold):
    matrix[np.abs(matrix) < threshold] = 0
    return matrix

def find_smallest_r(eigenvalues, a):
    sum = eigenvalues[0]
    i = 1
    totalsum = np.sum(eigenvalues)
    while(sum < a * totalsum):
        sum = sum + eigenvalues[i]
        i += 1
    return i

my_square_kernel_matrix = matrix_square_kernel(selected_features)
eigenvalues, eigenvectors = numpy.linalg.eig(np.array(my_square_kernel_matrix))
print(numpy.dot(eigenvectors.T, eigenvectors))
print('eignvalues')
print(eigenvalues)
print('eigenvectors')
print(eigenvectors)
eigenvalues = approximate_zero(eigenvalues, 0.0000001)
eigenvectors = approximate_zero(eigenvectors, 0.0000001)
# Create a diagonal matrix with eigenvalues on the diagonal
eigenvalues_matrix = numpy.diag(eigenvalues)
print(eigenvalues_matrix.shape)
print("Eigenvalues Matrix:")
print(eigenvalues_matrix)

phi_vector = phi_cal(eigenvalues_matrix, eigenvectors.T)
print('phi_vectors')
print(phi_vector.shape)
print(phi_vector)

new_kernel = numpy.dot(phi_vector.T, phi_vector)
print('my_square_kernel_matrix')
print(my_square_kernel_matrix.shape)
print(my_square_kernel_matrix)
print('new_kernel')
print(new_kernel.shape)
print(new_kernel)

if numpy.array_equal(new_kernel, my_square_kernel_matrix):
    print("The matrices are equal.")
else:
    print("The matrices are not equal.")


centeralizedkernelmatrise = my_square_kernel_matrix - numpy.mean(my_square_kernel_matrix, axis = 0)
eigenvaluescen, eigenvectorscen = numpy.linalg.eig(centeralizedkernelmatrise)
print('eigenvectorscen')
print(eigenvectorscen.T)
print('transpose')
print(eigenvectorscen)
eigenvaluescen = approximate_zero(eigenvaluescen, 0.0000001)
eigenvectorscen = approximate_zero(eigenvectorscen, 0.0000001)
# sorted_indices = numpy.argsort(eigenvaluescen)[::-1]
# eigenvaluescen = eigenvaluescen[sorted_indices]
# eigenvectorscen = eigenvectorscen[:, sorted_indices]
n = eigenvaluescen.shape[0]
print(n)
landa = eigenvaluescen/n
print(landa)
print('landa')
print(eigenvaluescen.shape)
eigenvectorscen = eigenvectorscen.T / np.sqrt(eigenvaluescen[:, np.newaxis])
print('eigenvectorscen')
print(eigenvectorscen.shape)
print('eigenvaluescen')
print(eigenvaluescen)
r = find_smallest_r(landa, 0.90)
print('r : ')
print(r)
neweigenvectorscen = eigenvectorscen[:][:r]
final = neweigenvectorscen.T
print(neweigenvectorscen.shape)
analysis = np.sum(landa[:r]) / sum(landa)
print(analysis)
Am = final.T @ centeralizedkernelmatrise
print('A matris : ')
print(Am.shape)
print(Am)
