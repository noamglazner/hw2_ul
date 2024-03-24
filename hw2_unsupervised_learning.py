import numpy as np
import torch
from sklearn.cluster import KMeans
from collections import Counter
import math
from sklearn.metrics.cluster import normalized_mutual_info_score
import numpy as np
from sklearn.datasets import fetch_openml

from sklearn.neighbors import NearestNeighbors


from scipy.ndimage import laplace

from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

from scipy.optimize import linear_sum_assignment

# Define the path to the dataset
file_train_path = 'datasets/mnist_background_random/mnist_background_random_train.amat'
#d
# Load the dataset as a text file

def load_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    images_data = []
    labels = []
    for line in lines:
        # Split the line by whitespace and convert to float
        data = [float(x) for x in line.split()]
        # The first element is the label
        label = int(data[28*28])
        labels.append(label)
        # The rest is image data
        image_data = data[:28*28]
        # Append the image data to the list
        images_data.append(image_data)
    return images_data, labels

def kmeans(image_np, n_clusters=2):
# Perform k-means clustering
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(images_np)
    # Get the cluster labels assigned by k-means
    cluster_labels = kmeans.labels_
    for i in range(len(cluster_labels)):
        if cluster_labels[i] == 0:
            cluster_labels[i] = 8
        else:
            cluster_labels[i] = 3
    # Convert labels_tensor_train to a numpy array
    ground_truth_labels_np = labels_tensor_train.numpy()
    # Calculate accuracy
    accuracy = calculate_accuracy(cluster_labels, ground_truth_labels_np)
    NMI = normalized_mutual_info_score(ground_truth_labels_np, cluster_labels)
    return accuracy, NMI
# Calculate the accuracy by finding the best matching between cluster labels and ground truth labels
def calculate_accuracy(cluster_labels, ground_truth_labels):
    # Construct a contingency table
    accuracy=0
    for i in range(len(cluster_labels)):
        if cluster_labels[i] == ground_truth_labels[i]:
            accuracy += 1
    return max(1-accuracy / len(cluster_labels),  accuracy / len(cluster_labels))

def calculate_nmi(cluster_labels, ground_truth_labels ):
    # Calculate mutual information
    mutual_info = 0.0
    total_points = len(ground_truth_labels)
    ground_truth_counter = Counter(ground_truth_labels)
    cluster_counter = Counter(cluster_labels)
    joint_counter = Counter(zip(ground_truth_labels, cluster_labels))

    for (gt_label, cluster_label), count in joint_counter.items():
        mutual_info += (count / total_points) * math.log((count * total_points) / (ground_truth_counter[gt_label] * cluster_counter[cluster_label]))

    # Calculate entropy
    entropy_ground_truth = 0.0
    entropy_cluster = 0.0

    for label, count in ground_truth_counter.items():
        prob = count / total_points
        entropy_ground_truth -= prob * math.log(prob)

    for label, count in cluster_counter.items():
        prob = count / total_points
        entropy_cluster -= prob * math.log(prob)

    # Calculate NMI
    nmi = mutual_info / math.sqrt(entropy_ground_truth * entropy_cluster) if entropy_ground_truth * entropy_cluster != 0 else 0.0

    return nmi

def laplacian_score(X, k_neighbors=5):
    """
    Compute Laplacian scores for features in X.

    Parameters:
    X : array-like, shape (n_samples, n_features)
        Input data.
    k_neighbors : int, optional (default=5)
        Number of neighbors for constructing the k-nearest neighbor graph.

    Returns:
    scores : array, shape (n_features,)
        Laplacian scores for each feature.
    """
    sigma = 3
    n = len(X)
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            distance = np.linalg.norm(X[i] - X[j])
            W[i][j] = W[j][i] = np.exp(- distance ** 2 / (2 * sigma ** 2))

    # Compute the degree matrix
    D = np.diag(np.sum(W, axis=1))

    # Compute the Laplacian matrix
    L = D - W

    # Compute Laplacian scores
    eigenvalues, _ = np.linalg.eigh(L)
    laplacian_eigenvalues = np.sum(eigenvalues[:, np.newaxis], axis=1)
    normelized_features = normelize(X)
    scores = np.zeros(784)
    for i in range(X.shape[1]):
        b = np.reshape(normelized_features[:, i], (1, 2000))
        scores[i] = np.dot(np.dot(b , L) ,b.transpose())

    return scores
def important_features(X, scores, n_features=200):


    indices = np.argsort(scores)[::-1][:n_features]
    # cut the original matrix to the most important features
    X = X[:, indices]
    return X
#
#
# def laplacian_score_image(image_data, k_neighbors=5):
#     """
#     Compute Laplacian scores for pixels in an image.
#
#     Parameters:
#     image_data : array-like, shape (n_samples, n_features)
#         Input image data.
#     k_neighbors : int, optional (default=5)
#         Number of neighbors for constructing the k-nearest neighbor graph.
#
#     Returns:
#     scores : array, shape (n_features,)
#         Laplacian scores for each pixel.
#     """
#
#     # Flatten image data
#     n_samples, n_features = image_data.shape[0], np.prod(image_data.shape[1:])
#     flat_image_data = image_data.reshape(n_samples, -1)
#
#     # Compute Laplacian scores
#     scores = laplacian_score(flat_image_data, k_neighbors)
#
#     return scores.reshape(image_data.shape[1:])
#

# Load MNIST data


# Compute Laplacian scores for the example image

# Plot the original image



def normelize(X):
   print(X.shape)
   #first we calculate l2 for each feature to normelize the data
   sums = []
   for i in range(X.shape[1]):
          sum = 0
          for l in range(X.shape[0]):
              sum += X[l][i] ** 2
          sum = math.sqrt(sum)
          sums.append(sum)
          for l in range(X.shape[0]):
            X[l][i] = X[l][i] / sums[i]

   return X

   # return sums


# Load training data
images_train_data, labels_train = load_data(file_train_path)

# Convert the lists to numpy arrays
images_train_np = np.array(images_train_data)

# Filter out examples with labels 3 and 8
indices_3_8 = [i for i, label in enumerate(labels_train) if label == 3 or label == 8]
selected_indices = indices_3_8[:2000]  # Selecting 1000 examples each for labels 3 and 8
selected_images_train_np = images_train_np[selected_indices]
selected_labels_train = [labels_train[i] for i in selected_indices]

# Reshape the data to images
images_train = selected_images_train_np.reshape(-1, 28, 28)  # MNIST images are 28x28
# Normalize the pixel values to the range [0, 1]
images_train = images_train / 255.0
# Convert the images to PyTorch tensors
images_tensor_train = torch.tensor(images_train, dtype=torch.float32)
# Transform the tensor to the required format (optional)
# For example, if you want to add a channel dimension for compatibility with PyTorch conventions
images_tensor_train = images_tensor_train.unsqueeze(1)  # Add a channel dimension
# Convert labels to PyTorch tensors
labels_tensor_train = torch.tensor(selected_labels_train, dtype=torch.long)
# Print shape of the tensors
print("Shape of the train images tensor:", images_tensor_train.shape)
print("Shape of the train labels tensor:", labels_tensor_train.shape)
images_2d = images_tensor_train.view(images_tensor_train.size(0), -1)
# Convert the PyTorch tensor to a
images_np = images_2d.numpy()
#Part 1
# print the accuracy and NMI of the kmeans of  images_np
results = kmeans(images_np, 2)
print("NMI:", results[1])
print("Accuracy:", results[0])

# Part 2
scores = laplacian_score(images_np)
scores_reshape = scores.reshape(28, 28)

plt.imshow(scores_reshape, cmap='gray')
plt.title("Original Image")
plt.show()
#Part 3
new_features = important_features(images_np, scores, 200)
# print the accuracy and NMI of the kmeans of the new features
results = kmeans(new_features, 2)
print("NMI:", results[1])
print("Accuracy:", results[0])
print(scores_reshape.shape)
