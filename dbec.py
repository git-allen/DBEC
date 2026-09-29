import numpy as np
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import math


def entropy(label_indexes, embeddings_labels, min_points=3):
    """Finds the entropy for a given set of labels

    Args:
        label_indexes (array): Array of embeddings
        embeddings_labels (array): the distance metric to use
        min_points (number): the number of min points to be considered for entropy

    Returns:
        number: entropy score (-1 if not validated because of min points)
    """

    # if number of points in the group is less than min points
    # do not consider entropy as high
    if len(label_indexes) < min_points:
        return -1
    
    test_labels = []
    for i in label_indexes:
        # if more than one label for an array consider all labels
        if (isinstance(embeddings_labels[i], list)):
            test_labels.extend(embeddings_labels[i])
        else:
            test_labels.append(embeddings_labels[i])
    n_labels = len(test_labels)

    counts = Counter(test_labels).values()
    counts = np.array((list(counts)))
    probs = counts / n_labels
    n_classes = len(probs)

    # all points are the same label
    if n_classes <= 1:
        return 0
    ent = - np.sum(probs * np.log2(probs))

    return ent

def embeddings_within_epsilon(embeddings, embedding, epsilon, metric='euclidean'):
    """Finds the list of embeddings within the epsilon of a given embedding

    Args:
        embeddings (array): Array of embeddings
        embedding: the specific embedding to find neighbors for
        epsilon (number): distance value
        metric (str): the distance metric to use

    Returns:
        array: array of indices of embeddings within epsilon
    """
    if metric == 'cosine':
        similarities = cosine_similarity(embedding.reshape(1, -1), embeddings)
        distances = 1 - similarities  # Convert similarity to distance
        dist_within_eps = np.where(distances <= epsilon)[1]
    else:
        distances = np.linalg.norm(embeddings - embedding, axis=1)
        dist_within_eps = np.where(distances <= epsilon)[0]
    return dist_within_eps

def get_all_neighbors_within_epsilon(embeddings, epsilon, metric='euclidean'):
    """Precomputes all neighbors within epsilon to save redundant recalculations"""
    N = len(embeddings)
    neighbors = []
    # To guarantee exact bit-for-bit results, we reuse the exact original calculation
    for i in range(N):
        neighbors.append(embeddings_within_epsilon(embeddings, embeddings[i], epsilon, metric))
    return neighbors

def mean_nn_distance(embeddings, neighbor=8, metric='euclidean'):
    """Finds the mean average distances

    Args:
        embeddings (array): Array of embeddings
        neighbor (int): the nearest neighbor; defaults to 8
        metric (str): the distance metric to use

    Returns:
        number: numeric value representing the mean
    """
    N = len(embeddings)
    chunk_size = 2000
    eighth_nn_distances = np.zeros(N)
    
    # Chunking prevents O(N^2) memory crash on large datasets while preserving exact math
    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        dist_chunk = distance.cdist(embeddings[i:end], embeddings, metric=metric)
        eighth_nn_distances[i:end] = np.partition(dist_chunk, neighbor, axis=1)[:, neighbor-1]

    # Calculate the mean of the 8th nearest neighbors' distances
    mean_eighth_nn_distance = np.mean(eighth_nn_distances)
    return mean_eighth_nn_distance

def percentile_nn_distance(embeddings, neighbor=8, metric='euclidean', percentile=85):
    """Finds the epsilon distance using a percentile to reduce outliers."""
    N = len(embeddings)
    chunk_size = 2000
    kth_nn_distances = np.zeros(N)
    
    # Chunking prevents O(N^2) memory crash on large datasets while preserving exact math
    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        dist_chunk = distance.cdist(embeddings[i:end], embeddings, metric=metric)
        kth_nn_distances[i:end] = np.partition(dist_chunk, neighbor, axis=1)[:, neighbor-1]

    # Use a percentile (e.g., 85th) instead of mean. 
    # This ensures 85% of points have at least 'neighbor' points within epsilon.
    radius = np.percentile(kth_nn_distances, percentile)
    return radius


def entropy_per_embedding(embeddings, embeddings_labels, epsilon, min_points=8, metric='euclidean', precomputed_neighbors=None):
    """Finds the entropy for each embedding

    Args:
        embeddings (array): Array of embeddings
        embeddings_labels (array): Array of labels for each embedding
        epsilon (number): distance value
        min_points (number): the number of min points to be considered for entropy
        metric (str): default euclidean distance
        precomputed_neighbors (list): cached neighbor list to prevent redundant calculations

    Returns:
        dict: A dictionary with embedding entropy information. 
            point_count: within the epsilon
            point_entropy: the point entropy
    """
    point_count = []
    point_entropy = []
    for point_idx in range(len(embeddings)):
        if precomputed_neighbors is not None:
            dist_within_eps = precomputed_neighbors[point_idx]
        else:
            dist_within_eps = embeddings_within_epsilon(embeddings, embeddings[point_idx], epsilon, metric)
        point_count.append(len(dist_within_eps))
        point_entropy.append(entropy(dist_within_eps,embeddings_labels, min_points))

    return {
        "point_count": point_count,
        "point_entropy": point_entropy
    }


def mark_points_classification(point_entropy, threshold=.5):
    """Mark each point based on entropy either high (H) or low (L) or boarder (B)

    Args:
        point_entropy (array): Array of entropies
        threshold (number): the threshold that the entropy needs to be over to be considered high

    Returns:
        array: H or L or O, indicating if the point is considered high (H) or low (L) or Outlier (O)
    """
    point_entropy_high = []
    for point_idx in range(len(point_entropy)):
        if point_entropy[point_idx] > threshold:
            point_entropy_high.append('H')
        elif point_entropy[point_idx] != -1:
            point_entropy_high.append('L')
        else:
            point_entropy_high.append('O')

    return point_entropy_high


def find_core_points(embeddings, point_entropy, epsilon, min_samples=.5, metric='euclidean', core='H', precomputed_neighbors=None):
    """Mark core points, each point that has more than N high entropy points with norm distance

    Args:
        embeddings (array): Array of embeddings
        point_entropy (str): high (H) or low (L) or boarder (B)
        epsilon (number): distance value
        min_samples (number): min number of samples percentage
        metric (str): default euclidean distance
        core (str): default H for High for finding High or Low entropy
        precomputed_neighbors (list): cached neighbor list to prevent redundant calculations

    Returns:
        array: # Core (C) or Boarder (B) point
    """
    point_type = ['B'] * len(embeddings)  # Core or Board point
    for point_idx in range(len(embeddings)):
        if point_entropy[point_idx] == core:
            if precomputed_neighbors is not None:
                dist_within_eps = precomputed_neighbors[point_idx]
            else:
                dist_within_eps = embeddings_within_epsilon(embeddings, embeddings[point_idx], epsilon, metric)
            n_found = [x for x in dist_within_eps if point_entropy[x] == core]
            percent = len(n_found)/len(dist_within_eps)
            if percent >= min_samples:
                point_type[point_idx] = 'C'
    return point_type

def build_entropy_clusters(embeddings, point_type, epsilon, metric='euclidean', precomputed_neighbors=None):
    """build clusters based on core points

    Args:
        embeddings (array): Array of embeddings
        point_type (array): Core (C) or Boarder (B)
        epsilon (number): distance value
        metric (str): default euclidean distance
        precomputed_neighbors (list): cached neighbor list to prevent redundant calculations

    Returns:
        array: Cluster labels for each point in the dataset.
    """
    entropy_cluster  = [(-1,)] * len(embeddings)
    cluster_count = 0
    for point_idx in range(len(embeddings)):
        if point_type[point_idx] == 'C' and entropy_cluster[point_idx] == (-1,):
            cluster_count += 1
            entropy_cluster[point_idx] = (cluster_count,)
            
            if precomputed_neighbors is not None:
                dist_within_eps = precomputed_neighbors[point_idx]
            else:
                dist_within_eps = embeddings_within_epsilon(embeddings, embeddings[point_idx], epsilon, metric)
                
            neighbors = set(dist_within_eps) - {point_idx}
            while neighbors:
                neighbor_index = neighbors.pop()
                if entropy_cluster[neighbor_index] == (-1,):
                    entropy_cluster[neighbor_index] = (cluster_count,)
                    if point_type[neighbor_index] == 'C':
                        if precomputed_neighbors is not None:
                            dist_within_eps = precomputed_neighbors[neighbor_index]
                        else:
                            dist_within_eps = embeddings_within_epsilon(embeddings, embeddings[neighbor_index], epsilon, metric)
                            
                        new_neighbors = set(dist_within_eps) - {neighbor_index}
                        if (len(new_neighbors) > 0): neighbors.update(new_neighbors)
                elif cluster_count not in entropy_cluster[neighbor_index]:
                    entropy_cluster[neighbor_index] = entropy_cluster[neighbor_index] + (cluster_count,)
                
    return entropy_cluster


def run_entropy_clustering(embeddings, embeddings_labels, neighbor=8, min_points_entropy=8, min_samples_high=0.5, entropy_threshold=0.5, metric='euclidean', core='H'):
    """
    Runs the entire entropy-based clustering pipeline in the specified order.

    Args:
        embeddings (np.ndarray): The embeddings data.
        embeddings_labels (np.ndarray or list): The labels associated with the embeddings.
        neighbor (int): The number of nearest neighbors to consider.
        min_points_entropy (int): Minimum points to calculate entropy.
        min_samples_high (float): Minimum proportion of high entropy neighbors.
        entropy_threshold (float): Threshold for marking points as High/Low entropy.
        metric (str): Distance metric ('euclidean' or 'cosine').
        core (str): default H for High for finding High or Low entropy.

    Returns:
        np.ndarray: The cluster labels assigned to each embedding.
    """
    epsilon = mean_nn_distance(embeddings, neighbor, metric)
    precomputed_neighbors = get_all_neighbors_within_epsilon(embeddings, epsilon, metric)
    
    entropy_data = entropy_per_embedding(embeddings, embeddings_labels, epsilon, min_points_entropy, metric, precomputed_neighbors)
    point_entropy_high = mark_points_classification(entropy_data["point_entropy"], entropy_threshold)
    point_type = find_core_points(embeddings, point_entropy_high, epsilon, min_samples_high, metric, core, precomputed_neighbors)
    cluster_labels = build_entropy_clusters(embeddings, point_type, epsilon, metric, precomputed_neighbors)
    return np.array(cluster_labels, dtype=object)


def capture_entropy_run(embeddings, embeddings_labels, neighbor=8, min_points_entropy=8, min_samples=0.5, entropy_threshold=0.3, metric='euclidean', core='H'):
    """
    Runs the entire entropy-based clustering pipeline collecting analytics returned in an object.
    """
    eps = mean_nn_distance(embeddings, neighbor, metric)
    precomputed_neighbors = get_all_neighbors_within_epsilon(embeddings, eps, metric)
    
    entropies_list = entropy_per_embedding(embeddings, embeddings_labels, eps, min_points_entropy, metric, precomputed_neighbors)
    dataset_labels = set(embeddings_labels)
    number_labels = len(dataset_labels)
    max_entropy = math.log2(number_labels) if number_labels > 0 else 0
    fraction = number_labels * entropy_threshold
    entropy_boundary = math.log2(fraction) if fraction > 0 else 0

    point_entropy = mark_points_classification(entropies_list['point_entropy'], entropy_boundary)
    number_high = len([x for x in point_entropy if x == 'H'])
    number_low = len([x for x in point_entropy if x == 'L'])
    number_outlier = len([x for x in point_entropy if x == 'O'])

    point_type = find_core_points(embeddings, point_entropy, eps, min_samples, metric, core, precomputed_neighbors)
    number_core = len([x for x in point_type if x == 'C'])

    entropy_clusters = build_entropy_clusters(embeddings, point_type, eps, metric, precomputed_neighbors)
    clusters = list({value for item in entropy_clusters for value in item})
    clusters = [x for x in clusters if x != -1]
    num_clusters = len(clusters)

    cluster_sizes = Counter([item for tup in entropy_clusters for item in tup if item != -1])
    
    cluster_c_counts = Counter()
    cluster_b_counts = Counter()
    cluster_h_counts = Counter()
    cluster_l_counts = Counter()
    cluster_o_counts = Counter()
    
    for pt_type, pt_entropy, cluster_tup in zip(point_type, point_entropy, entropy_clusters):
        for cluster_id in cluster_tup:
            if cluster_id != -1:
                if pt_type == 'C': cluster_c_counts[cluster_id] += 1
                elif pt_type == 'B': cluster_b_counts[cluster_id] += 1
                
                if pt_entropy == 'H': cluster_h_counts[cluster_id] += 1
                elif pt_entropy == 'L': cluster_l_counts[cluster_id] += 1
                elif pt_entropy == 'O': cluster_o_counts[cluster_id] += 1

    return {
        'dataset_labels': dataset_labels,
        'max_entropy': max_entropy,
        'entropy_boundary': entropy_boundary,
        'number_low': number_low,
        'number_high': number_high,
        'number_core': number_core,
        'clusters': clusters,
        'num_clusters': num_clusters,
        'point_entropy': point_entropy,
        'point_type': point_type,
        'entropy_clusters': entropy_clusters,
        'cluster_sizes': dict(cluster_sizes),
        'cluster_c_counts': dict(cluster_c_counts),
        'cluster_b_counts': dict(cluster_b_counts),
        'cluster_h_counts': dict(cluster_h_counts),
        'cluster_l_counts': dict(cluster_l_counts),
        'cluster_o_counts': dict(cluster_o_counts)
    }


class DBEC:
    """
    Density-Based Entropy Clustering (DBEC)
    A clustering algorithm for finding areas of high entropy.
    """
    def __init__(self, neighbor=8, min_points_entropy=8, min_samples_high=0.5, entropy_threshold=0.5, metric='euclidean', core='H'):
        self.neighbor = neighbor
        self.min_points_entropy = min_points_entropy
        self.min_samples_high = min_samples_high
        self.entropy_threshold = entropy_threshold
        self.metric = metric
        self.core = core
        self.labels_ = None
        self.epsilon_ = None

    def fit(self, X, y):
        """
        Fit the DBEC model.
        
        Args:
            X (np.ndarray): The embeddings data.
            y (np.ndarray or list): The labels associated with the embeddings.
        """
        self.epsilon_ = mean_nn_distance(X, self.neighbor, self.metric)
        precomputed_neighbors = get_all_neighbors_within_epsilon(X, self.epsilon_, self.metric)
        
        entropy_data = entropy_per_embedding(X, y, self.epsilon_, self.min_points_entropy, self.metric, precomputed_neighbors)
        point_entropy_high = mark_points_classification(entropy_data["point_entropy"], self.entropy_threshold)
        point_type = find_core_points(X, point_entropy_high, self.epsilon_, self.min_samples_high, self.metric, self.core, precomputed_neighbors)
        cluster_labels = build_entropy_clusters(X, point_type, self.epsilon_, self.metric, precomputed_neighbors)
        self.labels_ = np.array(cluster_labels, dtype=object)
        return self

    def fit_predict(self, X, y):
        """
        Fit the DBEC model and return cluster labels.
        """
        self.fit(X, y)
        return self.labels_

    def get_run_metrics(self, X, y):
        """
        Calculates and returns comprehensive run metrics to evaluate the confidence 
        and composition of the resulting clusters.
        
        Args:
            X (np.ndarray): The embeddings data.
            y (np.ndarray or list): The labels associated with the embeddings.
            
        Returns:
            dict: Comprehensive cluster metrics (e.g. max_entropy, point breakdown, sizes).
        """
        return capture_entropy_run(
            X, y, 
            neighbor=self.neighbor, 
            min_points_entropy=self.min_points_entropy,
            min_samples=self.min_samples_high,
            entropy_threshold=self.entropy_threshold,
            metric=self.metric,
            core=self.core
        )
