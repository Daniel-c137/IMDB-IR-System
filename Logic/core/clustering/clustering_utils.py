import matplotlib.pyplot as plt
import numpy as np
import random
import operator
import wandb
import math

from typing import List, Tuple
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial import distance
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from collections import Counter
from clustering_metrics import *


class ClusteringUtils:

    def cluster_kmeans(self, emb_vecs: List, n_clusters: int, max_iter: int = 100) -> Tuple[List, List]:
        """
        Clusters input vectors using the K-means method.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.
        n_clusters: int
            The number of clusters to form.

        Returns
        --------
        Tuple[List, List]
            Two lists:
            1. A list containing the cluster centers.
            2. A list containing the cluster index for each input vector.
        """
        v_size = len(emb_vecs[0])
        centers = random.sample(np.unique(emb_vecs, axis=0).tolist(), n_clusters)
        clusters = [None for _ in emb_vecs]
        for _ in range(max_iter):
            # assigning vectors to indices
            for i, v in enumerate(emb_vecs):
                min_idx = None
                min_dist = math.inf
                for j, c in enumerate(centers):
                    d = distance.euclidean(c, v)
                    if min_dist > d:
                        min_idx = j
                        min_dist = d
                clusters[i] = min_idx
            s = [[0 for _ in range(v_size)] for _ in centers]
            count = [0 for _ in centers]
            for i, v in enumerate(emb_vecs):
                s[clusters[i]] = list(map(operator.add, s[clusters[i]], v))
                count[clusters[i]] += 1
            try:
                for i in range(n_clusters):
                    centers[i] = [s[i][j] / count[i] for j in  range(v_size)]
            except:
                return self.cluster_kmeans(emb_vecs, n_clusters, max_iter)
        return centers, clusters
                

    def get_most_frequent_words(self, documents: List[str], top_n: int = 10) -> List[Tuple[str, int]]:
        """
        Finds the most frequent words in a list of documents.

        Parameters
        -----------
        documents: List[str]
            A list of documents, where each document is a string representing a list of words.
        top_n: int, optional
            The number of most frequent words to return. Default is 10.

        Returns
        --------
        List[Tuple[str, int]]
            A list of tuples, where each tuple contains a word and its frequency, sorted in descending order of frequency.
        """
        pass

    def cluster_kmeans_WCSS(self, emb_vecs: List, n_clusters: int) -> Tuple[List, List, float]:
        """ This function performs K-means clustering on a list of input vectors and calculates the Within-Cluster Sum of Squares (WCSS) for the resulting clusters.

        This function implements the K-means algorithm and returns the cluster centroids, cluster assignments for each input vector, and the WCSS value.

        The WCSS is a measure of the compactness of the clustering, and it is calculated as the sum of squared distances between each data point and its assigned cluster centroid. A lower WCSS value indicates that the data points are closer to their respective cluster centroids, suggesting a more compact and well-defined clustering.

        The K-means algorithm works by iteratively updating the cluster centroids and reassigning data points to the closest centroid until convergence or a maximum number of iterations is reached. This function uses a random initialization of the centroids and runs the algorithm for a maximum of 100 iterations.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.
        n_clusters: int
            The number of clusters to form.

        Returns
        --------
        Tuple[List, List, float]
            Three elements:
            1) A list containing the cluster centers.
            2) A list containing the cluster index for each input vector.
            3) The Within-Cluster Sum of Squares (WCSS) value for the clustering.
        """
        centers, clusters = self.cluster_kmeans(emb_vecs, n_clusters)
        wcss = 0
        for i, v in enumerate(emb_vecs):
            d = distance.euclidean(centers[clusters[i]], v)
            wcss += d
        return centers, clusters, wcss

    def cluster_hierarchical_single(self, emb_vecs: List) -> List:
        """
        Clusters input vectors using the hierarchical clustering method with single linkage.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.

        Returns
        --------
        List
            A list containing the cluster index for each input vector.
        """
        return linkage(emb_vecs, method='single')

    def cluster_hierarchical_complete(self, emb_vecs: List) -> List:
        """
        Clusters input vectors using the hierarchical clustering method with complete linkage.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.

        Returns
        --------
        List
            A list containing the cluster index for each input vector.
        """
        return linkage(emb_vecs, method='complete')

    def cluster_hierarchical_average(self, emb_vecs: List) -> List:
        """
        Clusters input vectors using the hierarchical clustering method with average linkage.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.

        Returns
        --------
        List
            A list containing the cluster index for each input vector.
        """
        return linkage(emb_vecs, method='average')

    def cluster_hierarchical_ward(self, emb_vecs: List) -> List:
        """
        Clusters input vectors using the hierarchical clustering method with Ward's method.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.

        Returns
        --------
        List
            A list containing the cluster index for each input vector.
        """
        return linkage(emb_vecs, method='ward')

    def visualize_kmeans_clustering_wandb(self, data, n_clusters, project_name, run_name):
        """ This function performs K-means clustering on the input data and visualizes the resulting clusters by logging a scatter plot to Weights & Biases (wandb).

        This function applies the K-means algorithm to the input data and generates a scatter plot where each data point is colored according to its assigned cluster.
        For visualization use convert_to_2d_tsne to make your scatter plot 2d and visualizable.
        The function performs the following steps:
        1. Initialize a new wandb run with the provided project and run names.
        2. Perform K-means clustering on the input data with the specified number of clusters.
        3. Obtain the cluster labels for each data point from the K-means model.
        4. Create a scatter plot of the data, coloring each point according to its cluster label.
        5. Log the scatter plot as an image to the wandb run, allowing visualization of the clustering results.
        6. Close the plot display window to conserve system resources (optional).

        Parameters
        -----------
        data: np.ndarray
            The input data to perform K-means clustering on.
        n_clusters: int
            The number of clusters to form during the K-means clustering process.
        project_name: str
            The name of the wandb project to log the clustering visualization.
        run_name: str
            The name of the wandb run to log the clustering visualization.

        Returns
        --------
        None
        """
        # Initialize wandb
        # run = wandb.init(project=project_name, name=run_name)

        # Perform K-means clustering
        centers, clusters = self.cluster_kmeans(data, n_clusters, 10)

        # Plot the clusters
        plt.scatter([d[0] for d in data], [d[1] for d in data], c=clusters)
        plt.scatter([d[0] for d in centers], [d[1] for d in centers], c=[i for i in range(n_clusters)], marker='X')
        plt.show()

        # Log the plot to wandb
        # run.log({"KMeans result chart": plt})

        # Close the plot display window if needed (optional)
        plt.close()
        pass

    def wandb_plot_hierarchical_clustering_dendrogram(self, data, project_name, linkage_method, run_name):
        """ This function performs hierarchical clustering on the provided data and generates a dendrogram plot, which is then logged to Weights & Biases (wandb).

        The dendrogram is a tree-like diagram that visualizes the hierarchical clustering process. It shows how the data points (or clusters) are progressively merged into larger clusters based on their similarity or distance.

        The function performs the following steps:
        1. Initialize a new wandb run with the provided project and run names.
        2. Perform hierarchical clustering on the input data using the specified linkage method.
        3. Create a linkage matrix, which represents the merging of clusters at each step of the hierarchical clustering process.
        4. Generate a dendrogram plot using the linkage matrix.
        5. Log the dendrogram plot as an image to the wandb run.
        6. Close the plot display window to conserve system resources.

        Parameters
        -----------
        data: np.ndarray
            The input data to perform hierarchical clustering on.
        linkage_method: str
            The linkage method for hierarchical clustering. It can be one of the following: "average", "ward", "complete", or "single".
        project_name: str
            The name of the wandb project to log the dendrogram plot.
        run_name: str
            The name of the wandb run to log the dendrogram plot.

        Returns
        --------
        None
        """
        # run = wandb.init(project=project_name, name=run_name)
        # Perform hierarchical clustering
        res = None
        if linkage_method == 'single':
            res = self.cluster_hierarchical_single(data)
        if linkage_method == 'complete':
            res = self.cluster_hierarchical_complete(data)
        if linkage_method == 'average':
            res = self.cluster_hierarchical_average(data)
        if linkage_method == 'ward':
            res = self.cluster_hierarchical_ward(data)

        # Create linkage matrix for dendrogram
        plt.figure()
        d = dendrogram(res)
        plt.show()
        # run.log({'dendrogram plot': plt})
        pass

    def plot_kmeans_cluster_scores(self, embeddings: List, true_labels: List, k_values: List[int], project_name=None,
                                   run_name=None):
        """ This function, using implemented metrics in clustering_metrics, calculates and plots both purity scores and silhouette scores for various numbers of clusters.
        Then using wandb plots the respective scores (each with a different color) for each k value.

        Parameters
        -----------
        embeddings : List
            A list of vectors representing the data points.
        true_labels : List
            A list of ground truth labels for each data point.
        k_values : List[int]
            A list containing the various values of 'k' (number of clusters) for which the scores will be calculated.
            Default is range(2, 9), which means it will calculate scores for k values from 2 to 8.
        project_name : str
            Your wandb project name. If None, the plot will not be logged to wandb. Default is None.
        run_name : str
            Your wandb run name. If None, the plot will not be logged to wandb. Default is None.

        Returns
        --------
        None
        """
        silhouette_scores = []
        purity_scores = []
        # Calculating Silhouette Scores and Purity Scores for different values of k
        for k in k_values:
            _, clusters= self.cluster_kmeans(embeddings, k)

            # Using implemented metrics in clustering_metrics, get the score for each k in k-means clustering
            # and visualize it.
            silhouette_scores.append(ClusteringMetrics().silhouette_score(embeddings, clusters))
            purity_scores.append(ClusteringMetrics().purity_score(true_labels, clusters))
            pass

        # Plotting the scores
        scores = {'silhouette score': silhouette_scores, 'purity scores': purity_scores}
        plt.bar(k_values, )
        fig, ax = plt.subplots(layout='constrained')
        width = 0.25
        multiplier = 0
        x = np.arange(len(k_values))
        for attribute, measurement in scores.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, measurement, width, label=attribute)
            ax.bar_label(rects, padding=3)
            multiplier += 1
        ax.set_ylabel('score')
        ax.set_xlabel('K')
        ax.set_title('Score Plot')
        # plt.show()

        # Logging the plot to wandb
        if project_name and run_name:
            import wandb
            # run = wandb.init(project=project_name, name=run_name)
            # wandb.log({"Cluster Scores": plt})

    def visualize_elbow_method_wcss(self, embeddings: List, k_values: List[int], project_name: str, run_name: str):
        """ This function implements the elbow method to determine the optimal number of clusters for K-means clustering based on the Within-Cluster Sum of Squares (WCSS).

        The elbow method is a heuristic used to determine the optimal number of clusters in K-means clustering. It involves plotting the WCSS values for different values of K (number of clusters) and finding the "elbow" point in the curve, where the marginal improvement in WCSS starts to diminish. This point is considered as the optimal number of clusters.

        The function performs the following steps:
        1. Iterate over the specified range of K values.
        2. For each K value, perform K-means clustering using the `cluster_kmeans_WCSS` function and store the resulting WCSS value.
        3. Create a line plot of WCSS values against the number of clusters (K).
        4. Log the plot to Weights & Biases (wandb) for visualization and tracking.

        Parameters
        -----------
        embeddings: List
            A list of vectors representing the data points to be clustered.
        k_values: List[int]
            A list of K values (number of clusters) to explore for the elbow method.
        project_name: str
            The name of the wandb project to log the elbow method plot.
        run_name: str
            The name of the wandb run to log the elbow method plot.

        Returns
        --------
        None
        """
        # Initialize wandb
        # run = wandb.init(project=project_name, name=run_name)

        # Compute WCSS values for different K values
        wcss_values = []
        for k in k_values:
            _, _, score = self.cluster_kmeans_WCSS(embeddings, k)
            wcss_values.append(score)

        # Plot the elbow method
        plt.plot(k_values, wcss_values, 'o-', color='lightgreen')
        plt.show()

        # Log the plot to wandb
        # wandb.log({"Elbow Method": wandb.Image(plt)})

        plt.close()
# x1 = np.array([3, 1, 1, 2, 1, 6, 6, 6, 5, 6,\
#                7, 8, 9, 8, 9, 9, 8, 4, 4, 5, 4])
# x2 = np.array([5, 4, 5, 6, 5, 8, 6, 7, 6, 7, \
#                1, 2, 1, 2, 3, 2, 3, 9, 10, 9, 10])
# X = np.array(list(zip(x1, x2))).reshape(len(x1), 2).tolist()

# util = ClusteringUtils()
# util.visualize_kmeans_clustering_wandb(X, 3, 'project name', 'run name')
# util.visualize_elbow_method_wcss(X, [i for i in range(1, 10)], 'project name ', 'run name')
# util.wandb_plot_hierarchical_clustering_dendrogram(X, 'project name', 'ward', 'run name')