import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# import sys
# import os/
# sys.path.append(os.path.abspath(os.path.join('..', 'IMDB-IR-System/Logic/core/word_embedding')))
# from fasttext_model import FastText

from ..word_embedding.fasttext_model import FastText
from dimension_reduction import DimensionReduction
from clustering_metrics import ClusteringMetrics
from clustering_utils import ClusteringUtils

# Main Function: Clustering Tasks

# 0. Embedding Extraction
# Using the previous preprocessor and fasttext model, collect all the embeddings of our data and store them.
ft_model = FastText()
path = 'IMDB_crawled_give.json'
ft_data_loader = FastTextDataLoader(path)
X, y = ft_data_loader.create_train_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
ft_model.prepare(None, 'load')
embeddings = [ft_model.get_query_embedding(sentence, do_preprocess=True) for sentence in X_train]

# 1. Dimension Reduction
# Perform Principal Component Analysis (PCA):
#     - Reduce the dimensionality of features using PCA. (you can use the reduced feature afterward or use to the whole embeddings)
#     - Find the Singular Values and use the explained_variance_ratio_ attribute to determine the percentage of variance explained by each principal component.
#     - Draw plots to visualize the results.(We will use TSNE for this purpose.)
print('PCA')
dim = DimensionReduction()
pca_reduced_features = dim.pca_reduce_dimension(embeddings, 2)
# dim.wandb_plot_explained_variance_by_components(embeddings, 'PCA project', 'PCA run')

# Implement t-SNE (t-Distributed Stochastic Neighbor Embedding):
#     - Create the convert_to_2d_tsne function, which takes a list of embedding vectors as input and reduces the dimensionality to two dimensions using the t-SNE method.
#     - Use the output vectors from this step to draw the diagram.
print('TSNE')
tsne_reduced_features = dim.convert_to_2d_tsne(embeddings)
# dim.wandb_plot_2d_tsne(embeddings, 'TSNE project', 'TSNE run')

# 2. Clustering
## K-Means Clustering
# Implement the K-means clustering algorithm from scratch.
# Create document clusters using K-Means.
# Run the algorithm with several different values of k.
# For each run:
#     - Determine the genre of each cluster based on the number of documents in each cluster.
#     - Draw the resulting clustering using the two-dimensional vectors from the previous section.
#     - Check the implementation and efficiency of the algorithm in clustering similar documents.
# Draw the silhouette score graph for different values of k and perform silhouette analysis to choose the appropriate k.
# Plot the purity value for k using the labeled data and report the purity value for the final k. (Use the provided functions in utilities)
print('KMEANS')
clustering = ClusteringUtils()

metrics = ClusteringMetrics()
centers, clusters = clustering.cluster_kmeans(pca_reduced_features.tolist(), 20)
clustering.visualize_kmeans_clustering_wandb(pca_reduced_features, 20, 'KMEANS PROJECT', 'KMEANS RUN')

# for k in range(1, 50):
    # centers, clusters = clustering.cluster_kmeans(pca_reduced_features.tolist(), k)
    # clustering.visualize_kmeans_clustering_wandb(pca_reduced_features, k, 'KMEANS PROJECT', 'KMEANS RUN')

clustering.plot_kmeans_cluster_scores(pca_reduced_features, y_train, [i for i in range(50)], 'KMEANS PROJECT', 'KMEANS RUN')

## Hierarchical Clustering
clustering.wandb_plot_hierarchical_clustering_dendrogram(embeddings, 'dendrogram project', 'average', 'dendrogram run')

# 3. Evaluation
clustering.plot_kmeans_cluster_scores(pca_reduced_features, y_train, [i for i in range(50)], 'KMEANS PROJECT', 'KMEANS RUN')
