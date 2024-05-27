import numpy as np
import matplotlib.pyplot as plt
import wandb

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class DimensionReduction:

    def __init__(self):
        self.pca = PCA(0.7)
        self.tsne_2d = TSNE(n_components=2, random_state=42)

    def pca_reduce_dimension(self, embeddings, n_components):
        """
        Performs dimensional reduction using PCA with n components left behind.

        Parameters
        ----------
            embeddings (list): A list of embeddings of documents.
            n_components (int): Number of components to keep.

        Returns
        -------
            list: A list of reduced embeddings.
        """
        return self.pca.fit_transform(embeddings)

    def convert_to_2d_tsne(self, emb_vecs):
        """
        Converts each raw embedding vector to a 2D vector.

        Parameters
        ----------
            emb_vecs (list): A list of vectors.

        Returns
        --------
            list: A list of 2D vectors.
        """
        return self.tsne_2d.fit_transform(np.array(emb_vecs))

    def wandb_plot_2d_tsne(self, data, project_name, run_name):
        """ This function performs t-SNE (t-Distributed Stochastic Neighbor Embedding) dimensionality reduction on the input data and visualizes the resulting 2D embeddings by logging a scatter plot to Weights & Biases (wandb).

        t-SNE is a widely used technique for visualizing high-dimensional data in a lower-dimensional space, typically 2D. It aims to preserve the local structure of the data points while capturing the global structure as well. This function applies t-SNE to the input data and generates a scatter plot of the resulting 2D embeddings, allowing for visual exploration and analysis of the data's structure and potential clusters.

        The scatter plot is a useful way to visualize the t-SNE embeddings, as it shows the spatial distribution of the data points in the reduced 2D space.

        The function performs the following steps:
        1. Initialize a new wandb run with the provided project and run names.
        2. Perform t-SNE dimensionality reduction on the input data, obtaining 2D embeddings.
        3. Create a scatter plot of the 2D embeddings using matplotlib.
        4. Log the scatter plot as an image to the wandb run, allowing visualization of the t-SNE embeddings.

        Parameters
        -----------
        data: np.ndarray
            The input data to perform t-SNE dimensionality reduction on.
        project_name: str
            The name of the wandb project to log the t-SNE scatter plot.
        run_name: str
            The name of the wandb run to log the t-SNE scatter plot.

        Returns
        --------
        None
        """
        # Initialize wandb
        # run = wandb.init(project=project_name, name=run_name)

        # Perform t-SNE dimensionality reduction
        result = self.convert_to_2d_tsne(data)

        # Plot the t-SNE embeddings
        plt.scatter([r[0] for r in result], [r[1] for r in result])
        plt.show()

        # Log the plot to wandb
        # wandb.log({"t-SNE 2D Embeddings": wandb.Image(plt)})

        # Close the plot display window if needed (optional)
        # plt.close()

    def wandb_plot_explained_variance_by_components(self, data, project_name, run_name):
        """
        This function plots the cumulative explained variance ratio against the number of components for a given dataset and logs the plot to Weights & Biases (wandb).

        The cumulative explained variance ratio is a metric used in dimensionality reduction techniques, such as Principal Component Analysis (PCA), to determine the amount of information (variance) retained by the selected number of components. It helps in deciding how many components to keep while balancing the trade-off between retaining valuable information and reducing the dimensionality of the data.

        The function performs the following steps:
        1. Fit a PCA model to the input data and compute the cumulative explained variance ratio.
        2. Create a line plot using Matplotlib, where the x-axis represents the number of components, and the y-axis represents the corresponding cumulative explained variance ratio.
        3. Initialize a new wandb run with the provided project and run names.
        4. Log the plot as an image to the wandb run, allowing visualization of the explained variance by components.


        Parameters
        -----------
        data: np.ndarray
            The input data for which the explained variance by components will be computed and plotted.
        project_name: str
            The name of the wandb project to log the explained variance plot.
        run_name: str
            The name of the wandb run to log the explained variance plot.

        Returns
        --------
        None
        """

        # Fit PCA and compute cumulative explained variance ratio
        result = self.pca_reduce_dimension(data, 2)
        print('Cumulative Variance Ratio is:', self.pca.explained_variance_ratio_)

        # Create the plot
        plt.scatter([r[0] for r in result], [r[1] for r in result])
        plt.show()

        # Initialize wandb
        # run = wandb.init(project=project_name, name=run_name)

        # Log the plot to wandb
        # wandb.log({"Explained Variance": wandb.Image(plt)})

        # Close the plot display window if needed (optional)
        plt.close()
