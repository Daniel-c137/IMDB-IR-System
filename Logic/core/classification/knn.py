import numpy as np
from sklearn.metrics import classification_report
from collections import Counter

from basic_classifier import BasicClassifier
from data_loader import ReviewLoader


class KnnClassifier(BasicClassifier):
    def __init__(self, n_neighbors):
        super().__init__()
        self.k = n_neighbors

    def fit(self, x, y):
        """
        Fit the model using X as training data and y as target values
        use the Euclidean distance to find the k nearest neighbors
        Warning: Maybe you need to reduce the size of X to avoid memory errors

        Parameters
        ----------
        x: np.ndarray
            An m * n matrix - m is count of docs and n is embedding size
        y: np.ndarray
            The real class label for each doc
        Returns
        -------
        self
            Returns self as a classifier
        """
        self.X = x
        self.y = y

    def predict(self, x):
        """
        Parameters
        ----------
        x: np.ndarray
            An k * n matrix - k is count of docs and n is embedding size
        Returns
        -------
        np.ndarray
            Return the predicted class for each doc
            with the highest probability (argmax)
        """
        y_pred = []
        for i in range(x.shape[0]):
            ds = np.linalg.norm(self.X - x[i, :], axis=1)
            indices = np.argsort(ds)[:self.k]
            labels = self.y[indices]
            l = Counter(labels).most_common(1)[0][0]
            y_pred.append(l)
        return np.array(y_pred)

    def prediction_report(self, x, y):
        """
        Parameters
        ----------
        x: np.ndarray
            An k * n matrix - k is count of docs and n is embedding size
        y: np.ndarray
            The real class label for each doc
        Returns
        -------
        str
            Return the classification report
        """
        y_pred = self.predict(x)
        return classification_report(y, y_pred)


# F1 Accuracy : 70%
if __name__ == '__main__':
    """
    Fit the model with the training data and predict the test data, then print the classification report
    """
    rl = ReviewLoader('IMDB Dataset.csv')
    rl.load_data()
    rl.get_embeddings()
    X_train, X_test, y_train, y_test = rl.split_data(test_data_ratio=0.33)
    
    classifier = KnnClassifier(10)
    classifier.fit(X_train, y_train)

    print(classifier.prediction_report(X_test, y_test))
