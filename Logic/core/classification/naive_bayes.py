import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report

from .basic_classifier import BasicClassifier
from .data_loader import ReviewLoader


class NaiveBayes(BasicClassifier):
    def __init__(self, count_vectorizer, alpha=1):
        super().__init__()
        self.cv = count_vectorizer
        self.num_classes = None
        self.classes = None
        self.number_of_features = None
        self.number_of_samples = None
        self.prior = None
        self.feature_probabilities = None
        self.log_probs = None
        self.alpha = alpha

    def fit(self, x, y):
        """
        Fit the features and the labels
        Calculate prior and feature probabilities

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
        self.classes = np.unique(y)
        self.num_classes = len(self.classes)
        self.number_of_samples = x.shape[0]
        self.number_of_features = x.shape[1]

        self.prior = np.zeros(self.num_classes)
        self.feature_probabilities = np.zeros((self.num_classes, self.number_of_features))

        for i, clazz in enumerate(self.classes):
            class_docs = x[y == clazz]
            self.prior[i] = class_docs.shape[0] / self.number_of_samples
            prob = (class_docs.sum(axis=0) + self.alpha) / (class_docs.sum() + self.alpha * self.number_of_features)
            self.feature_probabilities[i] = prob
        
        self.log_probs = np.log(self.feature_probabilities)
        self.log_prior = np.log(self.prior)

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
        log_likelihood = x @ self.log_probs.T
        log_posterior = log_likelihood + self.log_prior
        class_pred = np.argmax(log_posterior, axis=1)
        return self.classes[class_pred]

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

    def get_percent_of_positive_reviews(self, sentences):
        """
        You have to override this method because we are using a different embedding method in this class.
        """
        x = self.cv.transform(sentences).toarray()
        predictions = self.predict(x)
        positive_reviews = np.sum(predictions == 1)
        return positive_reviews / len(sentences)


# F1 Accuracy : 85%
if __name__ == '__main__':
    """
    First, find the embeddings of the revies using the CountVectorizer, then fit the model with the training data.
    Finally, predict the test data and print the classification report
    You can use scikit-learn's CountVectorizer to find the embeddings.
    """
    rl = ReviewLoader('IMDB Dataset.csv')
    rl.load_data()
    rl.get_embeddings()
    cv = CountVectorizer()
    X = cv.fit_transform(rl.review_tokens)
    y = rl.sentiments
    X_train, X_test, y_train, y_test = rl.split_data(test_data_ratio=0.33)
    classifier = NaiveBayes(cv)
    classifier.fit(X_train, y_train)
    report = classifier.prediction_report(X_test, y_test)
    print(report)
