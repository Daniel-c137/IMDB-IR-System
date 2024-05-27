import numpy as np
import pandas as pd
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ..word_embedding.fasttext_model import FastText


class ReviewLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.fasttext_model = FastText()
        self.review_tokens = []
        self.sentiments = []
        self.embeddings = []

    def load_data(self):
        """
        Load the data from the csv file and preprocess the text. Then save the normalized tokens and the sentiment labels.
        Also, load the fasttext model.
        """
        df = pd.read_csv(self.file_path)
        print(df.head(5))
        self.review_tokens = df['review'].apply(self.fasttext_model.preprocess).tolist()
        self.sentiments = LabelEncoder().fit_transform(df['sentiment'])
        self.fasttext_model.prepare(None, 'load')

    def get_embeddings(self):
        """
        Get the embeddings for the reviews using the fasttext model.
        """
        if len(self.embeddings) > 0:
            return  self.embeddings
        self.embeddings = [self.fasttext_model.get_query_embedding(review, do_preprocess=True) for review in self.review_tokens]
        return  self.embeddings

    def split_data(self, test_data_ratio=0.2):
        """
        Split the data into training and testing data.

        Parameters
        ----------
        test_data_ratio: float
            The ratio of the test data
        Returns
        -------
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
            Return the training and testing data for the embeddings and the sentiments.
            in the order of x_train, x_test, y_train, y_test
        """
        X = np.array(self.embeddings)
        y = np.array(self.sentiments)
        
        return train_test_split(X, y, test_size=test_data_ratio, random_state=42)
