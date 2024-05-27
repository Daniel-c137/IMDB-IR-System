import fasttext
import re
import string

from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from numpy.linalg import norm
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

from fasttext_data_loader import FastTextDataLoader


def preprocess_text(text, minimum_length=2, stopword_removal=True, stopwords_domain=[], lower_case=True,
                       punctuation_removal=True):
    """
    preprocess text by removing stopwords, punctuations, and converting to lowercase, and also filter based on a min length
    for stopwords use nltk.corpus.stopwords.words('english')
    for punctuations use string.punctuation

    Parameters
    ----------
    text: str
        text to be preprocessed
    minimum_length: int
        minimum length of the token
    stopword_removal: bool
        whether to remove stopwords
    stopwords_domain: list
        list of stopwords to be removed base on domain
    lower_case: bool
        whether to convert to lowercase
    punctuation_removal: bool
        whether to remove punctuations
    """
    tokens = word_tokenize(text)
    res = []
    for t in tokens:
        r = t
        if stopword_removal:
            for sw in stopwords_domain:
                r = r.replace(sw, '')
        if lower_case:
            r = r.lower()
        if punctuation_removal:
            r = r.translate(str.maketrans('', '', string.punctuation))
        if len(text) >= minimum_length:
            res.append(r)
    return ' '.join(res)

class FastText:
    """
    A class used to train a FastText model and generate embeddings for text data.

    Attributes
    ----------
    method : str
        The training method for the FastText model.
    model : fasttext.FastText._FastText
        The trained FastText model.
    """

    def __init__(self, preprocessor=preprocess_text, method='skipgram'):
        """
        Initializes the FastText with a preprocessor and a training method.

        Parameters
        ----------
        method : str, optional
            The training method for the FastText model.
        """
        self.method = method
        self.model = None
        self.preprocess = preprocessor


    def train(self, texts):
        """
        Trains the FastText model with the given texts.

        Parameters
        ----------
        texts : list of str
            The texts to train the FastText model.
        """
        df = pd.DataFrame()
        # df['label'] = ['__label__' + str(l) for l in labels]
        df['text'] = [self.preprocess(t) for t in texts]
        df[['text']].to_csv('temp.txt', index=None, header=None)
        self.model = fasttext.train_unsupervised(input='temp.txt', model=self.method, minCount=50)

    def get_query_embedding(self, query, do_preprocess):
        """
        Generates an embedding for the given query.

        Parameters
        ----------
        query : str
            The query to generate an embedding for.
        tf_idf_vectorizer : sklearn.feature_extraction.text.TfidfVectorizer
            The TfidfVectorizer to transform the query.
        do_preprocess : bool, optional
            Whether to preprocess the query.

        Returns
        -------
        np.ndarray
            The embedding for the query.
        """
        if do_preprocess:
            query = self.preprocess(query)
        return self.model.get_sentence_vector(query)
        



    def analogy(self, word1, word2, word3):
        """
        Perform an analogy task: word1 is to word2 as word3 is to __.

        Args:
            word1 (str): The first word in the analogy.
            word2 (str): The second word in the analogy.
            word3 (str): The third word in the analogy.

        Returns:
            str: The word that completes the analogy.
        """
        # Obtain word embeddings for the words in the analogy
        w1 = self.model[word1]
        w2 = self.model[word2]
        w3 = self.model[word3]

        # Perform vector arithmetic
        result_vector = w1 / norm(w1) - w2 / norm(w2) + w3 / norm(w3)

        # Create a dictionary mapping each word in the vocabulary to its corresponding vector
        # this function is built-in inside the fasttext model.

        # Exclude the input words from the possible results
        input_words = [word1, word2, word3]

        # Find the word whose vector is closest to the result vector
        best_word = None
        max_sim = 0
        for w in self.model.words:
            if w in input_words:
                continue
            d = np.dot(self.model[w], result_vector) / (norm(self.model[w]) * norm(result_vector))
            if d >= max_sim:
                max_sim = d
                best_word = w
        return best_word


    def save_model(self, path='FastText_model.bin'):
        """
        Saves the FastText model to a file.

        Parameters
        ----------
        path : str, optional
            The path to save the FastText model.
        """
        self.model.save_model(path)

    def load_model(self, path="FastText_model.bin"):
        """
        Loads the FastText model from a file.

        Parameters
        ----------
        path : str, optional
            The path to load the FastText model.
        """
        self.model = fasttext.load_model(path)

    def prepare(self, dataset, mode, save=False, path='FastText_model.bin'):
        """
        Prepares the FastText model.

        Parameters
        ----------
        dataset : list of str
            The dataset to train the FastText model.
        mode : str
            The mode to prepare the FastText model.
        """
        if mode == 'train':
            self.train(dataset)
        if mode == 'load':
            self.load_model(path)
        if save:
            self.save_model(path)

if __name__ == "__main__":
    ft_model = FastText(preprocessor=preprocess_text, method='skipgram')

    path = 'IMDB_crawled_give.json'
    ft_data_loader = FastTextDataLoader(path)

    X, Y = ft_data_loader.create_train_data()

    # ft_model.prepare(None, None, 'load')
    ft_model.prepare(X, 'train')
    ft_model.prepare(None, None, save=True)

    print(10 * "*" + "Similarity" + 10 * "*")
    word = 'queen'
    neighbors = ft_model.model.get_nearest_neighbors(word, k=5)

    for neighbor in neighbors:
        print(f"Word: {neighbor[1]}, Similarity: {neighbor[0]}")

    print(10 * "*" + "Analogy" + 10 * "*")
    word1 = "man"
    word2 = "king"
    word3 = "queen"
    print(f"Similarity between {word1} and {word2} is like similarity between {word3} and {ft_model.analogy(word1, word2, word3)}")
