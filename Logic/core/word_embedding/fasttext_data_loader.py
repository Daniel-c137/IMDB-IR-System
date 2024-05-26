import json
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder


class FastTextDataLoader:
    """
    This class is designed to load and pre-process data for training a FastText model.

    It takes the file path to a data source containing movie information (synopses, summaries, reviews, titles, genres) as input.
    The class provides methods to read the data into a pandas DataFrame, pre-process the text data, and create training data (features and labels)
    """
    def __init__(self, file_path):
        """
        Initializes the FastTextDataLoader class with the file path to the data source.

        Parameters
        ----------
        file_path: str
            The path to the file containing movie information.
        """
        self.file_path = file_path
        self.df = None

    def read_data_to_df(self):
        """
        Reads data from the specified file path and creates a pandas DataFrame containing movie information.

        You can use an IndexReader class to access the data based on document IDs.
        It extracts synopses, summaries, reviews, titles, and genres for each movie.
        The extracted data is then stored in a pandas DataFrame with appropriate column names.

        Returns
        ----------
            pd.DataFrame: A pandas DataFrame containing movie information (synopses, summaries, reviews, titles, genres).
        """
        with open(self.file_path, 'r') as f:
            data = json.load(f)
            pd.json_normalize(data)
            self.df = pd.json_normalize(data)
            required_fields = ['synposis', 'summaries', 'reviews', 'title', 'genres']
            labels_field = 'genres'
            for c in self.df.columns:
                if c not in required_fields:
                    self.df = self.df.drop(c, axis=1)
            for f in required_fields:
                if f != labels_field:
                    self.df[f] = self.df[f].apply(to_str)
            self.df = self.df.reset_index()
            nones = []
            for index, row in self.df.iterrows():
                if row['summaries'] is None or len(row['summaries']) == 0:
                    nones.append(index)
            for index in nones:
                self.df = self.df.drop(index)
            self.df = self.df.explode(labels_field, ignore_index=True)


    def create_train_data(self):
        """
        Reads data using the read_data_to_df function, pre-processes the text data, and creates training data (features and labels).

        Returns:
            tuple: A tuple containing two NumPy arrays: X (preprocessed text data) and y (encoded genre labels).
        """
        self.read_data_to_df()
        le = LabelEncoder()
        x = self.df['summaries']
        le.fit(self.df['genres'])
        y = le.transform(self.df['genres'])
        return x, y

def to_str(x):
    if x == None:
        return x
    if len(x) == 0:
        return x
    if isinstance(x, str):
        return x
    try:
        return ' '.join(x)
    except:
        return ' '.join(x[0])