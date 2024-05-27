from ..word_embedding.fasttext_model import FastText


class BasicClassifier:
    def __init__(self):
        pass

    def fit(self, x, y):
        raise NotImplementedError()

    def predict(self, x):
        raise NotImplementedError()

    def prediction_report(self, x, y):
        raise NotImplementedError()

    def get_percent_of_positive_reviews(self, sentences):
        """
        Get the percentage of positive reviews in the given sentences
        Parameters
        ----------
        sentences: list
            The list of sentences to get the percentage of positive reviews
        Returns
        -------
        float
            The percentage of positive reviews
        """
        cnt = 0
        ft_model = FastText()
        ft_model.prepare(None, 'load')
        for sentence in sentences:
            prediction = self.predict(ft_model.get_query_embedding(sentence))[0]
            if prediction == 1:
                cnt += 1

        return (cnt / len(sentences)) * 100

