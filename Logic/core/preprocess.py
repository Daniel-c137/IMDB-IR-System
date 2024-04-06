from unidecode import unidecode
import contractions
import json
import string
import re
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('punkt', )
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize

tag_dict = {"j": wordnet.ADJ,
            "n": wordnet.NOUN,
            "v": wordnet.VERB,
            "r": wordnet.ADV
            }
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    return tag_dict.get(tag, wordnet.NOUN)

class Preprocessor:

    def __init__(self, documents: list):
        """
        Initialize the class.

        Parameters
        ----------
        documents : list
            The list of documents to be preprocessed, path to stop words, or other parameters.
        """
        # TODO
        self.documents = documents
        with open('Logic/core/stopwords.txt', 'r') as f:
            self.stopwords = set([w[:-1] for w in f.readlines()])
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    def preprocess(self):
        """
        Preprocess the text using the methods in the class.

        Returns
        ----------
        List[str]
            The preprocessed documents.
        """
        for document in self.documents:
            self.apply_to_fields(document, self.prepreprocess)
            self.apply_to_fields(document, self.remove_links, text_only=True)
            self.apply_to_fields(document, self.remove_punctuations, text_only=True)
            self.apply_to_fields(document, self.remove_stopwords, text_only=True)
            self.apply_to_fields(document, self.tokenize)
            self.apply_to_fields(document, self.normalize)
        return self.documents
    
    def apply_to_fields(self, document, f, apply_to_names=True, text_only=False):
        # texts
        document['title'] = f(document['title'])
        document['first_page_summary'] = f(document['first_page_summary'])
        document['summaries'] = [f(s) for s in document['summaries']]
        document['synopsis'] = [f(s) for s in document['synopsis']]
        document['reviews'] = [[f(s[0])] for s in document['reviews']]
        if text_only:
            return

        # people
        if apply_to_names:
            document['directors'] = [f(d) for d in document['directors']]
            document['writers'] = [f(w) for w in document['writers']]
            document['stars'] = [f(s) for s in document['stars']]

        # other
        document['genres'] = [f(g) for g in document['genres']]
        document['languages'] = [f(l) for l in document['languages']]
        document['countries_of_origin'] = [f(c) for c in document['countries_of_origin']]

    def prepreprocess(self, text: str):
        result = []
        lowered = unidecode(text).lower()
        for word in lowered.split():
            result.append(contractions.fix(word))
        return ' '.join(result)


    def normalize(self, tokens: list):
        """
        Normalize the text by converting it to a lower case, stemming, lemmatization, etc.

        Parameters
        ----------
        text : list
            The text to be normalized.

        Returns
        ----------
        str
            The normalized text.
        """
        return [self.lemmatizer.lemmatize(token, get_wordnet_pos(token)) for token in tokens]

    def remove_links(self, text: str):
        """
        Remove links from the text.

        Parameters
        ----------
        text : str
            The text to be processed.

        Returns
        ----------
        str
            The text with links removed.
        """
        patterns = [r'\S*http\S*', r'\S*www\S*', r'\S+\.ir\S*', r'\S+\.com\S*', r'\S+\.org\S*', r'\S*@\S*']
        for p in patterns:
            text = re.sub(p, '', text)
        return text

    def remove_punctuations(self, text: str):
        """
        Remove punctuations from the text.

        Parameters
        ----------
        text : str
            The text to be processed.

        Returns
        ----------
        str
            The text with punctuations removed.
        """
        text = re.sub(r'([A-Za-z])(\.)([A-Za-z])', r'\1\2 \3', text)
        return text.translate(str.maketrans('', '', string.punctuation))

    def tokenize(self, text: str):
        """
        Tokenize the words in the text.

        Parameters
        ----------
        text : str
            The text to be tokenized.

        Returns
        ----------
        list
            The list of words.
        """
        return word_tokenize(text)

    def remove_stopwords(self, text: str):
        """
        Remove stopwords from the text.

        Parameters
        ----------
        text : str
            The text to remove stopwords from.

        Returns
        ----------
        list
            The list of words with stopwords removed.
        """
        for word in self.stopwords:
            text = text.replace(word, '')
        return text