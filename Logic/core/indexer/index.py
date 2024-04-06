import time
import os
import json
import copy
from indexes_enum import Indexes

class Index:
    def __init__(self, preprocessed_documents: list):
        """
        Create a class for indexing.
        """

        self.preprocessed_documents = preprocessed_documents

        self.idx_to_fields = {
            Indexes.STARS: ['first_page_summary', 'directors', 'writers', 'stars', 'summaries', 'synopsis', 'reviews'],
            Indexes.GENRES: ['first_page_summary', 'summaries', 'synopsis', 'reviews', 'genres'],
            Indexes.SUMMARIES: ['summaries']
        }

        self.index = None
        self.path = 'index/'
        try:
            self.index = self.load_index(self.path)
        except:
            print('at least one of index files not found. Getting ')
            self.index = {
                Indexes.DOCUMENTS.value: self.index_documents(),
                Indexes.STARS.value: self.index_stars(),
                Indexes.GENRES.value: self.index_genres(),
                Indexes.SUMMARIES.value: self.index_summaries(),
            }
            for idx in Indexes:
                self.store_index(self.path, idx.value)
    
    def get_token_tf(self, doc, id, s, doc_tfs, type):
        if s not in doc_tfs:
            doc_tfs[id][s] = sum([(doc[field].count(s) if field in doc.keys() else 0) for field in self.idx_to_fields[type]])
        return doc_tfs[id][s]

    def index_documents(self):
        """
        Index the documents based on the document ID. In other words, create a dictionary
        where the key is the document ID and the value is the document.

        Returns
        ----------
        dict
            The index of the documents based on the document ID.
        """

        current_index = {}
        for doc in self.preprocessed_documents:
            current_index[doc['id']] = doc
        return current_index

    def index_stars(self):
        """
        Index the documents based on the stars.

        Returns
        ----------
        dict
            The index of the documents based on the stars. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """
        doc_tfs = {}

        stars_map = {}
        for doc in self.preprocessed_documents:
            id = doc['id']
            doc_tfs[id] = {}
            for s in doc['stars']:
                if s not in stars_map.keys():
                    stars_map[s] = {id: self.get_token_tf(doc, id, s, doc_tfs, Indexes.STARS)}
                elif id not in stars_map[s].keys():
                    stars_map[s][id] = self.get_token_tf(doc, id, s, doc_tfs, Indexes.STARS)
        stars_map = dict(sorted(stars_map.items()))
        return stars_map

    def index_genres(self):
        """
        Index the documents based on the genres.

        Returns
        ----------
        dict
            The index of the documents based on the genres. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """
        doc_tfs = {}

        genres_map = {}
        for doc in self.preprocessed_documents:
            id = doc['id']
            doc_tfs[id] = {}
            for s in doc['genres']:
                if s not in genres_map.keys():
                    genres_map[s] = {id: self.get_token_tf(doc, id, s, doc_tfs, Indexes.GENRES)}
                elif id not in genres_map[s].keys():
                    genres_map[s][id] = self.get_token_tf(doc, id, s, doc_tfs, Indexes.GENRES)
        genres_map = dict(sorted(genres_map.items()))
        return genres_map

    def index_summaries(self):
        """
        Index the documents based on the summaries (not first_page_summary).

        Returns
        ----------
        dict
            The index of the documents based on the summaries. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """

        current_index = {}
        doc_tfs = {}
        for doc in self.preprocessed_documents:
            id = doc['id']
            doc_tfs[id] = {}
            for s in doc['summaries']:
                if s not in current_index.keys():
                    current_index[s] = {id: self.get_token_tf(doc, id, s, doc_tfs, Indexes.SUMMARIES)}
                elif id not in current_index[s].keys():
                    current_index[s][id] = self.get_token_tf(doc, id, s, doc_tfs, Indexes.SUMMARIES)
        current_index = dict(sorted(current_index.items()))

        return current_index

    def get_posting_list(self, word: str, index_type: str):
        """
        get posting_list of a word

        Parameters
        ----------
        word: str
            word we want to check
        index_type: str
            type of index we want to check (documents, stars, genres, summaries)

        Return
        ----------
        list
            posting list of the word (you should return the list of document IDs that contain the word and ignore the tf)
        """

        try:
            return self.index[index_type][word].keys()
        except:
            return []

    def add_document_to_index(self, document: dict):
        """
        Add a document to all the indexes

        Parameters
        ----------
        document : dict
            Document to add to all the indexes
        """

        id = document['id']
        for index in Indexes:
            m = self.index[index.value]
            if index == Indexes.DOCUMENTS:
                m[id] = document
                continue
            doc_tfs = {id: {}}
            for s in document[index.value]:
                if s not in m.keys():
                    m[s] = {id: self.get_token_tf(document, id, s, doc_tfs, index)}
                elif id not in m[s].keys():
                    m[s][id] = self.get_token_tf(document, id, s, doc_tfs, index)
            self.store_index(self.path, index.value)

    def remove_document_from_index(self, document_id: str):
        """
        Remove a document from all the indexes

        Parameters
        ----------
        document_id : str
            ID of the document to remove from all the indexes
        """

        for index in Indexes:
            m = self.index[index.value]
            if index == Indexes.DOCUMENTS:
                m.pop(document_id)
                continue
            for _,t in m.items():
                t.pop(document_id, -1)
            self.store_index(self.path, index.value)

    def check_add_remove_is_correct(self):
        """
        Check if the add and remove is correct
        """

        dummy_document = {
            'id': '100',
            'stars': ['tim', 'henry'],
            'genres': ['drama', 'crime'],
            'summaries': ['good']
        }

        index_before_add = copy.deepcopy(self.index)
        self.add_document_to_index(dummy_document)
        index_after_add = copy.deepcopy(self.index)

        if index_after_add[Indexes.DOCUMENTS.value]['100'] != dummy_document:
            print('Add is incorrect, document')
            return

        if (set(index_after_add[Indexes.STARS.value]['tim']).difference(set(index_before_add[Indexes.STARS.value]['tim']))
                != {dummy_document['id']}):
            print('Add is incorrect, tim')
            return

        if (set(index_after_add[Indexes.STARS.value]['henry']).difference(set(index_before_add[Indexes.STARS.value]['henry']))
                != {dummy_document['id']}):
            print('Add is incorrect, henry')
            return
        if (set(index_after_add[Indexes.GENRES.value]['drama']).difference(set(index_before_add[Indexes.GENRES.value]['drama']))
                != {dummy_document['id']}):
            print('Add is incorrect, drama')
            return

        if (set(index_after_add[Indexes.GENRES.value]['crime']).difference(set(index_before_add[Indexes.GENRES.value]['crime']))
                != {dummy_document['id']}):
            print('Add is incorrect, crime')
            return

        if (set(index_after_add[Indexes.SUMMARIES.value]['good']).difference(set(index_before_add[Indexes.SUMMARIES.value]['good']))
                != {dummy_document['id']}):
            print('Add is incorrect, good')
            return

        print('Add is correct')

        self.remove_document_from_index('100')
        index_after_remove = copy.deepcopy(self.index)

        if index_after_remove == index_before_add:
            print('Remove is correct')
        else:
            print('Remove is incorrect')

    def store_index(self, path: str, index_name: str = None):
        """
        Stores the index in a file (such as a JSON file)

        Parameters
        ----------
        path : str
            Path to store the file
        index_name: str
            name of index we want to store (documents, stars, genres, summaries)
        """

        if not os.path.exists(path):
            os.makedirs(path)

        if index_name not in self.index:
            raise ValueError('Invalid index name')
        
        with open(path + index_name + '_index.json', 'w+') as f:
            f.write(json.dumps(self.index[index_name], indent=1))
            f.close()

    def load_index(self, path: str):
        """
        Loads the index from a file (such as a JSON file)

        Parameters
        ----------
        path : str
            Path to load the file
        """
        def get_map(idx):
            with open(path + idx + '.json', 'r') as f:
                return json.load(f)

        return {idx.value:get_map(idx.value) for idx in Indexes}

    def check_if_index_loaded_correctly(self, index_type: str, loaded_index: dict):
        """
        Check if the index is loaded correctly

        Parameters
        ----------
        index_type : str
            Type of index to check (documents, stars, genres, summaries)
        loaded_index : dict
            The loaded index

        Returns
        ----------
        bool
            True if index is loaded correctly, False otherwise
        """

        return self.index[index_type] == loaded_index

    def check_if_indexing_is_good(self, index_type: str, check_word: str = 'good'):
        """
        Checks if the indexing is good. Do not change this function. You can use this
        function to check if your indexing is correct.

        Parameters
        ----------
        index_type : str
            Type of index to check (documents, stars, genres, summaries)
        check_word : str
            The word to check in the index

        Returns
        ----------
        bool
            True if indexing is good, False otherwise
        """

        # brute force to check check_word in the summaries
        start = time.time()
        docs = []
        for document in self.preprocessed_documents:
            if index_type not in document or document[index_type] is None:
                continue

            for field in document[index_type]:
                if check_word in field:
                    docs.append(document['id'])
                    break

            # if we have found 3 documents with the word, we can break
            if len(docs) == 3:
                break

        end = time.time()
        brute_force_time = end - start

        # check by getting the posting list of the word
        start = time.time()
        # TODO: based on your implementation, you may need to change the following line
        posting_list = self.get_posting_list(check_word, index_type)

        end = time.time()
        implemented_time = end - start

        print('Brute force time: ', brute_force_time)
        print('Implemented time: ', implemented_time)

        if set(docs).issubset(set(posting_list)):
            print('Indexing is correct')

            if implemented_time < brute_force_time:
                print('Indexing is good')
                return True
            else:
                print('Indexing is bad')
                return False
        else:
            print('Indexing is wrong')
            return False

documents = None
with open('IMDB_crawled_pre_processed.json', 'r') as f:
    documents = json.load(f)
    f.close()
idx = Index(documents)
for i in Indexes:
    assert idx.check_if_index_loaded_correctly(i.value, idx.index[i.value])
assert idx.check_if_indexing_is_good(Indexes.GENRES.value, 'action')