import numpy as np


class Scorer:
    def __init__(self, index, number_of_documents):
        """
        Initializes the Scorer.

        Parameters
        ----------
        index : dict
            The index to score the documents with.
        number_of_documents : int
            The number of documents in the index.
        """

        self.index = index
        self.idf = {}
        self.N = number_of_documents
        self.k = 5
        self.b = 0.1

    def get_list_of_documents(self, query):
        """
        Returns a list of documents that contain at least one of the terms in the query.

        Parameters
        ----------
        query: List[str]
            The query to be scored

        Returns
        -------
        list
            A list of documents that contain at least one of the terms in the query.

        Note
        ---------
            The current approach is not optimal but we use it due to the indexing structure of the dict we're using.
            If we had pairs of (document_id, tf) sorted by document_id, we could improve this.
                We could initialize a list of pointers, each pointing to the first element of each list.
                Then, we could iterate through the lists in parallel.

        """
        list_of_documents = []
        for term in query:
            if term in self.index.keys():
                list_of_documents.extend(self.index[term].keys())
        return list(set(list_of_documents))

    def get_idf(self, term):
        """
        Returns the inverse document frequency of a term.

        Parameters
        ----------
        term : str
            The term to get the inverse document frequency for.

        Returns
        -------
        float
            The inverse document frequency of the term.

        Note
        -------
            It was better to store dfs in a separate dict in preprocessing.
        """
        idf = self.idf.get(term, None)
        if idf is None:
            docs = self.index.get(term, None)
            if docs is None:
                return 0
            idf = np.log10(self.N / len(docs.keys()))
            self.idf[term] = idf
        return idf

    def get_query_tfs(self, query):
        """
        Returns the term frequencies of the terms in the query.

        Parameters
        ----------
        query : List[str]
            The query to get the term frequencies for.

        Returns
        -------
        dict
            A dictionary of the term frequencies of the terms in the query.
        """

        return {term: query.count(term) for term in set(query)}

    def compute_scores_with_vector_space_model(self, query, method):
        """
        compute scores with vector space model

        Parameters
        ----------
        query: List[str]
            The query to be scored
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c))
            The method to use for searching.

        Returns
        -------
        dict
            A dictionary of the document IDs and their scores.
        """

        result = {}
        for id in self.get_list_of_documents(query):
            doc_meth, quey_meth = method.split('.')
            score = self.get_vector_space_model_score(query, self.get_query_tfs(query), id, doc_meth, quey_meth)
            result[id] = score
        return result


    def get_vector_space_model_score(self, query, query_tfs, document_id, document_method, query_method):
        """
        Returns the Vector Space Model score of a document for a query.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        query_tfs : dict
            The term frequencies of the terms in the query.
        document_id : str
            The document to calculate the score for.
        document_method : str (n|l)(n|t)(n|c)
            The method to use for the document.
        query_method : str (n|l)(n|t)(n|c)
            The method to use for the query.

        Returns
        -------
        float
            The Vector Space Model score of the document for the query.
        """
        def get_tf_idf(tf_meth, df_meth, tf, idf):
            new_tf = tf if tf_meth == 'n' else 1 + (np.log10(tf) if tf != 0 else 0) if tf_meth == 'l' else None
            new_idf = 1 if df_meth == 'n' else idf if df_meth == 't' else None
            return new_tf * new_idf
        def normalize(v):
            return v / np.linalg.norm(v)
        
        doc_vec = []
        q_vec = []
        q_tf_meth, q_idf_meth, q_vec_meth = query_method
        d_tf_meth, d_idf_meth, d_vec_meth = document_method
        
        for q, tf in query_tfs.items():
            raw_idf = self.get_idf(q)
            w_q = get_tf_idf(q_tf_meth, q_idf_meth, tf, raw_idf)
            docs = self.index.get(q, None)
            w_d = get_tf_idf(d_tf_meth, d_idf_meth, 0 if docs is None else docs.get(document_id, 0), raw_idf)
            doc_vec.append(w_d)
            q_vec.append(w_q)
        q_vec = np.array(q_vec)
        q_vec = q_vec if q_vec_meth == 'n' else normalize(q_vec) if q_vec_meth == 'c' else None
        doc_vec = np.array(doc_vec)
        doc_vec = doc_vec if d_vec_meth == 'n' else normalize(doc_vec) if d_vec_meth == 'c' else None
        return np.dot(doc_vec, q_vec)


    def compute_socres_with_okapi_bm25(self, query, average_document_field_length, document_lengths):
        """
        compute scores with okapi bm25

        Parameters
        ----------
        query: List[str]
            The query to be scored
        average_document_field_length : float
            The average length of the documents in the index.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.

        Returns
        -------
        dict
            A dictionary of the document IDs and their scores.
        """

        result = {}
        for id in self.get_list_of_documents(query):
            result[id] = self.get_okapi_bm25_score(query, id, average_document_field_length, document_lengths[id])
        return result

    def get_okapi_bm25_score(
        self, query, document_id, average_document_field_length, document_length
    ):
        """
        Returns the Okapi BM25 score of a document for a query.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        document_id : str
            The document to calculate the score for.
        average_document_field_length : float
            The average length of the documents in the index.
        document_lengths : Int
            Length of current document

        Returns
        -------
        float
            The Okapi BM25 score of the document for the query.
        """
        const = self.k * ((1 - self.b) + (self.b * document_length / average_document_field_length))
        def calculate_tuning(tf):
            return ((self.k + 1) * tf) / (const + tf)

        rsv = 0
        for q in query:
            rsv += self.get_idf(q) * calculate_tuning(self.index[q].get(document_id, 0))
        return rsv

    def compute_scores_with_unigram_model(
        self, query, smoothing_method, document_lengths, alpha=0.5, lamda=0.5
    ):
        """
        Calculates the scores for each document based on the unigram model.

        Parameters
        ----------
        query : str
            The query to search for.
        smoothing_method : str (bayes | naive | mixture)
            The method used for smoothing the probabilities in the unigram model.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.
        alpha : float, optional
            The parameter used in bayesian smoothing method. Defaults to 0.5.
        lamda : float, optional
            The parameter used in some smoothing methods to balance between the document
            probability and the collection probability. Defaults to 0.5.

        Returns
        -------
        float
            A dictionary of the document IDs and their scores.
        """

        res = {}
        for id in self.get_list_of_documents(query):
            score = self.compute_score_with_unigram_model(query, id, smoothing_method, document_lengths, alpha, lamda)
            res[id] = score
        return res 

    def compute_score_with_unigram_model(
        self, query, document_id, smoothing_method, document_lengths, alpha, lamda
    ):
        """
        Calculates the scores for each document based on the unigram model.

        Parameters
        ----------
        query : str
            The query to search for.
        document_id : str
            The document to calculate the score for.
        smoothing_method : str (bayes | naive | mixture)
            The method used for smoothing the probabilities in the unigram model.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.
        alpha : float, optional
            The parameter used in bayesian smoothing method. Defaults to 0.5.
        lamda : float, optional
            The parameter used in some smoothing methods to balance between the document
            probability and the collection probability. Defaults to 0.5.

        Returns
        -------
        float
            The Unigram score of the document for the query.
        """

        query_tfs = self.get_query_tfs(query)
        l = document_lengths[document_id]

        if len(query_tfs) == 0:
            return 0
        
        if smoothing_method == 'bayes':
            res = 1
            for q, _ in query_tfs:
                docs = self.index.get(q, {})
                tf = docs.get(document_id, 0)
                ptmd = tf / l
                ptmc = sum(docs.values()) / sum(self.index.values())
                res *= (tf + alpha * ptmc) / (l + alpha)
            return res

        if smoothing_method == 'naive':
            res = 1
            for q, _ in query_tfs:
                docs = self.index.get(q, {})
                tf = docs.get(document_id, 0)
                ptmd = tf / l
                res *= ptmd
            return res

        if smoothing_method == 'mixture':
            res = 1
            for q, _ in query_tfs:
                docs = self.index.get(q, {})
                tf = docs.get(document_id, 0)
                ptmd = tf / l
                ptmc = sum(docs.values()) / sum(self.index.values())
                res *= ptmd * lamda + ptmc * (1 - lamda)
            return res