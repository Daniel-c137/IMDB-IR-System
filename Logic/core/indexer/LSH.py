import numpy as np
import itertools
import random
import json

class MinHashLSH:
    def __init__(self, documents, num_hashes):
        """
        Initialize the MinHashLSH

        Parameters
        ----------
        documents : list of str
            The input documents for similarity analysis.
        num_hashes : int
            Number of hashes for mini-hashing.
        """
        self.documents = documents
        self.num_hashes = num_hashes
        self.s_map = {}
        self.s_cnt = 0
        self.m_cnt = 0

    def shingle_document(self, document, k=2):
        """
        Convert a document into a set of shingles.

        Parameters
        ----------
        document : str
            The input document.
        k : int
            The size of each shingle.

        Returns
        ----------
        set
            A set of shingles.
        """
        shingles = set()
        for i in range(len(document) - k + 1):
            s = document[i:i+k]
            if s not in self.s_map.keys():
                self.s_map[s] = self.s_cnt
                self.s_cnt += 1
            shingles.add(self.s_map[s])
        return shingles

    def build_characteristic_matrix(self):
        """
        Build the characteristic matrix representing the presence of shingles in documents.

        Returns
        ----------
        numpy.ndarray
            The binary characteristic matrix.
        """
        shingles = []
        for doc in self.documents:
            s = self.shingle_document(doc, 2)
            shingles.append((self.m_cnt, s))
            self.m_cnt += 1
        arr = np.zeros(shape=(self.s_cnt, self.m_cnt))
        for i, shingle in shingles:
            for s in shingle:
                arr[s][i] = 1

        return arr

    def min_hash_signature(self):
        """
        Perform Min-Hashing to generate hash signatures for documents.

        Returns
        ----------
        numpy.ndarray
            The Min-Hash signatures matrix.
        """
        char_matrix = self.build_characteristic_matrix()
        arr = np.zeros(shape=(self.num_hashes, self.m_cnt))
        for h in range(self.num_hashes):
            perm = [i for i in range(self.s_cnt)]
            random.shuffle(perm)
            for j in range(self.m_cnt):
                col = char_matrix[:, j]
                for i in perm:
                    if col[i] == 1:
                        arr[h][j] = i + 1
                        break
        return arr

    def lsh_buckets(self, signature, bands=10):
        """
        Group documents into Locality-Sensitive Hashing (LSH) buckets based on Min-Hash signatures.

        Parameters
        ----------
        signature : numpy.ndarray
            Min-Hash signatures for documents.
        bands : int
            Number of bands for LSH.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        
        rows = len(signature)
        rows_per_band = int(rows / bands)
        bands_list = []
        candidate_pairs = {}

        for i in range(0, rows, rows_per_band):
            hashes = []
            for j in range(self.m_cnt):
                sig = signature[:, j]
                h = hash(bytes(sig[i:i+rows_per_band]))
                other_id = None
                try:
                    other_id = hashes.index(h)
                except:
                    pass
                finally:
                    hashes.append(h)
                    if other_id is not None:
                        if h not in candidate_pairs.keys():
                            candidate_pairs[h] = set()            
                        candidate_pairs[h].add(j)
                        candidate_pairs[h].add(other_id)
        return candidate_pairs

    def perform_lsh(self):
        """
        Perform the entire Locality-Sensitive Hashing (LSH) process.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        return self.lsh_buckets(self.min_hash_signature())

    def jaccard_score(self, first_set, second_set):
        """
        Calculate jaccard score for two sets.

        Parameters
        ----------
        first_set : set
            Set of first shingled document.
        second_set : set
            Set of second shingled document.

        Returns
        ----------
        float
            Jaccard score.
        """
        return len(first_set.intersection(second_set))/ len(first_set.union(second_set))

    def jaccard_similarity_test(self, buckets, all_documents):
        """
        Test your near duplicate detection code based on jaccard similarity.

        Parameters
        ----------
        buckets : dict
            A dictionary mapping bucket IDs to lists of document indices.
        all_documents : list
            The input documents for similarity analysis.
        """
        correct_near_duplicates = 0
        all_near_duplicates = 0

        for bucket_id in buckets.keys():
            docs_in_this_bucket = buckets[bucket_id]
            unique_doc_ids = set(docs_in_this_bucket)
            if len(unique_doc_ids) > 1:
                combinations = list(itertools.combinations(unique_doc_ids, 2))
                for comb in combinations:
                    all_near_duplicates += 1

                    first_doc_id = comb[0]
                    second_doc_id = comb[1]

                    first_shingled_doc = self.shingle_document(all_documents[first_doc_id], 2)
                    second_shingled_doc = self.shingle_document(all_documents[second_doc_id], 2)

                    near_duplicated_jaccard_score = self.jaccard_score(first_shingled_doc, second_shingled_doc)
                    current_score = 0

                    for _ in range(5):
                        random_doc_id = first_doc_id
                        while random_doc_id == first_doc_id or random_doc_id == second_doc_id:
                            random_doc_id = random.randint(0, len(all_documents) - 1)
                        random_shingled_doc = self.shingle_document(all_documents[random_doc_id], 2)

                        random_jaccard_score = self.jaccard_score(first_shingled_doc, random_shingled_doc)

                        if near_duplicated_jaccard_score > random_jaccard_score:
                            current_score += 1

                    if current_score == 5:
                        correct_near_duplicates += 1

        # a good score is around 0.8
        print("your final score in near duplicate detection:", correct_near_duplicates / all_near_duplicates)


def test():
    j = None
    path = 'Logic/core/LSHFakeData.json'
    with open(path, 'r') as f:
        j = json.load(f)
        f.close()
    documents = []
    for m in j:
        documents.append(' '.join(m['summaries']))
    lsh = MinHashLSH(documents, 100)
    lsh.jaccard_similarity_test(lsh.perform_lsh(), documents)

# test()