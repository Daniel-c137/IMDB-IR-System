from .indexer.index_reader import Index_reader
from .indexer.indexes_enum import Indexes
import math
import json

class SpellCorrection:
    def __init__(self, all_documents, path='index/'):
        """
        Initialize the SpellCorrection

        Parameters
        ----------
        all_documents : list of str
            The input documents.
        """
        self.index = {
            Indexes.STARS: Index_reader(path, index_name=Indexes.STARS).index,
            Indexes.GENRES: Index_reader(path, index_name=Indexes.GENRES).index,
            Indexes.SUMMARIES: Index_reader(path, index_name=Indexes.SUMMARIES).index,
        }
        self.all_shingled_words, self.word_counter = self.shingling_and_counting(all_documents)

    def shingle_word(self, word, k=2):
        """
        Convert a word into a set of shingles.

        Parameters
        ----------
        word : str
            The input word.
        k : int
            The size of each shingle.

        Returns
        -------
        set
            A set of shingles.
        """
        if len(word) <= k:
            return set(word)
        shingles = set()
        for i in range(len(word) - k + 1):
            shingles.add(word[i:i+k])

        return shingles
    
    def tf_score(self, tf):
        return math.log10(tf)
    
    def jaccard_score(self, first_set, second_set):
        """
        Calculate jaccard score.

        Parameters
        ----------
        first_set : set
            First set of shingles.
        second_set : set
            Second set of shingles.

        Returns
        -------
        float
            Jaccard score.
        """

        return (len(first_set.intersection(second_set)) ** 1.8) / len(first_set.union(second_set))

    def shingling_and_counting(self, all_documents):
        """
        Shingle all words of the corpus and count TF of each word.

        Parameters
        ----------
        all_documents : list of str
            The input documents.

        Returns
        -------
        all_shingled_words : dict
            A dictionary from words to their shingle sets.
        word_counter : dict
            A dictionary from words to their TFs.
        """
        all_shingled_words = dict()
        word_counter = dict()

        def find_tf(word):
            total = 0
            for idx in [Indexes.SUMMARIES, Indexes.STARS, Indexes.GENRES]:
                try:
                    total += sum(self.index[idx][word].values())
                except:
                    # print(f'word {word} not found in {idx.value}.')
                    pass
            return total

        indexed_fields = [idx.value for idx in Indexes]
        self.max_cnt = 0
        self.min_cnt = math.inf
        for doc in all_documents:
            for field, words in doc.items():
                if field not in indexed_fields:
                    continue
                for w in words:
                    if not w.isdigit() and w not in all_shingled_words.keys():
                        all_shingled_words[w] = [self.shingle_word(w, k + 1) for k in range(3)]
                        cnt = find_tf(w)
                        word_counter[w] = cnt
                        self.max_cnt = max(cnt, self.max_cnt)
                        self.min_cnt = min(cnt, self.min_cnt)

        word_counter = dict(sorted(word_counter.items(), key=lambda item: -1 * item[1]))
                
        return all_shingled_words, word_counter
    
    def find_nearest_words(self, word):
        """
        Find correct form of a misspelled word.

        Parameters
        ----------
        word : stf
            The misspelled word.

        Returns
        -------
        list of str
            5 nearest words.
        """
        top5_candidates = [(None, 0) for _ in range(5)]
        min_id = 0
        for t, tf in self.word_counter.items():
            if word == t:
                return [(t, 100)]
            score = 0
            word_shingles = self.all_shingled_words[t]
            for k in range(3):
                shingles = self.shingle_word(word, k + 1)
                score += self.jaccard_score(shingles, word_shingles[k])
            score *= self.tf_score(tf)
            if score > top5_candidates[min_id][1]:
                top5_candidates[min_id] = (t, score)
                min_id = top5_candidates.index(min(top5_candidates, key=lambda a: a[1]))
        
        r =  list(sorted(top5_candidates, key=lambda item: item[1], reverse=True))
        return r
    
    def spell_check(self, query):
        """
        Find correct form of a misspelled query.

        Parameters
        ----------
        query : stf
            The misspelled query.

        Returns
        -------
        str
            Correct form of the query.
        """
        final_result = []
        
        for word in query.split():
            result = self.find_nearest_words(word)
            final_result.append(result[0][0])

        return ' '.join(final_result)