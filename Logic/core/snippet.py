import re
import json

class Snippet:
    def __init__(self, number_of_words_on_each_side=5):
        """
        Initialize the Snippet

        Parameters
        ----------
        number_of_words_on_each_side : int
            The number of words on each side of the query word in the doc to be presented in the snippet.
        """
        with open('Logic/core/stopwords.txt', 'r') as f:
            self.stopwords = set([w[:-1] for w in f.readlines()])
        self.number_of_words_on_each_side = number_of_words_on_each_side

    def remove_stop_words_from_query(self, query):
        """
        Remove stop words from the input string.

        Parameters
        ----------
        query : str
            The query that you need to delete stop words from.

        Returns
        -------
        str
            The query without stop words.
        """

        for word in self.stopwords:
            query = query.replace(word, '')
        return re.sub(r'\s\s+', ' ', query)

        return

    def find_snippet(self, doc, query):
        """
        Find snippet in a doc based on a query.

        Parameters
        ----------
        doc : str
            The retrieved doc which the snippet should be extracted from that.
        query : str
            The query which the snippet should be extracted based on that.

        Returns
        -------
        final_snippet : str
            The final extracted snippet. IMPORTANT: The keyword should be wrapped by *** on both sides.
            For example: Sahwshank ***redemption*** is one of ... (for query: redemption)
        not_exist_words : list
            Words in the query which don't exist in the doc.
        """
        final_snippet = ""
        not_exist_words = []
        genre_found = False
        title_found = False
        token_intervals = {}

        title = ' '.join(doc['title'])
        genres = ' '.join(doc['genres'])
        summs = doc['summaries']

        qs = set(self.remove_stop_words_from_query(query).split())
        for q in qs:
            f = False
            if q in doc['title']:
                f = True
                title_found = True
                title = title.replace(q, f'**{q}**')
            if q in doc['genres']:
                f = True
                genre_found = True
                genres = genres.replace(q, f'**{q}**')
            if q in summs:
                f = True
                intervals = [(max(i - self.number_of_words_on_each_side,0), min(i + self.number_of_words_on_each_side, len(summs))) for i, x in enumerate(summs) if x == q]
                best_interval = (None, -1)
                for begin, end in intervals:
                    unique_qs = set(q)
                    for w in summs[begin:end]:
                        if w in qs:
                            unique_qs.add(w)
                    score = len(unique_qs)
                    if best_interval[1] < score:
                        best_interval = ((begin, end), score)
                token_intervals[q] = best_interval[0]
            if not f:
                not_exist_words.append(q)  
            
        if title_found:
            final_snippet = 'title: ' + title + ' ... '
        if genre_found:
            final_snippet = final_snippet + 'genres: ' + genres + ' ... '
        token_intervals = dict(sorted(token_intervals.items(), key=lambda x: x[1][0]))
        prev_end = None
        for begin, end in token_intervals.values():
            print(begin, end)
            b = begin
            if prev_end and prev_end >= begin:
                if prev_end < len(summs):
                    b =  prev_end + 1
            final_snippet = final_snippet + ' '.join([f'**{s}**' if s in qs else s for s in summs[b: end]]) + ' ... '
            prev_end = end

        return final_snippet, not_exist_words
