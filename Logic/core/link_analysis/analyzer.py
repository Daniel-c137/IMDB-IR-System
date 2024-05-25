import numpy as np
from .graph import LinkGraph
from ..indexer.indexes_enum import Indexes
from ..indexer.index_reader import Index_reader

class LinkAnalyzer:
    def __init__(self, root_set):
        """
        Initialize the Link Analyzer attributes:

        Parameters
        ----------
        root_set: list
            A list of movie dictionaries with the following keys:
            "id": A unique ID for the movie
            "title": string of movie title
            "stars": A list of movie star names
        """
        self.root_set = root_set
        self.graph = LinkGraph()
        self.hubs = []
        self.authorities = []
        self.initiate_params()

    def initiate_params(self):
        """
        Initialize links graph, hubs list and authorities list based of root set

        Parameters
        ----------
        This function has no parameters. You can use self to get or change attributes
        """
        for movie in self.root_set:
            m = self.graph.add_node(movie['id'])
            self.hubs.append(m)
            for star in movie['stars']:
                s = self.graph.add_node(star)
                self.authorities.append(s)
                self.graph.add_edge(m, s)

    def expand_graph(self, corpus):
        """
        expand hubs, authorities and graph using given corpus

        Parameters
        ----------
        corpus: list
            A list of movie dictionaries with the following keys:
            "id": A unique ID for the movie
            "stars": A list of movie star names

        Note
        ---------
        To build the base set, we need to add the hubs and authorities that are inside the corpus
        and refer to the nodes in the root set to the graph and to the list of hubs and authorities.
        """
        for movie in corpus:
            for r in self.root_set:
                for star in r['stars']:
                    if star in movie['stars']:
                        m = self.graph.add_node(movie['id'])
                        self.graph.add_edge(m, star)
                        self.hubs.append(m)
                        break

            for star in movie['stars']:
                for r in self.root_set:
                    if star in r['stars']:
                        s = self.graph.add_edge(star)
                        self.graph.add_edge(r['id'], s)
                        self.authorities.append(m)
                        break

    def hits(self, num_iteration=5, max_result=10):
        """
        Return the top movies and actors using the Hits algorithm

        Parameters
        ----------
        num_iteration: int
            Number of algorithm execution iterations
        max_result: int
            The maximum number of results to return. If None, all results are returned.

        Returns
        -------
        list
            List of names of 10 actors with the most scores obtained by Hits algorithm in descending order
        list
            List of names of 10 movies with the most scores obtained by Hits algorithm in descending order
        """
        a_s = []
        h_s = []

        for _ in range(num_iteration):
            for i,a in enumerate(self.authorities):
                self.authorities[i] = sum([self.authorities[j] if self.graph.get(i, j) == 1 else 0 for j in range(len(self.authorities))])
                pass
            for j,h in enumerate(self.hubs):
                self.hubs[j] = sum([self.hubs[i] if self.graph.get(i, j) == 1 else 0 for i in range(len(self.hubs))])
            self.authorities /= np.linalg.norm(self.authorities)
            self.hubs /= np.linalg.norm(self.hubs)
        
        a_s = sorted(range(len(self.authorities)), key=lambda i: self.authorities[i], reverse=True)[:10]
        h_s = sorted(range(len(self.hubs)), key=lambda i: self.hubs[i], reverse=True)[:10]

        return a_s, h_s

if __name__ == "__main__":
    # You can use this section to run and test the results of your link analyzer
    corpus = [] ‍   # TODO: it shoud be your crawled data
    root_set = []   # TODO: it shoud be a subset of your corpus

    analyzer = LinkAnalyzer(root_set=root_set)
    analyzer.expand_graph(corpus=corpus)
    actors, movies = analyzer.hits(max_result=5)
    print("Top Actors:")
    print(*actors, sep=' - ')
    print("Top Movies:")
    print(*movies, sep=' - ')
