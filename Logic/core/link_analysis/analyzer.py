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
        self.hub_map = {}
        self.authorities = []
        self.auth_map = {}
        self.initiate_params()

    def initiate_params(self):
        """
        Initialize links graph, hubs list and authorities list based of root set

        Parameters
        ----------
        This function has no parameters. You can use self to get or change attributes
        """
        for movie in self.root_set:
            m = self.graph.add_node(movie['title'])
            if m not in self.hub_map.values():
                self.hubs.append(1)
                self.hub_map[len(self.hubs) - 1] = m
            for star in movie['stars']:
                s = self.graph.add_node(star)
                if s not in self.auth_map.values():
                    self.authorities.append(1)
                    self.auth_map[len(self.authorities) - 1] = s
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
                        m = self.graph.add_node(movie['title'])
                        self.graph.add_edge(m, star)
                        if m not in self.hub_map.values():
                            self.hubs.append(1)
                            self.hub_map[len(self.hubs) - 1] = m
                        break
            for star in movie['stars']:
                for r in self.root_set:
                    if star in r['stars']:
                        self.graph.add_edge(r['title'], star)
                        self.graph.add_edge(r['title'], star)
                        if star not in self.auth_map.values():
                            self.authorities.append(1)
                            self.auth_map[len(self.authorities) - 1] = star
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
            for i in range(len(self.authorities)):
                self.authorities[i] = sum([self.hubs[j] if self.hub_map[j] in self.graph.g[self.auth_map[i]] else 0 for j in range(len(self.hubs))])
            for j in range(len(self.hubs)):
                self.hubs[j] = sum([self.authorities[i] if self.auth_map[i] in self.graph.g[self.hub_map[j]] else 0 for i in range(len(self.authorities))])
            self.authorities /= np.linalg.norm(self.authorities)
            self.hubs /= np.linalg.norm(self.hubs)

        a_s = sorted(range(len(self.authorities)), key=lambda i: self.authorities[i], reverse=True)[:10]
        h_s = sorted(range(len(self.hubs)), key=lambda i: self.hubs[i], reverse=True)[:10]

        return set([self.auth_map[i] for i in a_s]), set([self.hub_map[i] for i in h_s])

if __name__ == "__main__":
    # You can use this section to run and test the results of your link analyzer
    path = 'IMDB_crawled_give.json'
    data = None
    with open(path) as f:
        data = json.load(f)
    data = [d for d in data if d['stars'] is not None]
    corpus = data   # it shoud be your crawled data
    root_set = [c for c in corpus if 'godfather' in c['title'].lower()]   # it shoud be a subset of your corpus

    analyzer = LinkAnalyzer(root_set=root_set)
    analyzer.expand_graph(corpus=corpus)
    actors, movies = analyzer.hits(max_result=5)
    print("Top Actors:")
    print(*actors, sep=' - ')
    print("Top Movies:")
    print(*movies, sep=' - ')
