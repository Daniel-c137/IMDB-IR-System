import networkx as nx

class LinkGraph:
    """
    Use this class to implement the required graph in link analysis.
    You are free to modify this class according to your needs.
    You can add or remove methods from it.
    """
    def __init__(self):
        self.g = {}

    def add_edge(self, u_of_edge, v_of_edge):
        if u_of_edge not in self.g.keys():
            self.g[u_of_edge] = set()
        if v_of_edge not in self.g.keys():
            self.g[v_of_edge] = set()
        self.g[u_of_edge].add(v_of_edge)
        self.g[v_of_edge].add(u_of_edge)

    def add_node(self, node_to_add):
        if node_to_add not in self.g.keys():
            self.g[node_to_add] = set()
        return node_to_add

    def get_successors(self, node):
        pass

    def get_predecessors(self, node):
        pass
