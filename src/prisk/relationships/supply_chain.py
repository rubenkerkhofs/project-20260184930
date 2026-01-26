"""
Class representing the relationship between two companies.
The main goal of this class is to facilitate the creation
of suppliers and clients of firms.
"""

from prisk.asset.asset import Asset

import networkx as nx


class SupplyChainRelation:
    """Facilitates the creation of supplier-client relationships,
    whilst saving information about the relation"""

    relations = []

    def __init__(self, 
                 client: Asset, 
                 supplier: Asset, 
                 product: str, 
                 recipe_input: float) -> None:
        self.client = client
        self.supplier = supplier
        self.product = product
        self.client.add_supplier(supplier, product, recipe_input)
        self.supplier.add_client(client)
        self.relations.append((self.client, self.supplier, recipe_input))

    def __str__(self) -> str:
        """String representation"""
        return f"{self.supplier} --> {self.client}"

    @classmethod
    def get_networkx_graph(cls) -> nx.DiGraph:
        """Get a NetworkX graph representation of the supply chain"""
        G = nx.DiGraph()
        for relation in cls.relations:
            G.add_edge(relation[1], relation[0], weight=relation[2])
        return G

    @classmethod
    def visualize(cls) -> None:
        """Visualize the supply chain relations"""
        network = cls.get_networkx_graph()
        pos = nx.spring_layout(network)
        nx.draw(
            network,
            pos,
            with_labels=True,
            node_size=500,
            node_color="skyblue",
            font_size=5,
        )
