import networkx as nx
import matplotlib.pyplot as plt

def plot_citation_graph(citations_dict):
    G = nx.DiGraph()
    for citing, cited_list in citations_dict.items():
        for cited in cited_list:
            G.add_edge(citing, cited)
    
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=3000, font_size=10, arrowsize=20)
    plt.title("Citation Network")
    plt.show()
