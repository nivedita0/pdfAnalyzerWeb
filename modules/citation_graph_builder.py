import os
from modules.reference_enricher import enrich_references
import networkx as nx
import matplotlib.pyplot as plt

def get_pdf_files(folder_path):
    return [f for f in os.listdir(folder_path) if f.endswith(".pdf")]

def build_citation_edges(folder_path):
    files = get_pdf_files(folder_path)
    paper_refs = {}

    print("\n[+] Processing all PDFs...\n")
    for f in files:
        path = os.path.join(folder_path, f)
        enriched = enrich_references(path)
        paper_refs[f] = enriched

    edges = []

    for source, refs in paper_refs.items():
        for ref in refs:
            cited_doi = ref['metadata']['doi'] if ref['metadata'] else None
            if not cited_doi:
                continue

            for target, t_refs in paper_refs.items():
                if target == source:
                    continue
                # Check if the other paper has this DOI in its metadata
                target_doi_list = [r['metadata']['doi'] for r in t_refs if r['metadata']]
                if cited_doi in target_doi_list:
                    edges.append((source, target))
    
    return edges, files

def draw_citation_graph(edges, files):
    G = nx.DiGraph()
    G.add_nodes_from(files)
    G.add_edges_from(edges)

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=2500, node_color="lightgreen", font_size=9, arrows=True)
    plt.title("Citation Graph Across Uploaded Papers")
    plt.show()
