import re
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Set, Tuple, Any
from .extractor import PDFExtractor


class CitationNetwork:
    """Extract and analyze citation networks from research papers"""
    
    def __init__(self):
        self.extractor = PDFExtractor()
        self.papers = {}  # Store paper metadata
        self.graph = nx.DiGraph()  # Citation graph
        
    def extract_citations(self, pdf_path: str) -> List[Dict[str, str]]:
        """Extract citations from a research paper"""
        # Get sections from paper
        sections = self.extractor.extract_sections(pdf_path)
        
        # Extract paper metadata
        metadata = self.extractor.extract_metadata(pdf_path)
        paper_id = metadata.get("doi", pdf_path.split("/")[-1])
        self.papers[paper_id] = metadata
        
        # Add node to graph
        if paper_id not in self.graph.nodes():
            self.graph.add_node(paper_id, **metadata)
        
        # Process references section if available
        citations = []
        if "references" in sections:
            references_text = sections["references"]
            # Split references by common patterns
            ref_list = self._split_references(references_text)
            
            for ref in ref_list:
                citation = self._parse_citation(ref)
                if citation:
                    citations.append(citation)
                    # Create edge in graph if citation has identifier
                    if "doi" in citation:
                        cited_id = citation["doi"]
                        if cited_id not in self.graph.nodes():
                            self.graph.add_node(cited_id, **citation)
                        self.graph.add_edge(paper_id, cited_id)
        
        return citations
    
    def _split_references(self, references_text: str) -> List[str]:
        """Split references text into individual references"""
        # Try to split by numbered references [1], [2], etc.
        numbered_pattern = r"\[\d+\]|\(\d+\)"
        splits = re.split(numbered_pattern, references_text)
        
        # If we got reasonable splits, use them
        if len(splits) > 3:
            return [s.strip() for s in splits if s.strip()]
        
        # Otherwise try to split by newlines with heuristics
        lines = references_text.split('\n')
        references = []
        current_ref = ""
        
        for line in lines:
            if not line.strip():
                continue
                
            # New reference likely starts with author names
            if (re.match(r"^[A-Z][a-z]+,", line) or 
                re.match(r"^[A-Z][a-z]+ [A-Z]\.", line) or
                re.match(r"^[A-Z][a-z]+ et al\.", line)):
                if current_ref:
                    references.append(current_ref)
                current_ref = line
            else:
                current_ref += " " + line
                
        # Add the last reference
        if current_ref:
            references.append(current_ref)
            
        return references
    
    def _parse_citation(self, citation_text: str) -> Dict[str, str]:
        """Parse citation text into structured data"""
        result = {}
        
        # Extract DOI if present
        doi_match = re.search(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", citation_text, re.IGNORECASE)
        if doi_match:
            result["doi"] = doi_match.group(0)
            
        # Extract year
        year_match = re.search(r"\((\d{4})\)", citation_text)
        if year_match:
            result["year"] = year_match.group(1)
            
        # Extract title (everything between quotes, if present)
        title_match = re.search(r'"([^"]+)"', citation_text)
        if title_match:
            result["title"] = title_match.group(1)
        else:
            # Try to extract title heuristically
            # (after author names and before journal)
            parts = citation_text.split(".")
            if len(parts) >= 2:
                result["title"] = parts[1].strip()
                
        # Extract authors
        # This is a simplified approach - more sophisticated parsing would be needed
        if citation_text.strip():
            first_part = citation_text.split(".")[0]
            result["authors"] = first_part.strip()
            
        return result
    
    def visualize_network(self, output_path: str = "citation_network.png") -> None:
        """Generate visualization of the citation network"""
        plt.figure(figsize=(12, 8))
        
        # Use different node sizes based on in-degree (citation count)
        sizes = [100 + 20 * self.graph.in_degree(n) for n in self.graph.nodes()]
        
        pos = nx.spring_layout(self.graph)
        nx.draw_networkx(
            self.graph, 
            pos=pos,
            with_labels=False,
            node_size=sizes,
            node_color="skyblue",
            edge_color="gray",
            alpha=0.7
        )
        
        # Add labels for important nodes (most cited)
        important_nodes = sorted(self.graph.nodes(), 
                              key=lambda n: self.graph.in_degree(n), 
                              reverse=True)[:10]
        labels = {n: self.graph.nodes[n].get("title", n) for n in important_nodes}
        nx.draw_networkx_labels(self.graph, pos, labels=labels, font_size=8)
        
        plt.title("Citation Network")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        return output_path
    
    def get_influential_papers(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most influential papers based on citation count"""
        if not self.graph.nodes():
            return []
            
        # Sort by in-degree (number of citations)
        influential = sorted(self.graph.nodes(), 
                          key=lambda n: self.graph.in_degree(n), 
                          reverse=True)[:limit]
                          
        return [
            {
                "id": paper_id,
                "citation_count": self.graph.in_degree(paper_id),
                **self.graph.nodes[paper_id]
            }
            for paper_id in influential
        ]