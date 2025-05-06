import os
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import matplotlib.pyplot as plt
import umap
import hdbscan
from .extractor import PDFExtractor
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


class TopicModeler:
    """Extract and visualize topics across multiple research papers"""
    
    def __init__(self, n_topics=10, embedding_model="all-MiniLM-L6-v2"):
        self.extractor = PDFExtractor()
        self.n_topics = n_topics
        self.papers = []
        self.paper_contents = []
        self.vectorizer = None
        self.topic_model = None
        self.doc_topic_matrix = None
        self.feature_names = None
        
        # Import here to avoid loading at module level
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer(embedding_model)
        except ImportError:
            print("Warning: SentenceTransformer not installed. Some features may be limited.")
            self.embedder = None
        
    def add_paper(self, pdf_path: str) -> None:
        """Add a paper to the topic model"""
        # Extract text from the paper
        text = self.extractor.extract_text(pdf_path)
        
        # Get metadata
        metadata = {}
        try:
            metadata = self.extractor.extract_metadata(pdf_path)
        except:
            # If metadata extraction fails, use filename
            filename = os.path.basename(pdf_path)
            metadata = {"title": filename, "filename": filename}
        
        # Store paper information
        self.papers.append({
            "path": pdf_path,
            "metadata": metadata
        })
        
        self.paper_contents.append(text)
    
    def build_topic_model(self, method="lda") -> None:
        """Build topic model from added papers"""
        if not self.papers:
            raise ValueError("No papers added to the model")
            
        # Create document-term matrix
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.85,
            stop_words='english'
        )
        
        dtm = self.vectorizer.fit_transform(self.paper_contents)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        # Build topic model
        if method == "lda":
            self.topic_model = LatentDirichletAllocation(
                n_components=self.n_topics,
                max_iter=10,
                learning_method='online',
                random_state=42
            )
        elif method == "nmf":
            self.topic_model = NMF(
                n_components=self.n_topics,
                random_state=42,
                alpha=.1,
                l1_ratio=.5
            )
        else:
            raise ValueError(f"Unknown topic modeling method: {method}")
            
        self.doc_topic_matrix = self.topic_model.fit_transform(dtm)
        
        # Add dominant topics to papers
        for i, paper in enumerate(self.papers):
            dominant_topic = np.argmax(self.doc_topic_matrix[i])
            paper["dominant_topic"] = int(dominant_topic)
            paper["topic_distribution"] = self.doc_topic_matrix[i].tolist()
    
    def get_topic_keywords(self, n_words=10) -> List[List[str]]:
        """Get top keywords for each topic"""
        if self.topic_model is None:
            raise ValueError("Topic model not built yet")
            
        keywords = []
        for topic_idx, topic in enumerate(self.topic_model.components_):
            # Get top words for this topic
            top_features_ind = topic.argsort()[:-n_words-1:-1]
            top_features = [self.feature_names[i] for i in top_features_ind]
            keywords.append(top_features)
            
        return keywords
    
    def visualize_topic_distribution(self, output_path="topic_distribution.html") -> str:
        """Create interactive visualization of topic distribution"""
        if self.topic_model is None:
            raise ValueError("Topic model not built yet")
            
        # Prepare data for visualization
        data = []
        for i, paper in enumerate(self.papers):
            paper_data = {
                "title": paper["metadata"].get("title", f"Paper {i}"),
                "dominant_topic": paper["dominant_topic"]
            }
            
            # Add topic distribution
            for topic_idx, weight in enumerate(paper["topic_distribution"]):
                paper_data[f"Topic {topic_idx+1}"] = weight
                
            data.append(paper_data)
            
        df = pd.DataFrame(data)
        
        # Create stacked bar chart
        topic_cols = [f"Topic {i+1}" for i in range(self.n_topics)]
        fig = px.bar(
            df,
            x="title",
            y=topic_cols,
            title="Topic Distribution Across Papers",
            labels={"value": "Topic Weight", "variable": "Topic"},
            height=600
        )
        
        fig.update_layout(
            xaxis_title="Paper",
            yaxis_title="Topic Weight",
            barmode='stack',
            xaxis={'categoryorder':'total descending'}
        )
        
        # Save the plot
        fig.write_html(output_path)
        return output_path
        
    def visualize_topic_clusters(self, output_path="topic_clusters.html") -> str:
        """Create 2D visualization of paper clusters by topic"""
        if self.embedder is None:
            raise ValueError("SentenceTransformer not available")
            
        if not self.papers:
            raise ValueError("No papers added to the model")
            
        # Get embeddings for each paper (using abstracts or first chunk)
        contents = []
        for paper in self.papers:
            sections = self.extractor.extract_sections(paper["path"])
            if "abstract" in sections:
                contents.append(sections["abstract"])
            else:
                # Use first 1000 characters if no abstract
                contents.append(self.paper_contents[self.papers.index(paper)][:1000])
                
        # Create embeddings
        embeddings = self.embedder.encode(contents)
        
        # Reduce dimensionality for visualization
        reducer = umap.UMAP(n_components=2, random_state=42)
        reduced_embeddings = reducer.fit_transform(embeddings)
        
        # Create dataframe for plotting
        df = pd.DataFrame({
            'x': reduced_embeddings[:,0],
            'y': reduced_embeddings[:,1],
            'title': [p["metadata"].get("title", f"Paper {i}") for i, p in enumerate(self.papers)],
            'topic': [p.get("dominant_topic", 0) for p in self.papers]
        })
        
        # Create interactive scatter plot
        fig = px.scatter(
            df, 
            x='x', 
            y='y',
            color='topic',
            hover_data=['title'],
            title='Research Paper Topic Clusters',
            labels={'topic': 'Dominant Topic'},
            color_continuous_scale=px.colors.qualitative.G10
        )
        
        fig.update_layout(
            xaxis_title="",
            yaxis_title="",
            xaxis={'showticklabels': False},
            yaxis={'showticklabels': False}
        )
        
        # Save the plot
        fig.write_html(output_path)
        return output_path