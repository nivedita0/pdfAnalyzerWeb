from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from .extractor import PDFExtractor
from .llm import LLMInterface

class RAGPipeline:
    """Retrieval-Augmented Generation pipeline for PDF documents"""
    
    def __init__(self, model_name="all-MiniLM-L6-v2", chunk_size=500):
        self.embedder = SentenceTransformer(model_name)
        self.extractor = PDFExtractor()
        self.llm = LLMInterface()
        self.chunk_size = chunk_size
        self.index = None
        self.chunks = []
        
    def process_document(self, pdf_path: str) -> None:
        """Process a document and prepare it for question answering"""
        # Extract text from PDF
        text = self.extractor.extract_text(pdf_path)
        
        # Split text into chunks
        self.chunks = self._split_text(text)
        
        # Create embeddings and index
        self._create_index()
        
    def _split_text(self, text: str) -> List[str]:
        """Split text into chunks of specified size"""
        words = text.split()
        return [" ".join(words[i:i+self.chunk_size]) for i in range(0, len(words), self.chunk_size)]
        
    def _create_index(self) -> None:
        """Create FAISS index from text chunks"""
        if not self.chunks:
            raise ValueError("No text chunks to index. Call process_document first.")

        embeddings = []
        batch_size = 32  # Adjust based on your memory
        for i in range(0, len(self.chunks), batch_size):
            batch = self.chunks[i:i+batch_size]
            batch_embeddings = self.embedder.encode(batch)
            embeddings.extend(batch_embeddings)
        embeddings = np.array(embeddings)
        dimension = embeddings.shape[1]
        
        # Create a new index
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        
    def search(self, query: str, top_k=3) -> List[str]:
        """Search for most relevant chunks to the query"""
        if not self.index:
            raise ValueError("No index available. Call process_document first.")
            
        # Create query embedding and search
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)
        _, indices = self.index.search(query_embedding, top_k)
        
        return [self.chunks[i] for i in indices[0]]
        
    def answer_question(self, question: str) -> str:
        """Answer a question about the document using RAG"""
        if not self.index:
            raise ValueError("No document has been processed yet.")
            
        # Get relevant chunks
        relevant_chunks = self.search(question)
        context = "\n".join(relevant_chunks)
        
        # Create prompt for LLM
        prompt = f"Answer the question based on the context below:\n\nContext:\n{context}\n\nQuestion: {question}"
        
        # Get answer from LLM
        answer = self.llm.generate(prompt)
        return answer