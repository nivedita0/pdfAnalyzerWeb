# from typing import List, Dict, Any, Optional
# import numpy as np
# import faiss
# from sentence_transformers import SentenceTransformer
# import os
# import json
# from .extractor import PDFExtractor
# from .llm import LLMInterface
# import re
# import uuid


# class MultiDocumentRAG:
#     """Enhanced RAG system supporting multiple research papers"""
    
#     def __init__(self, 
#                  embedding_model: str = "all-MiniLM-L6-v2", 
#                  chunk_size: int = 300,
#                  chunk_overlap: int = 50,
#                  llm_backend: str = "ollama",
#                  llm_model: str = "llama3"):
#         self.embedder = SentenceTransformer(embedding_model)
#         self.extractor = PDFExtractor()
#         self.llm = LLMInterface(default_model=llm_model, backend=llm_backend)
        
#         self.chunk_size = chunk_size
#         self.chunk_overlap = chunk_overlap
#         self.index = None
#         self.documents = []
#         self.chunks = []
#         self.chunk_documents = []  # Maps each chunk to its source document
#         self.metadata = {}  # Store document metadata
        
#     def add_document(self, pdf_path: str, document_id: Optional[str] = None) -> str:
#         """Process a document and add it to the knowledge base"""
#         if document_id is None:
#             document_id = str(uuid.uuid4())
            
#         # Extract text and metadata
#         text = self.extractor.extract_text(pdf_path)
        
#         # Try to extract metadata
#         try:
#             meta = self.extractor.extract_metadata(pdf_path)
#         except:
#             # Use filename as fallback
#             filename = os.path.basename(pdf_path)
#             meta = {"title": filename, "filename": filename}
        
#         # Store document
#         self.documents.append({
#             "id": document_id,
#             "path": pdf_path,
#             "text": text
#         })
        
#         self.metadata[document_id] = meta
        
#         # Process document into chunks
#         self._process_document(document_id, text)
        
#         # Rebuild index if needed
#         if self.chunks:
#             self._build_index()
            
#         return document_id
    
#     def _process_document(self, document_id: str, text: str) -> None:
#         """Split document into chunks with overlap"""
#         # Split text into sentences for more natural chunks
#         sentences = re.split(r'(?<=[.!?])\s+', text)
        
#         current_chunk = []
#         current_length = 0
        
#         for sentence in sentences:
#             words = sentence.split()
#             sentence_length = len(words)
            
#             # If adding this sentence exceeds chunk size and we have content
#             if current_length + sentence_length > self.chunk_size and current_chunk:
#                 # Store the current chunk
#                 chunk_text = " ".join(current_chunk)
#                 self.chunks.append(chunk_text)
#                 self.chunk_documents.append(document_id)
                
#                 # Start new chunk with overlap
#                 overlap_words = current_chunk[-self.chunk_overlap:] if self.chunk_overlap > 0 else []
#                 current_chunk = overlap_words + [sentence]
#                 current_length = len(overlap_words) + sentence_length
#             else:
#                 # Add sentence to current chunk
#                 current_chunk.append(sentence)
#                 current_length += sentence_length
        
#         # Add the last chunk if not empty
#         if current_chunk:
#             chunk_text = " ".join(current_chunk)
#             self.chunks.append(chunk_text)
#             self.chunk_documents.append(document_id)
    
#     def _build_index(self) -> None:
#         """Build or update the FAISS index with document chunks"""
#         # Generate embeddings for all chunks
#         embeddings = []
#         batch_size = 16  # Adjust based on your memory
#         for i in range(0, len(self.chunks), batch_size):
#             batch = self.chunks[i:i+batch_size]
#             import gc
#             gc.collect()
#             batch_embeddings = self.embedder.encode(batch)
#             embeddings.extend(batch_embeddings)
#         embeddings = np.array(embeddings)
#         dimension = embeddings.shape[1]
        
#         # Create new index
#         self.index = faiss.IndexFlatL2(dimension)
#         self.index.add(embeddings)
    
#     def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
#         """Search for most relevant chunks to the query"""
#         if not self.index:
#             raise ValueError("No documents indexed yet")
            
#         # Encode query
#         query_embedding = self.embedder.encode([query], convert_to_numpy=True)
        
#         # Search index
#         distances, indices = self.index.search(query_embedding, top_k)
        
#         # Format results
#         results = []
#         for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
#             if idx < len(self.chunks):  # Valid index
#                 doc_id = self.chunk_documents[idx]
#                 results.append({
#                     "chunk": self.chunks[idx],
#                     "document_id": doc_id,
#                     "document_title": self.metadata[doc_id].get("title", "Unknown"),
#                     "rank": rank + 1,
#                     "distance": float(dist)
#                 })
                
#         return results
    
#     def answer_question(self, question: str, 
#                         top_k: int = 5, 
#                         add_metadata: bool = True) -> Dict[str, Any]:
#         """Answer a question using the RAG pipeline"""
#         # Retrieve relevant chunks
#         results = self.search(question, top_k=top_k)
        
#         if not results:
#             return {
#                 "answer": "I don't have enough information to answer that question.",
#                 "sources": []
#             }
        
#         # Format context for LLM
#         context = ""
#         sources = []
        
#         for i, result in enumerate(results):
#             # Add chunk with document info
#             context += f"\nPassage {i+1} [From: {result['document_title']}]:\n{result['chunk']}\n"
            
#             # Track source for citation
#             if result["document_id"] not in [s["id"] for s in sources]:
#                 sources.append({
#                     "id": result["document_id"],
#                     "title": result["document_title"],
#                     **self.metadata[result["document_id"]]
#                 })
        
#         # Create prompt for LLM
#         prompt = f"""Answer the question based only on the following passages from research papers.
        
# Context:
# {context}

# Question: {question}

# Instructions:
# 1. Answer based ONLY on the information in the passages.
# 2. If the passages don't contain enough information to answer, say "The provided documents don't contain enough information to answer this question."
# 3. Don't make up information not found in the passages.
# 4. Be precise and technical when appropriate.
# 5. Cite the passage numbers [e.g., Passage 1, Passage 2] when referring to specific information.
# """

#         # Get answer from LLM
#         answer = self.llm.generate(prompt)
        
#         return {
#             "answer": answer,
#             "sources": sources,
#             "retrieved_chunks": results
#         }
    
#     def semantic_search(self, query: str, top_k: int = 10) -> Dict[str, Any]:
#         """Perform semantic search across all documents"""
#         results = self.search(query, top_k=top_k)
        
#         # Group results by document
#         docs = {}
#         for result in results:
#             doc_id = result["document_id"]
#             if doc_id not in docs:
#                 docs[doc_id] = {
#                     "id": doc_id,
#                     "title": result["document_title"],
#                     "chunks": [],
#                     "metadata": self.metadata[doc_id],
#                     "relevance": 0
#                 }
            
#             # Add chunk and update document relevance (inverse of distance)
#             docs[doc_id]["chunks"].append({
#                 "text": result["chunk"],
#                 "rank": result["rank"],
#                 "relevance": 1.0 / (1.0 + result["distance"])
#             })
            
#             docs[doc_id]["relevance"] += 1.0 / (1.0 + result["distance"])
        
#         # Sort documents by relevance
#         sorted_docs = sorted(docs.values(), key=lambda x: x["relevance"], reverse=True)
        
#         return {
#             "query": query,
#             "documents": sorted_docs
#         }
    
#     def save_knowledge_base(self, output_dir: str) -> None:
#         """Save the knowledge base to disk"""
#         os.makedirs(output_dir, exist_ok=True)
        
#         # Save document metadata
#         with open(os.path.join(output_dir, "metadata.json"), "w") as f:
#             json.dump(self.metadata, f, indent=2)
        
#         # Save chunks and document mappings
#         with open(os.path.join(output_dir, "chunks.json"), "w") as f:
#             json.dump({
#                 "chunks": self.chunks,
#                 "chunk_documents": self.chunk_documents
#             }, f, indent=2)
        
#         # Save FAISS index
#         if self.index:
#             faiss.write_index(self.index, os.path.join(output_dir, "index.faiss"))
    
#     def load_knowledge_base(self, input_dir: str) -> None:
#         """Load a previously saved knowledge base"""
#         # Load metadata
#         with open(os.path.join(input_dir, "metadata.json"), "r") as f:
#             self.metadata = json.load(f)
        
#         # Load chunks
#         with open(os.path.join(input_dir, "chunks.json"), "r") as f:
#             chunks_data = json.load(f)
#             self.chunks = chunks_data["chunks"]
#             self.chunk_documents = chunks_data["chunk_documents"]
        
#         # Load FAISS index
#         index_path = os.path.join(input_dir, "index.faiss")
#         if os.path.exists(index_path):
#             self.index = faiss.read_index(index_path)