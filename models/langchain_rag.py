import os
from typing import List, Dict, Any, Optional
import uuid
from tqdm import tqdm
import csv
import hashlib
from datetime import datetime

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
#from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain_ollama import OllamaLLM as Ollama 

# Local imports
from .extractor import PDFExtractor

class LangChainRAG:
    """RAG implementation using LangChain components"""
    
    def __init__(self, 
                 embedding_model: str = "paraphrase-MiniLM-L3-v2", 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 llm_model: str = "llama3"):
        
        self.extractor = PDFExtractor()
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.llm_model = llm_model
        
        # Initialize embedding model
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        # Will store document info
        self.documents = {}
        self.vectorstore = None
        
        # Create directory for FAISS index
        os.makedirs("faiss_index", exist_ok=True)
    def add_document(self, pdf_path: str, document_id: Optional[str] = None) -> str:
        """Process a document and add it to the knowledge base with detailed logging"""
        if document_id is None:
            document_id = str(uuid.uuid4())
            
        try:
            print(f"\nADDING DOCUMENT: {os.path.basename(pdf_path)}")
            print(f"Document ID: {document_id}")
            
            # Extract document pieces with metadata
            doc_chunks = self._extract_pdf_chunks(pdf_path)
            print(f"Extracted {len(doc_chunks)} chunks from document")
            
            # Extract metadata
            metadata = self.extractor.extract_metadata(pdf_path)
            print("\nDOCUMENT METADATA:")
            for key, value in metadata.items():
                print(f"  {key}: {value}")
            
            # Save all extracted content to a single file for reference
            content_file = os.path.join("extraction_logs", f"full_content_{document_id}.txt")
            with open(content_file, "w", encoding="utf-8") as f:
                f.write(f"FULL CONTENT FOR: {pdf_path}\n")
                f.write(f"Document ID: {document_id}\n")
                f.write("="*50 + "\n\n")
                
                # Write metadata
                f.write("METADATA:\n")
                for key, value in metadata.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n" + "="*50 + "\n\n")
                
                # Write all chunk content
                for i, chunk in enumerate(doc_chunks):
                    f.write(f"CHUNK {i+1}:\n")
                    f.write(f"Page: {chunk.metadata.get('page', 'Unknown')}\n")
                    f.write("-"*30 + "\n")
                    f.write(chunk.page_content)
                    f.write("\n\n" + "-"*50 + "\n\n")
            
            print(f"Saved full document content to {content_file}")
            
            # Add document info
            self.documents[document_id] = {
                "id": document_id,
                "path": pdf_path,
                "chunks_count": len(doc_chunks)
            }
            self.documents[document_id].update(metadata)
            
            # Update chunk metadata with document info
            for chunk in doc_chunks:
                chunk.metadata["document_id"] = document_id
                chunk.metadata["title"] = metadata.get("title", os.path.basename(pdf_path))
                
            print("Creating text splitter for semantic chunking...")
            # Create chunks with RecursiveCharacterTextSplitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size, 
                chunk_overlap=self.chunk_overlap
            )
            chunks = text_splitter.split_documents(doc_chunks)
            print(f"Split into {len(chunks)} final chunks for indexing")
            
            # Save processed chunks for reference
            processed_file = os.path.join("extraction_logs", f"processed_chunks_{document_id}.txt")
            with open(processed_file, "w", encoding="utf-8") as f:
                f.write(f"PROCESSED CHUNKS FOR: {pdf_path}\n")
                f.write(f"Document ID: {document_id}\n")
                f.write(f"Total chunks: {len(chunks)}\n")
                f.write("="*50 + "\n\n")
                
                for i, chunk in enumerate(chunks):
                    f.write(f"PROCESSED CHUNK {i+1}:\n")
                    f.write(f"Page: {chunk.metadata.get('page', 'Unknown')}\n")
                    f.write("-"*30 + "\n")
                    f.write(chunk.page_content[:300])  # First 300 chars
                    f.write(f"\n...(total length: {len(chunk.page_content)} characters)")
                    f.write("\n\n" + "-"*50 + "\n\n")
            
            print(f"Saved processed chunks to {processed_file}")
            
            # Create or update vectorstore
            print("Creating embeddings and vector store...")
            if self.vectorstore is None:
                self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
                print("Created new FAISS vector store")
            else:
                self.vectorstore.add_documents(chunks)
                print("Updated existing FAISS vector store")
                
            # Save index
            self.vectorstore.save_local("faiss_index")
            print("Saved FAISS index to disk")
            
            return document_id
        except Exception as e:
            print(f"Error processing document: {e}")
            raise   

    def _extract_pdf_chunks(self, pdf_path: str) -> List[Document]:
        """Extract content from PDF including text and tables with detailed logging"""
        print("\n" + "="*50)
        print(f"EXTRACTING DATA FROM: {os.path.basename(pdf_path)}")
        print("="*50)
        
        try:
            # Try using pdfplumber for better table extraction
            import pdfplumber
            documents = []
            
            with pdfplumber.open(pdf_path) as pdf:
                print(f"PDF has {len(pdf.pages)} pages")
                
                # Create a log file to save all extracted content
                log_dir = "extraction_logs"
                os.makedirs(log_dir, exist_ok=True)
                log_file = os.path.join(log_dir, f"extraction_{os.path.basename(pdf_path)}.txt")
                
                with open(log_file, "w", encoding="utf-8") as log:
                    log.write(f"EXTRACTION LOG FOR: {pdf_path}\n")
                    log.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    log.write("="*50 + "\n\n")
                    
                    for i, page in enumerate(pdf.pages):
                        print(f"Processing page {i+1}/{len(pdf.pages)}")
                        
                        # Extract text
                        text = page.extract_text() or ""
                        print(f"Page {i+1} text length: {len(text)} characters")
                        log.write(f"PAGE {i+1} TEXT:\n{text}\n\n")
                        
                        # Extract tables
                        tables = page.extract_tables()
                        print(f"Page {i+1} tables found: {len(tables)}")
                        
                        table_texts = []
                        for idx, table in enumerate(tables):
                            print(f"  Table {idx+1} dimensions: {len(table)}x{len(table[0]) if table and len(table) > 0 else 0}")
                            
                            # Format table for logging
                            table_str = "\n".join([
                                "\t".join([str(cell) if cell is not None else "" for cell in row])
                                for row in table
                            ])
                            table_texts.append(f"Table {idx+1}:\n{table_str}")
                            
                            # Log the table
                            log.write(f"PAGE {i+1} TABLE {idx+1}:\n{table_str}\n\n")

                        # Combine text and tables
                        combined_text = text
                        if table_texts:
                            combined_text += "\n\nTables found on page:\n" + "\n\n".join(table_texts)
                        
                        # Create document with metadata
                        metadata = {"page": i + 1, "source": pdf_path}
                        document = Document(page_content=combined_text, metadata=metadata)
                        documents.append(document)
                        
                        # Log the complete document content
                        log.write(f"PAGE {i+1} COMBINED DOCUMENT:\n{combined_text}\n\n")
                        log.write("-"*50 + "\n\n")
                    
                    # Log summary
                    log.write(f"EXTRACTION SUMMARY:\n")
                    log.write(f"Total pages processed: {len(pdf.pages)}\n")
                    log.write(f"Total documents created: {len(documents)}\n")
                
                print(f"Extraction complete. Log saved to {log_file}")
                print(f"Created {len(documents)} document chunks")
                return documents
            
        except ImportError as e:
            print(f"Warning: pdfplumber not available ({e}). Falling back to PyMuPDF.")
            # Fall back to PyMuPDF if pdfplumber is not available
            documents = []
            text = self.extractor.extract_text(pdf_path)
            print(f"Extracted {len(text)} characters of text using PyMuPDF")
            
            # Create a single document
            metadata = {"source": pdf_path}
            documents.append(Document(page_content=text, metadata=metadata))
            return documents
        except Exception as e:
            print(f"Error in PDF extraction: {e}")
            raise

    def answer_question(self, question: str, add_metadata: bool = True) -> Dict[str, Any]:
        """Answer a question using the RAG pipeline with detailed logging"""
        print("\n" + "="*50)
        print(f"PROCESSING QUESTION: {question}")
        print("="*50)
        
        if not self.vectorstore:
            print("No vector store available - no documents have been processed")
            return {
                "answer": "No documents have been processed yet.",
                "sources": []
            }
            
        try:
            # Create log file for this question
            log_dir = "qa_logs"
            os.makedirs(log_dir, exist_ok=True)
            log_id = hashlib.md5(question.encode()).hexdigest()[:8]
            log_file = os.path.join(log_dir, f"qa_{log_id}.txt")
            
            with open(log_file, "w", encoding="utf-8") as log:
                log.write(f"QUESTION: {question}\n")
                log.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                log.write("="*50 + "\n\n")
                
                # First retrieve relevant documents
                print("Retrieving relevant context...")
                docs = self.vectorstore.similarity_search(
                    question, 
                    k=5  # Get 5 most relevant chunks
                )
                
                # Log retrieved documents
                log.write("RETRIEVED CONTEXT:\n")
                for i, doc in enumerate(docs):
                    doc_id = doc.metadata.get("document_id", "Unknown")
                    page = doc.metadata.get("page", "Unknown")
                    source = doc.metadata.get("source", "Unknown")
                    
                    print(f"Context {i+1}: From document {doc_id}, page {page}")
                    log.write(f"CONTEXT {i+1}:\n")
                    log.write(f"Source: {source}\n")
                    log.write(f"Document ID: {doc_id}\n")
                    log.write(f"Page: {page}\n")
                    log.write("-"*30 + "\n")
                    log.write(doc.page_content)
                    log.write("\n\n" + "-"*50 + "\n\n")
                    
                # Create context for LLM
                context = "\n\n".join([doc.page_content for doc in docs])
                context_length = len(context)
                print(f"Retrieved {len(docs)} context chunks, total length: {context_length} characters")
                
                # Create LLM with better configuration
                print("Creating LLM...")
                llm = self.get_llm()
                
                # Use a better retrieval strategy
                print("Setting up retriever and QA chain...")
                retriever = self.vectorstore.as_retriever(
                    search_type="mmr",  # Maximum Marginal Relevance
                    search_kwargs={"k": 3, "fetch_k": 10}  # Adjust based on performance
                )
                
                # Create custom prompt template with better instructions
                from langchain.prompts import PromptTemplate
                
                template = """
                You are a helpful research assistant. Use the following pieces of context from academic papers to provide a detailed, accurate answer to the question.
                
                Guidelines:
                - Answer based only on the context provided, don't make up information
                - If the context doesn't contain enough information, acknowledge limitations in your answer
                - Include relevant details, facts, and figures from the context
                - Structure your answer with clear paragraphs for readability
                - If there are conflicting viewpoints in the context, present both sides

                Context:
                {context}

                Question: {question}

                Answer:
                """
                
                log.write("PROMPT TEMPLATE:\n")
                log.write(template)
                log.write("\n\n" + "-"*50 + "\n\n")
                
                # Log actual prompt with context (truncated if too long)
                actual_prompt = template.replace("{context}", context[:1000] + "..." if len(context) > 1000 else context)
                actual_prompt = actual_prompt.replace("{question}", question)
                log.write("ACTUAL PROMPT (truncated if too long):\n")
                log.write(actual_prompt)
                log.write("\n\n" + "-"*50 + "\n\n")
                
                # Use compact QA chain for better performance
                print("Generating answer...")
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",  # Simple document concatenation
                    retriever=retriever,
                    chain_type_kwargs={
                        "prompt": PromptTemplate(
                            template=template,
                            input_variables=["context", "question"]
                        )
                    },
                    return_source_documents=True
                )
                
                # Get answer with source documents
                result = qa_chain.invoke({"query": question})
                
                # Format answer with sources
                answer = result["result"]
                source_docs = result.get("source_documents", [])
                
                # Log the answer
                print("Answer generated successfully")
                log.write("GENERATED ANSWER:\n")
                log.write(answer)
                log.write("\n\n" + "-"*50 + "\n\n")
                
                # Process sources
                sources = []
                log.write("SOURCES:\n")
                
                for doc in source_docs:
                    doc_id = doc.metadata.get("document_id")
                    page = doc.metadata.get("page", "Unknown page")
                    if doc_id and doc_id in self.documents:
                        doc_info = self.documents[doc_id].copy()
                        doc_info["page"] = page
                        if doc_info not in sources:
                            sources.append(doc_info)
                            log.write(f"- Document: {doc_info.get('title', 'Unknown')}, Page: {page}\n")
                
                print(f"QA process complete. Log saved to {log_file}")
                return {
                    "answer": answer,
                    "sources": sources
                }
                
        except Exception as e:
            print(f"Error answering question: {e}")
            # More graceful fallback with text-based extraction
            try:
                print("Using fallback extraction method...")
                docs = self.vectorstore.similarity_search(question, k=3)
                context = "\n\n".join([f"From {doc.metadata.get('source', 'document')}, page {doc.metadata.get('page', 'unknown')}:\n{doc.page_content[:500]}..." for doc in docs])
                
                return {
                    "answer": f"I encountered an issue with the language model, but I found these relevant passages:\n\n{context}",
                    "sources": []
                }
            except Exception as fallback_error:
                print(f"Fallback also failed: {fallback_error}")
                return {
                    "answer": f"An error occurred while answering your question: {str(e)}",
                    "sources": []
                }
        
    def save_knowledge_base(self, output_dir: str) -> None:
        """Save the knowledge base to disk"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save document info
        import json
        with open(os.path.join(output_dir, "documents.json"), "w") as f:
            json.dump(self.documents, f, indent=2)
            
        # Save vectorstore
        if self.vectorstore:
            self.vectorstore.save_local(os.path.join(output_dir, "faiss_index"))
    
    def load_knowledge_base(self, input_dir: str) -> None:
        """Load a previously saved knowledge base"""
        # Load document info
        import json
        try:
            with open(os.path.join(input_dir, "documents.json"), "r") as f:
                self.documents = json.load(f)
                
            # Load vectorstore
            self.vectorstore = FAISS.load_local(
                os.path.join(input_dir, "faiss_index"), 
                self.embeddings
            )
        except Exception as e:
            print(f"Error loading knowledge base: {e}")
            raise

    def get_llm(self, model_name=None, use_fallback=False):
        """Get a language model with fallbacks"""
        if model_name is None:
            model_name = self.llm_model
            
        try:
            # First try: Ollama with specified model
            if not use_fallback:
                return OllamaLLM(
                    model=model_name,
                    temperature=0.1
                )
            
            # Fallback 1: Try a different local model
            alternative_models = ["llama2", "mistral", "gemma"]
            for alt_model in alternative_models:
                if alt_model != model_name:
                    try:
                        return Ollama(
                            model=alt_model,
                            temperature=0.1,
                            timeout=30
                        )
                    except:
                        continue
            
            # Fallback 2: Try OpenAI if API key is available
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if openai_api_key:
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    temperature=0.1,
                    model_name="gpt-3.5-turbo"
                )
                
            # Fallback 3: Use rule-based summarization
            raise ValueError("No available LLM found")
            
        except Exception as e:
            print(f"Error initializing LLM: {e}")
            raise

    # Create or update self._split_text method with improved chunking:
    def _split_text(self, text: str, chunk_type="semantic") -> List[str]:
        """Split text into chunks using different strategies"""
        if chunk_type == "semantic":
            # Try to create semantically meaningful chunks
            import re
            
            # Try to split on section headers first
            section_pattern = r'\n\s*(?:[0-9]+\.)+\s+([A-Z][A-Za-z\s]+)\s*\n|^\s*([A-Z][A-Z\s]+)\s*$'
            sections = re.split(section_pattern, text, flags=re.MULTILINE)
            
            if len(sections) > 1:  # If we found sections
                chunks = []
                for i in range(0, len(sections), 3):  # Group by 3 because re.split includes the matched groups
                    section_text = sections[i]
                    if i+1 < len(sections) and sections[i+1]:  # If we have a section title
                        title = sections[i+1]
                    elif i+2 < len(sections) and sections[i+2]:
                        title = sections[i+2]
                    else:
                        title = ""
                    
                    if len(section_text.split()) > 20:  # Only include substantial sections
                        if title:
                            chunks.append(f"SECTION: {title}\n\n{section_text}")
                        else:
                            chunks.append(section_text)
                return chunks
            
            # If not enough section headers, try to split by paragraphs
            paragraphs = re.split(r'\n\s*\n', text)
            chunks = []
            current_chunk = ""
            
            for para in paragraphs:
                if len(current_chunk) + len(para) < self.chunk_size:
                    current_chunk += para + "\n\n"
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = para + "\n\n"
                    
            if current_chunk:
                chunks.append(current_chunk)
                
            return chunks
        else:
            # Fall back to simple word-based chunking
            words = text.split()
            return [" ".join(words[i:i+self.chunk_size]) for i in range(0, len(words), self.chunk_size-self.chunk_overlap)]

    def hybrid_answer(self, question: str) -> Dict[str, Any]:
        """Generate answer using multiple approaches and combine results"""
        try:
            # Try primary approach
            primary_response = self.answer_question(question)
            
            # Check if answer seems incomplete or contains error indicators
            low_quality_indicators = [
                "I don't know", 
                "I don't have enough information",
                "error",
                "not available",
                "not specified in the context"
            ]
            
            is_low_quality = any(indicator in primary_response.get("answer", "").lower() for indicator in low_quality_indicators)
            
            if not is_low_quality and len(primary_response.get("answer", "").split()) > 30:
                # Good primary response
                return primary_response
            
            # Try with a different model
            try:
                secondary_llm = self.get_llm(use_fallback=True)
                # Use the chunks we already retrieved
                docs = self.vectorstore.similarity_search(question, k=3)
                context = "\n\n".join([doc.page_content for doc in docs])
                
                prompt = f"""
                Question: {question}
                
                Context information:
                {context}
                
                Answer the question based on the context information. If the context doesn't contain relevant information, acknowledge that the information is not available.
                """
                
                from langchain.chains import LLMChain
                from langchain.prompts import PromptTemplate
                
                chain = LLMChain(
                    llm=secondary_llm,
                    prompt=PromptTemplate.from_template("{text}"),
                )
                
                secondary_answer = chain.invoke({"text": prompt})["text"]
                
                # Combine answers if both are available
                if primary_response.get("answer") and len(primary_response["answer"]) > 20:
                    combined = f"{primary_response['answer']}\n\nAdditional information:\n{secondary_answer}"
                    primary_response["answer"] = combined
                    return primary_response
                else:
                    return {"answer": secondary_answer, "sources": primary_response.get("sources", [])}
            
            except Exception as e:
                print(f"Error in secondary answering approach: {e}")
                return primary_response
                
        except Exception as e:
            print(f"Error in hybrid_answer: {e}")
            # Last resort fallback
            try:
                # Extract keywords from question
                import re
                keywords = re.findall(r'\b[a-zA-Z]{3,}\b', question.lower())
                keywords = [k for k in keywords if k not in ('the', 'and', 'that', 'what', 'when', 'where', 'who', 'why', 'how')]
                
                # Try simple keyword search
                if self.vectorstore and keywords:
                    from tqdm import tqdm
                    results = []
                    for keyword in tqdm(keywords[:3]):  # Limit to 3 keywords
                        docs = self.vectorstore.similarity_search(keyword, k=1)
                        for doc in docs:
                            results.append(f"Information related to '{keyword}':\n{doc.page_content[:300]}...\n")
                    
                    if results:
                        return {
                            "answer": f"I encountered difficulties answering your question directly, but here is related information I found:\n\n{''.join(results)}",
                            "sources": []
                        }
                
                return {
                    "answer": "I'm unable to answer this question based on the available information.",
                    "sources": []
                }
            except:
                return {
                    "answer": "An error occurred while processing your question.",
                    "sources": []
                }
            
    def debug_show_document(self, document_id: str) -> None:
        """Display stored document data for debugging"""
        if document_id not in self.documents:
            print(f"Document ID {document_id} not found in store")
            return
        
        doc_info = self.documents[document_id]
        
        print("\n" + "="*50)
        print(f"DOCUMENT DEBUG: {doc_info.get('title', 'Untitled')}")
        print("="*50)
        print(f"Document ID: {document_id}")
        print(f"Path: {doc_info.get('path', 'Unknown')}")
        
        # Print all metadata
        print("\nMETADATA:")
        for key, value in doc_info.items():
            if key not in ['id', 'path']:
                print(f"  {key}: {value}")
        
        # Check for content files
        content_file = os.path.join("extraction_logs", f"full_content_{document_id}.txt")
        if os.path.exists(content_file):
            print(f"\nFull content available at: {content_file}")
        
        # Check vector representation
        if self.vectorstore:
            try:
                # Search for chunks from this document
                from langchain.schema.document import Document
                
                # Construct a dummy query using the document title
                query = f"about {doc_info.get('title', '')}"
                results = self.vectorstore.similarity_search(
                    query,
                    filter={"document_id": document_id},
                    k=5
                )
                
                print(f"\nDocument has approximately {doc_info.get('chunks_count', 'unknown')} chunks in the vector store")
                print(f"Sample chunks by searching for: '{query}'")
                
                for i, doc in enumerate(results[:3]):  # Show only first 3
                    print(f"\nCHUNK {i+1}:")
                    print(f"Page: {doc.metadata.get('page', 'Unknown')}")
                    print(f"Content (first 150 chars): {doc.page_content[:150]}...")
            
            except Exception as e:
                print(f"Error querying vector store: {e}")
        
        print("\n" + "="*50)