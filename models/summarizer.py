from typing import Dict, List, Any, Optional
from .extractor import PDFExtractor
from .llm import LLMInterface
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
import torch
import re

# Download necessary NLTK resources (add to project setup)
# nltk.download('punkt')
# nltk.download('stopwords')

class PaperSummarizer:
    """Advanced research paper summarization with hybrid techniques"""
    
    def __init__(self, llm_backend="ollama", llm_model="llama3"):
        self.extractor = PDFExtractor()
        self.llm = LLMInterface(default_model=llm_model, backend=llm_backend)
        
        # Download required NLTK data
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        except Exception as e:
            print(f"Warning: Failed to download NLTK data: {e}")
            
        # Initialize the abstractive summarization model
        self.abstractive_model = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=0 if torch.cuda.is_available() else -1
        )
        
    def hybrid_summarize(self, pdf_path: str, target_length: int = 500) -> Dict[str, str]:
        """Create hybrid summarization using extractive + abstractive techniques"""
        # Extract sections
        sections = self.extractor.extract_sections(pdf_path)
        
        results = {}
        
        # Process key sections
        for section_name in ["abstract", "introduction", "conclusion"]:
            if section_name in sections:
                # First do extractive summarization to get key sentences
                extractive_summary = self._extractive_summarize(
                    sections[section_name], 
                    int(target_length * 1.5)  # Extract more than we need
                )
                
                # Then use LLM for abstractive summarization
                prompt = f"""Below is an extracted section from a research paper. 
                Please create a concise summary of this {section_name} section in about {target_length} words.
                Keep the key points and technical details accurate.
                
                {extractive_summary}"""
                
                abstractive_summary = self.llm.generate(prompt)
                results[section_name] = abstractive_summary
        
        # Create overall summary if we have enough sections
        if len(results) >= 2:
            combined_sections = " ".join(results.values())
            
            prompt = f"""Create a comprehensive but concise summary of this research paper based on these key sections.
            Focus on the main contributions, methods, and findings. Keep it under {target_length * 2} words.
            
            {combined_sections}"""
            
            results["overall"] = self.llm.generate(prompt)
        
        return results
    
    def _extractive_summarize(self, text: str, target_length: int) -> str:
        """Extract key sentences using TextRank algorithm"""
        # Tokenize text into sentences
        sentences = sent_tokenize(text)
        
        # If text is already short, just return it
        if len(text) <= target_length:
            return text
            
        # Handle case with very few sentences
        if len(sentences) <= 3:
            return " ".join(sentences)
        
        # Create sentence similarity matrix
        similarity_matrix = self._build_similarity_matrix(sentences)
        
        # Apply TextRank algorithm
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph)
        
        # Sort sentences by score
        ranked_sentences = sorted([(scores[i], s) for i, s in enumerate(sentences)], reverse=True)
        
        # Select top sentences up to target length
        selected_sentences = []
        current_length = 0
        
        for _, sentence in ranked_sentences:
            if current_length + len(sentence) <= target_length:
                selected_sentences.append(sentence)
                current_length += len(sentence)
            else:
                break
                
        # Re-order sentences to maintain original flow
        selected_indices = [i for i, (_, s) in enumerate(sentences) if s in selected_sentences]
        ordered_summary = [sentences[i] for i in sorted(selected_indices)]
        
        return " ".join(ordered_summary)
    
    def _build_similarity_matrix(self, sentences: List[str]) -> np.ndarray:
        """Build similarity matrix for sentences using word overlap"""
        # Initialize similarity matrix
        n = len(sentences)
        similarity_matrix = np.zeros((n, n))
        
        # Get English stopwords
        stop_words = set(stopwords.words('english'))
        
        # Calculate similarity between each pair of sentences
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                    
                # Simple word overlap similarity
                words_i = set(w.lower() for w in sentences[i].split() if w.lower() not in stop_words)
                words_j = set(w.lower() for w in sentences[j].split() if w.lower() not in stop_words)
                
                if not words_i or not words_j:
                    continue
                    
                # Jaccard similarity
                similarity = len(words_i.intersection(words_j)) / len(words_i.union(words_j))
                similarity_matrix[i][j] = similarity
                
        return similarity_matrix
        
    def extract_key_points(self, pdf_path: str, num_points: int = 5) -> List[str]:
        """Extract key points or findings from the paper"""
        # Extract sections
        sections = self.extractor.extract_sections(pdf_path)
        
        # Combine relevant sections
        text = ""
        for section in ["abstract", "introduction", "conclusion", "discussion", "results"]:
            if section in sections:
                text += sections[section] + "\n\n"
        
        # If we have substantial text, use LLM to extract key points
        if len(text) > 100:
            prompt = f"""
            Extract exactly {num_points} key findings or contributions from this research paper text.
            Format each point as a single clear sentence that captures a specific finding.
            Return only the numbered list, nothing else.
            
            Text:
            {text[:5000]}  # Limit text length for LLM
            """
            
            response = self.llm.generate(prompt)
            
            # Parse the response to extract points
            points = []
            for line in response.strip().split("\n"):
                line = line.strip()
                # Look for numbered points like "1.", "2.", etc.
                if line and (line[0].isdigit() or (len(line) > 2 and line[0:2] in ["1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9."])):
                    # Remove the number and any leading symbols
                    point = re.sub(r"^\d+[\.\)\s]+\s*", "", line).strip()
                    if point:
                        points.append(point)
            
            # If parsing fails, just split by lines
            if not points:
                points = [line.strip() for line in response.strip().split("\n") if line.strip()]
                
            return points[:num_points]  # Ensure we return the requested number
            
        return ["No key points extracted - insufficient text"]

    def extractive_summarize(self, doc_path: str) -> Dict[str, Any]:
        """
        Generate an extractive summary by selecting the most important sentences
        based on TF-IDF scores.
        """
        try:
            # Extract text from PDF
            text = self.extractor.extract_text(doc_path)
            
            # Split into sentences
            sentences = sent_tokenize(text)
            
            # Calculate TF-IDF scores
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(sentences)
            
            # Calculate sentence scores
            sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
            
            # Select top sentences (approximately 30% of the original)
            num_sentences = max(3, int(len(sentences) * 0.3))
            top_indices = sentence_scores.argsort()[-num_sentences:][::-1]
            top_indices.sort()  # Sort to maintain original order
            
            # Generate summary
            summary_sentences = [sentences[i] for i in top_indices]
            summary = ' '.join(summary_sentences)
            
            # Extract key points
            key_points = self._extract_key_points(text)
            
            return {
                'summary': summary,
                'key_points': key_points,
                'method': 'extractive'
            }
            
        except Exception as e:
            print(f"Error in extractive summarization: {e}")
            # Return a simple fallback summary
            return {
                'summary': f"Error generating detailed summary. Here's a brief overview:\n\n{text[:500]}...",
                'key_points': ["Unable to extract key points due to processing error"],
                'method': 'fallback'
            }
            
    def abstractive_summarize(self, doc_path: str) -> Dict[str, Any]:
        """
        Generate an abstractive summary using the BART model.
        """
        try:
            # Extract text from PDF
            text = self.extractor.extract_text(doc_path)
            
            # Split text into chunks (BART has a maximum input length)
            chunks = self._split_into_chunks(text, max_length=1024)
            
            # Generate summary for each chunk
            chunk_summaries = []
            for chunk in chunks:
                summary = self.abstractive_model(chunk, 
                                              max_length=150, 
                                              min_length=50, 
                                              do_sample=False)
                chunk_summaries.append(summary[0]['summary_text'])
            
            # Combine chunk summaries
            combined_summary = ' '.join(chunk_summaries)
            
            # Extract key points
            key_points = self._extract_key_points(text)
            
            return {
                'summary': combined_summary,
                'key_points': key_points,
                'method': 'abstractive'
            }
            
        except Exception as e:
            print(f"Error in abstractive summarization: {e}")
            raise
            
    def hybrid_summarize(self, doc_path: str) -> Dict[str, Any]:
        """
        Generate a hybrid summary by combining extractive and abstractive approaches.
        """
        try:
            # First generate extractive summary
            extractive_result = self.extractive_summarize(doc_path)
            extractive_summary = extractive_result['summary']
            
            # Then generate abstractive summary of the extractive summary
            abstractive_summary = self.abstractive_model(extractive_summary,
                                                      max_length=200,
                                                      min_length=100,
                                                      do_sample=False)
            
            return {
                'summary': abstractive_summary[0]['summary_text'],
                'key_points': extractive_result['key_points'],
                'method': 'hybrid'
            }
            
        except Exception as e:
            print(f"Error in hybrid summarization: {e}")
            raise
            
    def _extract_key_points(self, text: str) -> List[str]:
        """
        Extract key points from the text using a combination of techniques.
        """
        try:
            # Split into sentences
            sentences = sent_tokenize(text)
            
            # Look for sentences that might contain key points
            key_point_patterns = [
                r'key\s+point[s]?\s*:',
                r'important\s+to\s+note',
                r'significant\s+finding[s]?',
                r'main\s+conclusion[s]?',
                r'primary\s+result[s]?'
            ]
            
            key_points = []
            for sentence in sentences:
                if any(re.search(pattern, sentence.lower()) for pattern in key_point_patterns):
                    key_points.append(sentence.strip())
            
            # If no key points found, use TF-IDF to find important sentences
            if not key_points:
                vectorizer = TfidfVectorizer(stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(sentences)
                sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
                
                # Get top 5 sentences
                top_indices = sentence_scores.argsort()[-5:][::-1]
                key_points = [sentences[i].strip() for i in top_indices]
            
            return key_points
            
        except Exception as e:
            print(f"Error extracting key points: {e}")
            return []
            
    def _split_into_chunks(self, text: str, max_length: int = 1024) -> List[str]:
        """
        Split text into chunks of maximum length while trying to keep sentences intact.
        """
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            if current_length + sentence_length > max_length:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks

    def smart_summarize(self, pdf_path: str) -> Dict[str, Any]:
        """
        Automatically choose and apply the best summarization method based on document characteristics.
        """
        try:
            # Extract text and analyze document characteristics
            text = self.extractor.extract_text(pdf_path)
            sections = self.extractor.extract_sections(pdf_path)
            
            # Analyze document characteristics
            doc_length = len(text)
            has_abstract = 'abstract' in sections
            has_conclusion = 'conclusion' in sections
            num_sections = len(sections)
            
            # Decision logic for choosing summarization method
            if doc_length < 2000:  # Very short document
                # For very short documents, use extractive summarization
                return self.extractive_summarize(pdf_path)
            elif has_abstract and has_conclusion and num_sections >= 3:
                # For well-structured documents with abstract and conclusion, use hybrid
                return self.hybrid_summarize(pdf_path)
            elif doc_length > 10000:  # Very long document
                # For very long documents, use extractive first to reduce length
                extractive_result = self.extractive_summarize(pdf_path)
                # Then use abstractive on the extractive summary
                abstractive_summary = self.abstractive_model(
                    extractive_result['summary'],
                    max_length=200,
                    min_length=100,
                    do_sample=False
                )
                return {
                    'summary': abstractive_summary[0]['summary_text'],
                    'key_points': extractive_result['key_points'],
                    'method': 'smart_hybrid'
                }
            else:
                # Default to abstractive for other cases
                return self.abstractive_summarize(pdf_path)
                
        except Exception as e:
            print(f"Error in smart summarization: {e}")
            # Instead of falling back to extractive, return a simple summary
            return {
                'summary': f"Error generating detailed summary. Here's a brief overview:\n\n{text[:500]}...",
                'key_points': ["Unable to extract key points due to processing error"],
                'method': 'fallback'
            }