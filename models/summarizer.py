from typing import Dict, List, Any, Optional
from .extractor import PDFExtractor
from .llm import LLMInterface
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import numpy as np
import networkx as nx

# Download necessary NLTK resources (add to project setup)
# nltk.download('punkt')
# nltk.download('stopwords')

class PaperSummarizer:
    """Advanced research paper summarization with hybrid techniques"""
    
    def __init__(self, llm_backend="ollama", llm_model="llama3"):
        self.extractor = PDFExtractor()
        self.llm = LLMInterface(default_model=llm_model, backend=llm_backend)
        
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