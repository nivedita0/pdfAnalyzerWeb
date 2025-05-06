from typing import Dict, Any, Optional

class RuleBasedQA:
    """Provides rule-based answers for common questions when LLM fails"""
    
    def __init__(self):
        self.document_info = {}
    
    def register_document(self, doc_id: str, metadata: Dict[str, Any]):
        """Register a document's metadata for rule-based answering"""
        self.document_info[doc_id] = metadata
    
    def answer_question(self, question: str, doc_id: Optional[str] = None) -> Dict[str, Any]:
        """Answer questions based on simple rules"""
        question = question.lower().strip()
        
        # Filter to specific document if requested
        docs = {}
        if doc_id and doc_id in self.document_info:
            docs = {doc_id: self.document_info[doc_id]}
        else:
            docs = self.document_info
        
        # Empty response template
        response = {
            "answer": "I don't have enough information to answer this question.",
            "sources": []
        }
        
        # No documents available
        if not docs:
            response["answer"] = "No documents have been loaded yet. Please upload a document first."
            return response
        
        # Author questions
        if any(term in question for term in ['who wrote', 'author', 'written by', 'authors of', 'creator']):
            authors_info = []
            
            for d_id, info in docs.items():
                title = info.get('title', 'Untitled document')
                authors = info.get('authors', [])
                
                if authors:
                    authors_str = ', '.join(authors)
                    authors_info.append(f"The document '{title}' was written by {authors_str}.")
                else:
                    authors_info.append(f"The document '{title}' doesn't have author information available.")
            
            if authors_info:
                response["answer"] = '\n'.join(authors_info)
                response["sources"] = list(docs.values())
                return response
        
        # Year/date questions
        if any(term in question for term in ['when', 'year', 'date', 'published', 'publication']):
            year_info = []
            
            for d_id, info in docs.items():
                title = info.get('title', 'Untitled document')
                year = info.get('year', '')
                
                if year:
                    year_info.append(f"The document '{title}' was published in {year}.")
                else:
                    year_info.append(f"The publication year for '{title}' is not available.")
            
            if year_info:
                response["answer"] = '\n'.join(year_info)
                response["sources"] = list(docs.values())
                return response
        
        # Title questions
        if any(term in question for term in ['title', 'name', 'called', 'what is this']):
            titles = [f"'{info.get('title', 'Untitled document')}'" for info in docs.values()]
            
            if len(titles) == 1:
                response["answer"] = f"The document title is {titles[0]}."
            else:
                response["answer"] = f"The available documents are: {', '.join(titles)}."
            
            response["sources"] = list(docs.values())
            return response
        
        # DOI questions
        if any(term in question for term in ['doi', 'identifier']):
            doi_info = []
            
            for d_id, info in docs.items():
                title = info.get('title', 'Untitled document')
                doi = info.get('doi', '')
                
                if doi:
                    doi_info.append(f"The DOI for '{title}' is {doi}.")
                else:
                    doi_info.append(f"No DOI is available for '{title}'.")
            
            if doi_info:
                response["answer"] = '\n'.join(doi_info)
                response["sources"] = list(docs.values())
                return response
        
        return response