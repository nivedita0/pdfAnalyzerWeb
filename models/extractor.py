import fitz  # PyMuPDF
import os
from typing import List, Dict, Any
import re

class PDFExtractor:
    """Enhanced PDF extraction with advanced features"""
    
    def __init__(self, use_ocr=False):
        self.use_ocr = use_ocr
        if use_ocr:
            try:
                # Import PaddleOCR conditionally
                from paddleocr import PaddleOCR
                self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
            except ImportError:
                print("Warning: PaddleOCR not installed. OCR capabilities disabled.")
                self.use_ocr = False
    
    def extract_text(self, pdf_path: str) -> str:
        """Extract text from PDF using PyMuPDF with OCR fallback"""
        text = ""
        with fitz.open(pdf_path) as doc:
            for page in doc:
                page_text = page.get_text()
                # If page has very little text and OCR is enabled, try OCR
                if len(page_text.strip()) < 50 and self.use_ocr:
                    pix = page.get_pixmap()
                    img_path = f"temp_page_{page.number}.png"
                    pix.save(img_path)
                    try:
                        ocr_result = self.ocr.ocr(img_path)
                        page_text = "\n".join([line[1][0] for line in ocr_result[0]])
                    finally:
                        # Clean up temporary image
                        if os.path.exists(img_path):
                            os.remove(img_path)
                text += page_text + "\n"
        return text

    def extract_sections(self, pdf_path: str) -> Dict[str, str]:
        """Extract sections from a research paper"""
        text = self.extract_text(pdf_path)
        
        # Common section headers in research papers
        section_headers = [
            "abstract", "introduction", "related work", "background",
            "methodology", "methods", "experiments", "results", 
            "discussion", "conclusion", "references"
        ]
        
        sections = {}
        lines = text.split('\n')
        current_section = "preamble"
        sections[current_section] = []
        
        for line in lines:
            line_lower = line.lower().strip()
            # Check if line is a section header
            is_header = False
            for header in section_headers:
                if line_lower == header or line_lower.startswith(f"{header}."):
                    current_section = header
                    sections[current_section] = []
                    is_header = True
                    break
            
            if not is_header:
                sections[current_section].append(line)
        
        # Join the lines in each section
        for section in sections:
            sections[section] = '\n'.join(sections[section])
            
        return sections

    def extract_tables(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract tables using heuristics or OCR capabilities"""
        # This would be enhanced with PaddleOCR table detection
        tables = []
        # Basic implementation - will be expanded
        return tables

    def extract_metadata(self, pdf_path: str) -> dict:
        """Extract metadata from PDF document"""
        metadata = {
            "title": "",
            "authors": [],
            "year": "",
            "doi": "",
            "abstract": ""
        }
        
        try:
            # Open the PDF
            with fitz.open(pdf_path) as doc:
                # Try to get metadata from PDF document info
                info = doc.metadata
                if info:
                    if info.get('title'):
                        metadata["title"] = info.get('title')
                    if info.get('author'):
                        authors = info.get('author').split(',')
                        metadata["authors"] = [author.strip() for author in authors]
                    
                # If title not found in metadata, try to extract from first page
                if not metadata["title"]:
                    if doc.page_count > 0:
                        first_page_text = doc[0].get_text()
                        lines = first_page_text.strip().split('\n')
                        
                        # Skip lines that look like dates or page numbers
                        date_pattern = r'\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}'
                        page_pattern = r'^\d+$'
                        
                        for line in lines:
                            line = line.strip()
                            # Skip empty lines, dates, and page numbers
                            if (not line or 
                                re.search(date_pattern, line) or 
                                re.search(page_pattern, line) or
                                len(line) > 200):  # Skip very long lines
                                continue
                            
                            # Found a potential title
                            metadata["title"] = line
                            break
                
                # Try to extract publication year
                for page in doc:
                    text = page.get_text()
                    # Look for year patterns
                    year_patterns = [
                        r'copyright\s+©?\s*(\d{4})',
                        r'published\s+in\s+(\d{4})',
                        r'(\d{4})\s*\.\s*all\s+rights\s+reserved',
                        r'(?:19|20)\d{2}'  # General 4-digit year pattern
                    ]
                    
                    for pattern in year_patterns:
                        matches = re.findall(pattern, text.lower())
                        if matches:
                            for match in matches:
                                year = match
                                if 1900 <= int(year) <= 2100:  # Reasonable year range
                                    metadata["year"] = year
                                    break
                    
                    # Look for DOI
                    doi_pattern = r'doi:?\s*(10\.\d{4,}(?:\.\d+)*\/\S+)'
                    doi_matches = re.findall(doi_pattern, text.lower())
                    if doi_matches:
                        metadata["doi"] = doi_matches[0]
                    
                    # Extract abstract
                    abstract_pattern = r'abstract\s*\n(.*?)(?:\n\s*keywords|\n\s*introduction|\n\s*\d+\.)'
                    abstract_matches = re.search(abstract_pattern, text.lower(), re.DOTALL)
                    if abstract_matches:
                        metadata["abstract"] = abstract_matches.group(1).strip()
                    
                    # Stop after checking first few pages
                    if metadata["year"] and metadata["doi"] and metadata["abstract"]:
                        break
                
                # If still no title found, use filename as fallback
                if not metadata["title"]:
                    metadata["title"] = os.path.splitext(os.path.basename(pdf_path))[0]
        
        except Exception as e:
            print(f"Error extracting metadata: {e}")
            # Use filename as fallback if there's an error
            metadata["title"] = os.path.splitext(os.path.basename(pdf_path))[0]
        
        return metadata