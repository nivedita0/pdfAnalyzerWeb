import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
import io
import re
from typing import List, Dict, Any
import pdfplumber
import os

class DocumentAnalyzer:
    def __init__(self):
        self.figure_patterns = [
            r'Figure\s+\d+[.:]',
            r'Fig\.\s+\d+[.:]',
            r'Figure\s+\d+\s*[–-]\s*',
            r'Fig\.\s+\d+\s*[–-]\s*'
        ]
        
        self.reference_patterns = [
            r'References',
            r'Bibliography',
            r'Works Cited',
            r'Literature Cited'
        ]
        
    def extract_figures(self, pdf_path: str, output_dir: str) -> List[Dict[str, Any]]:
        """
        Extract figures from PDF with their captions
        """
        figures = []
        doc = fitz.open(pdf_path)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Get images on the page
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Convert to PIL Image
                image = Image.open(io.BytesIO(image_bytes))
                
                # Save image
                image_filename = f"figure_{page_num + 1}_{img_index + 1}.png"
                image_path = os.path.join(output_dir, image_filename)
                image.save(image_path)
                
                # Try to find caption
                caption = self._find_figure_caption(page, img_index)
                
                figures.append({
                    'page': page_num + 1,
                    'path': image_path,
                    'caption': caption,
                    'url': f"/extracted/{os.path.basename(output_dir)}/{image_filename}"
                })
        
        return figures
    
    def extract_tables(self, pdf_path: str, output_dir: str) -> List[Dict[str, Any]]:
        """
        Extract tables from PDF with their captions
        """
        tables = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Extract tables from the page
                page_tables = page.extract_tables()
                
                for table_index, table in enumerate(page_tables):
                    if not table:  # Skip empty tables
                        continue
                        
                    # Convert table to HTML
                    html_table = self._table_to_html(table)
                    
                    # Try to find caption
                    caption = self._find_table_caption(page, table_index)
                    
                    tables.append({
                        'page': page_num + 1,
                        'html': html_table,
                        'caption': caption
                    })
        
        return tables
    
    def extract_citations(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract citations from PDF, focusing on the references section
        """
        citations = []
        doc = fitz.open(pdf_path)
        
        # Find the references section
        ref_section_start = -1
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            # Check for reference section headers
            for pattern in self.reference_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    ref_section_start = page_num
                    break
            
            if ref_section_start != -1:
                break
        
        if ref_section_start == -1:
            # If no explicit references section found, use last 20% of pages
            ref_section_start = int(len(doc) * 0.8)
        
        # Extract citations from the references section
        for page_num in range(ref_section_start, len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            # Split text into lines for processing
            lines = text.split('\n')
            
            current_citation = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check if line starts with a number or common citation patterns
                if re.match(r'^\d+\.|^\[\d+\]|^[A-Z][a-z]+ et al\.|^[A-Z][a-z]+, [A-Z]\.', line):
                    # Process previous citation if exists
                    if current_citation:
                        citation = self._parse_reference(' '.join(current_citation))
                        if citation:
                            citations.append(citation)
                        current_citation = []
                    
                    current_citation.append(line)
                elif current_citation:
                    current_citation.append(line)
            
            # Process the last citation in the page
            if current_citation:
                citation = self._parse_reference(' '.join(current_citation))
                if citation:
                    citations.append(citation)
        
        return citations
    
    def _parse_reference(self, ref_text: str) -> Dict[str, Any]:
        """
        Parse a reference text into structured data
        """
        # Common patterns for references
        patterns = [
            # Author et al. (Year) Title. Journal, Volume(Issue), Pages.
            r'([A-Z][a-z]+(?:, [A-Z][a-z]+)* et al\.) \((\d{4})\) ([^\.]+)\. ([^\.]+), (\d+)(?:\((\d+)\))?, ([^\.]+)\.',
            
            # Author, A., & Author, B. (Year) Title. Journal, Volume(Issue), Pages.
            r'([A-Z][a-z]+, [A-Z]\.(?:, & [A-Z][a-z]+, [A-Z]\.)*) \((\d{4})\) ([^\.]+)\. ([^\.]+), (\d+)(?:\((\d+)\))?, ([^\.]+)\.',
            
            # Author, A. (Year) Title. Journal, Volume, Pages.
            r'([A-Z][a-z]+, [A-Z]\.) \((\d{4})\) ([^\.]+)\. ([^\.]+), (\d+), ([^\.]+)\.'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, ref_text)
            if match:
                groups = match.groups()
                authors = groups[0].split(', ')
                if 'et al.' in authors[-1]:
                    authors = authors[:-1]  # Remove 'et al.'
                
                return {
                    'authors': authors,
                    'year': groups[1],
                    'title': groups[2].strip(),
                    'journal': groups[3].strip(),
                    'volume': groups[4],
                    'issue': groups[5] if len(groups) > 5 and groups[5] else None,
                    'pages': groups[-1].strip()
                }
        
        return None
    
    def _find_figure_caption(self, page, figure_index: int) -> str:
        """
        Find caption for a figure on the page
        """
        text = page.get_text()
        lines = text.split('\n')
        
        # Look for caption patterns near the figure
        for i, line in enumerate(lines):
            if any(re.search(pattern, line) for pattern in self.figure_patterns):
                # Get the next few lines as potential caption
                caption_lines = []
                for j in range(i, min(i + 3, len(lines))):
                    caption_lines.append(lines[j])
                return ' '.join(caption_lines)
        
        return f"Figure {figure_index + 1}"
    
    def _find_table_caption(self, page, table_index: int) -> str:
        """
        Find caption for a table on the page
        """
        text = page.extract_text()
        lines = text.split('\n')
        
        # Look for table caption patterns
        table_patterns = [
            r'Table\s+\d+[.:]',
            r'Table\s+\d+\s*[–-]\s*'
        ]
        
        for i, line in enumerate(lines):
            if any(re.search(pattern, line) for pattern in table_patterns):
                # Get the next few lines as potential caption
                caption_lines = []
                for j in range(i, min(i + 3, len(lines))):
                    caption_lines.append(lines[j])
                return ' '.join(caption_lines)
        
        return f"Table {table_index + 1}"
    
    def _table_to_html(self, table: List[List[str]]) -> str:
        """
        Convert table data to HTML
        """
        html = ['<table class="table table-bordered">']
        
        # Add header row
        if table and table[0]:
            html.append('<thead><tr>')
            for cell in table[0]:
                html.append(f'<th>{cell}</th>')
            html.append('</tr></thead>')
        
        # Add body rows
        html.append('<tbody>')
        for row in table[1:]:
            html.append('<tr>')
            for cell in row:
                html.append(f'<td>{cell}</td>')
            html.append('</tr>')
        html.append('</tbody>')
        
        html.append('</table>')
        return '\n'.join(html) 