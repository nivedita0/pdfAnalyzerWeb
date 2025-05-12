import os
import fitz  # PyMuPDF
import numpy as np
import cv2
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import re
import csv
import pdfplumber

class TableFigureExtractor:
    """Extract tables and figures from PDF documents"""
    
    def __init__(self, use_ocr=True):
        self.use_ocr = use_ocr
        if use_ocr:
            try:
                # Import PaddleOCR conditionally
                from paddleocr import PaddleOCR
                self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
            except ImportError:
                print("Warning: PaddleOCR not installed. OCR capabilities disabled.")
                self.use_ocr = False
                
    def extract_figures(self, pdf_path: str, output_dir: str = "extracted_figures") -> List[Dict[str, Any]]:
        """Extract figures from PDF"""
        os.makedirs(output_dir, exist_ok=True)
        
        figures = []
        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc):
                # Extract images
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    xref = img[0]  # image reference
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Save image to file
                    image_filename = f"page{page_num+1}_img{img_index+1}.{base_image['ext']}"
                    image_path = os.path.join(output_dir, image_filename)
                    
                    with open(image_path, "wb") as f:
                        f.write(image_bytes)
                    
                    # Try to get caption (text below image)
                    caption = self._extract_caption_near_image(page, img)
                    
                    figures.append({
                        "page": page_num + 1,
                        "path": image_path,
                        "caption": caption,
                        "type": "image"
                    })
                
                # Extract vector graphics (often diagrams)
                paths = page.get_drawings()
                if paths:
                    # Render page to image
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    
                    # Save full page as image
                    image_filename = f"page{page_num+1}_drawing.png"
                    image_path = os.path.join(output_dir, image_filename)
                    img.save(image_path)
                    
                    figures.append({
                        "page": page_num + 1,
                        "path": image_path,
                        "caption": "Vector drawing/diagram",
                        "type": "drawing"
                    })
        
        return figures
    
    def _detect_table_areas(self, page) -> List[fitz.Rect]:
        """Detect areas that likely contain tables using heuristics"""
        # This is a simplistic approach - more sophisticated methods would be needed for production
        
        # Strategy 1: Look for horizontal/vertical lines that form a grid
        table_areas = []
        
        # Get all the line drawings on the page
        drawings = page.get_drawings()
        
        # Group lines by their horizontal or vertical orientation
        horizontal_lines = []
        vertical_lines = []
        
        for drawing in drawings:
            for item in drawing["items"]:
                if item[0] == "l":  # Line element
                    # Get coordinates (x0, y0, x1, y1)
                    line = item[1]
                    
                    # Check if horizontal (y0 ≈ y1)
                    if abs(line[1] - line[3]) < 2:
                        horizontal_lines.append((min(line[0], line[2]), line[1], max(line[0], line[2]), line[3]))
                    
                    # Check if vertical (x0 ≈ x1)
                    if abs(line[0] - line[2]) < 2:
                        vertical_lines.append((line[0], min(line[1], line[3]), line[2], max(line[1], line[3])))
        
        # If we have enough lines, try to find intersections
        if len(horizontal_lines) > 2 and len(vertical_lines) > 2:
            # Find bounds of potential tables
            h_clusters = self._cluster_lines(horizontal_lines, axis=1)
            v_clusters = self._cluster_lines(vertical_lines, axis=0)
            
            # For each pair of horizontal clusters that are close to each other
            for i in range(len(h_clusters) - 1):
                top_cluster = h_clusters[i]
                
                for j in range(i + 1, len(h_clusters)):
                    bottom_cluster = h_clusters[j]
                    
                    # Check if they're not too far apart
                    distance = abs(top_cluster["position"] - bottom_cluster["position"])
                    if distance < 300:  # Arbitrary threshold
                        # Find vertical lines that span between these horizontals
                        spanning_verticals = []
                        
                        for v_cluster in v_clusters:
                            v_pos = v_cluster["position"]
                            # Check if this vertical line spans between the horizontal clusters
                            for v_line in v_cluster["lines"]:
                                if v_line[1] <= top_cluster["position"] and v_line[3] >= bottom_cluster["position"]:
                                    spanning_verticals.append(v_pos)
                                    break
                        
                        # If we have at least 2 vertical lines spanning between horizontals, it's likely a table
                        if len(spanning_verticals) >= 2:
                            # Create a rectangle encompassing the table
                            x0 = min(spanning_verticals)
                            x1 = max(spanning_verticals)
                            y0 = top_cluster["position"]
                            y1 = bottom_cluster["position"]
                            
                            # Add some padding
                            table_areas.append(fitz.Rect(x0 - 5, y0 - 5, x1 + 5, y1 + 5))
        
        # If no tables found with line detection, try alternate method
        if not table_areas:
            # Look for text blocks that might indicate tables
            blocks = page.get_text("blocks")
            
            for block in blocks:
                # Check if this is likely a single-column table
                # Heuristic: Many short lines with similar x-coordinates
                lines = block[4].split("\n")
                
                if len(lines) > 3:  # At least 3 rows
                    # Count lines with tab characters or multiple spaces
                    tabbed_lines = 0
                    for line in lines:
                        if "\t" in line or "  " in line:
                            tabbed_lines += 1
                    
                    # If most lines have tab characters, likely a simple table
                    if tabbed_lines > len(lines) * 0.7:
                        table_areas.append(fitz.Rect(block[:4]))
        
        return table_areas
    
    def _cluster_lines(self, lines, axis=0):
        """Cluster lines based on their position on the specified axis"""
        # Extract positions along the specified axis
        positions = [line[axis] for line in lines]
        
        # Simple clustering - lines within 5 units are considered same cluster
        clusters = []
        for i, line in enumerate(lines):
            pos = positions[i]
            
            # Try to find an existing cluster for this line
            assigned = False
            for cluster in clusters:
                if abs(cluster["position"] - pos) < 5:
                    cluster["lines"].append(line)
                    # Update cluster position (average)
                    cluster["position"] = sum(positions[j] for j, l in enumerate(lines) 
                                           if l in cluster["lines"]) / len(cluster["lines"])
                    assigned = True
                    break
            
            # If not assigned to any cluster, create a new one
            if not assigned:
                clusters.append({"position": pos, "lines": [line]})
        
        return clusters
    
    def _extract_table_image(self, page, table_area):
        """Extract table area as an image"""
        # Render the area to an image
        matrix = fitz.Matrix(2, 2)  # Scale factor for better quality
        pix = page.get_pixmap(matrix=matrix, clip=table_area)
        
        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return img
    
    def _extract_table_data_with_ocr(self, image_path):
        """Extract structured table data using OCR"""
        if not self.use_ocr:
            return None
            
        try:
            # Use PaddleOCR to extract text
            ocr_result = self.ocr.ocr(image_path)
            
            if not ocr_result or not ocr_result[0]:
                return None
            
            # Extract text and positions
            lines = []
            for line in ocr_result[0]:
                text = line[1][0]
                box = line[0]
                
                # Calculate center point
                center_y = sum(p[1] for p in box) / 4
                center_x = sum(p[0] for p in box) / 4
                
                lines.append({
                    "text": text,
                    "center_x": center_x,
                    "center_y": center_y
                })
            
            # Group lines by their y-coordinate (rows)
            rows = self._group_by_position([line["center_y"] for line in lines], lines)
            
            # For each row, sort elements by x-coordinate
            table_data = []
            for row in rows:
                sorted_row = sorted(row, key=lambda x: x["center_x"])
                table_data.append([cell["text"] for cell in sorted_row])
            
            # Convert to pandas DataFrame
            if table_data:
                # Use first row as header if it appears to be a header
                if len(table_data) > 1:
                    df = pd.DataFrame(table_data[1:], columns=table_data[0])
                else:
                    df = pd.DataFrame(table_data)
                
                return df.to_dict(orient="records")
            
        except Exception as e:
            print(f"Error extracting table data: {e}")
            
        return None
    
    def _group_by_position(self, positions, items, threshold=10):
        """Group items by positions within threshold"""
        if not positions:
            return []
            
        # Sort positions and items together
        sorted_pairs = sorted(zip(positions, items), key=lambda x: x[0])
        sorted_positions = [p[0] for p in sorted_pairs]
        sorted_items = [p[1] for p in sorted_pairs]
        
        groups = []
        current_group = [sorted_items[0]]
        current_pos = sorted_positions[0]
        
        for i in range(1, len(sorted_positions)):
            if abs(sorted_positions[i] - current_pos) <= threshold:
                # Add to current group
                current_group.append(sorted_items[i])
            else:
                # Start a new group
                groups.append(current_group)
                current_group = [sorted_items[i]]
                current_pos = sorted_positions[i]
        
        # Add the last group
        groups.append(current_group)
        
        return groups
    
    def _extract_caption_near_table(self, page, table) -> str:
        """Try to extract caption text near a table"""
        try:
            # Get the text around the table
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
            
            return ""
            
        except Exception as e:
            print(f"Error extracting caption: {e}")
            return ""
    
    def _extract_caption_near_image(self, page, img_info) -> str:
        """Try to extract caption text near an image"""
        try:
            # Get image rectangle on page
            xref = img_info[0]
            bbox = page.get_image_bbox(xref)
            
            if bbox:
                # Look for text below the image (common caption location)
                caption_area = fitz.Rect(bbox.x0, bbox.y1, bbox.x1, bbox.y1 + 100)
                caption_text = page.get_text("text", clip=caption_area)
                
                # Clean up text and look for "Figure" mentions
                caption_text = caption_text.strip()
                if caption_text:
                    # If text starts with "Figure", it's likely a caption
                    if re.match(r"^(Figure|Fig\.)\s+\d+", caption_text):
                        return caption_text
                    # Look for "Figure" in the text
                    fig_match = re.search(r"(Figure|Fig\.)\s+\d+", caption_text)
                    if fig_match:
                        return caption_text
                    
                    # Just return the first line if it's short
                    if len(caption_text) < 200:
                        return caption_text
                        
                    # Return first sentence if it's not too long
                    first_sentence = caption_text.split('.')[0]
                    if len(first_sentence) < 100:
                        return first_sentence + "."
        
        except Exception as e:
            print(f"Error extracting caption: {e}")
            
        return ""
    
    def extract_tables(self, pdf_path: str, output_dir: str = "extracted_tables") -> List[Dict[str, Any]]:
        """Extract tables from PDF using pdfplumber"""
        tables = []
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    # Extract tables from the page
                    page_tables = page.extract_tables()
                    
                    for j, table in enumerate(page_tables):
                        if not table:  # Skip empty tables
                            continue
                            
                        # Create a CSV file for the table
                        table_filename = f"table_page{i+1}_{j+1}.csv"
                        table_path = os.path.join(output_dir, table_filename)
                        
                        # Save as CSV
                        with open(table_path, 'w', newline='', encoding='utf-8') as csvfile:
                            writer = csv.writer(csvfile)
                            for row in table:
                                writer.writerow([cell if cell is not None else "" for cell in row])
                        
                        # Convert table to HTML
                        html_table = self._table_to_html(table)
                        
                        # Try to get caption
                        caption = self._extract_caption_near_table(page, table)
                        
                        tables.append({
                            "page": i + 1,
                            "index": j + 1,
                            "path": table_path,
                            "html": html_table,
                            "caption": caption,
                            "rows": len(table),
                            "columns": max(len(row) for row in table) if table else 0
                        })
            
            return tables
            
        except Exception as e:
            print(f"Error extracting tables: {e}")
            return []

    def _table_to_html(self, table: List[List[str]]) -> str:
        """Convert table data to HTML format"""
        if not table:
            return ""
            
        html = ['<table class="table table-bordered">']
        
        # Add header row if it looks like a header
        if len(table) > 1:
            html.append('<thead><tr>')
            for cell in table[0]:
                html.append(f'<th>{cell if cell is not None else ""}</th>')
            html.append('</tr></thead>')
            table = table[1:]
        
        # Add body
        html.append('<tbody>')
        for row in table:
            html.append('<tr>')
            for cell in row:
                html.append(f'<td>{cell if cell is not None else ""}</td>')
            html.append('</tr>')
        html.append('</tbody>')
        html.append('</table>')
        
        return '\n'.join(html)

    def extract_tables_advanced(self, pdf_path):
        try:
            import camelot
            tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
            # Process and return tables
        except ImportError:
            print("Camelot not installed. Install with: pip install camelot-py opencv-python")