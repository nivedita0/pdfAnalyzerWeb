import fitz
import re

def extract_references(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text() for page in doc])
    refs_start = re.search(r'(References|REFERENCES|Bibliography)', text)
    if not refs_start:
        return []

    start_idx = refs_start.end()
    ref_text = text[start_idx:]
    references = re.split(r'\n\d+\.|\n\[?\d+\]?', ref_text)
    return [ref.strip() for ref in references if len(ref.strip()) > 20]
