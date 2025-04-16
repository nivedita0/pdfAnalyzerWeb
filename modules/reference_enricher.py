from bs4 import BeautifulSoup
import requests
from .grobid_client import process_pdf_with_grobid

def extract_references_grobid(pdf_path):
    xml_content = process_pdf_with_grobid(pdf_path)
    if not xml_content:
        return []

    soup = BeautifulSoup(xml_content, 'lxml')
    references = []

    for bibl_struct in soup.find_all('biblstruct'):
        title_tag = bibl_struct.find('title')
        if title_tag:
            title = title_tag.text.strip()
            if len(title) > 10:
                references.append(title)
    return references

def lookup_crossref(title):
    url = "https://api.crossref.org/works"
    params = {"query.bibliographic": title, "rows": 1}
    r = requests.get(url, params=params)
    
    if r.status_code == 200:
        items = r.json()["message"]["items"]
        if items:
            item = items[0]
            authors_raw = item.get("author", [])
            authors = []

            for a in authors_raw:
                if "family" in a:
                    authors.append(a["family"])
                elif "name" in a:
                    authors.append(a["name"])
                else:
                    authors.append("Unknown Author")

            return {
                "title": item.get("title", [""])[0],
                "authors": authors,
                "doi": item.get("DOI"),
                "year": item.get("published-print", {}).get("date-parts", [[None]])[0][0]
            }
    return None

def enrich_references(pdf_path):
    references = extract_references_grobid(pdf_path)
    enriched_data = []

    for ref in references:
        print(f"\n🔍 Looking up: {ref}")
        data = lookup_crossref(ref)
        enriched_data.append({
            "original": ref,
            "metadata": data
        })
    return enriched_data
