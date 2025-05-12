from flask import Flask, render_template, request, jsonify
import requests
from scholarly import scholarly
import json
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import urllib.parse
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from docx import Document
import re

app = Flask(__name__)
CORS(app)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'doc', 'docx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_references_from_text(text):
    # Basic reference patterns
    reference_patterns = [
        r'\[\d+\]\s*(.*?)(?=\[\d+\]|\Z)',  # [1] Author, Title...
        r'References:\s*(.*?)(?=\n\n|\Z)',  # References: Author, Title...
        r'REFERENCES\s*(.*?)(?=\n\n|\Z)',   # REFERENCES Author, Title...
        r'\d+\.\s*(.*?)(?=\d+\.\s*|\Z)',    # 1. Author, Title...
    ]
    
    references = []
    for pattern in reference_patterns:
        matches = re.finditer(pattern, text, re.DOTALL)
        for match in matches:
            ref_text = match.group(1).strip()
            if ref_text:
                # Try to extract title and authors
                parts = ref_text.split('. ', 1)
                authors = parts[0] if len(parts) > 1 else ''
                title = parts[1] if len(parts) > 1 else parts[0]
                
                references.append({
                    'title': title.strip(),
                    'authors': authors.strip(),
                    'url': None  # We could potentially search for DOIs or URLs in the text
                })
    
    return references

def extract_text_from_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
        return ""

def extract_text_from_docx(file_path):
    try:
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        print(f"Error extracting text from DOCX: {str(e)}")
        return ""

def extract_text_from_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error extracting text from TXT: {str(e)}")
        return ""

@app.route('/analyze_paper', methods=['POST'])
def analyze_paper():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'File type not supported'})
    
    try:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Extract text based on file type
        text = ""
        if filename.endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
        elif filename.endswith('.docx'):
            text = extract_text_from_docx(file_path)
        elif filename.endswith('.txt'):
            text = extract_text_from_txt(file_path)
        
        # Clean up the uploaded file
        os.remove(file_path)
        
        if not text:
            return jsonify({'success': False, 'error': 'Could not extract text from file'})
        
        # Extract references
        references = extract_references_from_text(text)
        
        # For citations, we'll search in Google Scholar
        citations = []
        try:
            # Extract potential title from the first few lines
            title = text.split('\n')[0].strip()
            search_query = scholarly.search_pubs(title)
            pub = next(search_query, None)
            
            if pub:
                # Get citing papers
                if hasattr(pub, 'citedby'):
                    cite_query = scholarly.citedby(pub)
                    for i, citation in enumerate(cite_query):
                        if i >= 10:  # Limit to 10 citations
                            break
                        citations.append({
                            'title': citation.bib.get('title', ''),
                            'authors': citation.bib.get('author', ''),
                            'url': citation.bib.get('url', None)
                        })
        except Exception as e:
            print(f"Error fetching citations: {str(e)}")
        
        return jsonify({
            'success': True,
            'references': references,
            'citations': citations
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def format_date(date_str):
    try:
        if isinstance(date_str, datetime):
            return date_str.strftime('%Y-%m-%d')
        return date_str
    except:
        return 'N/A'

def search_arxiv(query, year_filter=None, sort_by='relevance', max_results=10):
    base_url = "http://export.arxiv.org/api/query"
    
    # Add year filter to query if specified
    if year_filter:
        years_ago = datetime.now() - timedelta(days=365 * int(year_filter))
        date_query = f' AND submittedDate:[{years_ago.strftime("%Y%m%d")}000000 TO now]'
        query = query + date_query

    sort_mapping = {
        'relevance': 'relevance',
        'date': 'lastUpdatedDate',
        'citations': 'relevance'  # arXiv doesn't support citation sorting
    }

    params = {
        'search_query': f'all:{query}',
        'start': 0,
        'max_results': max_results,
        'sortBy': sort_mapping[sort_by],
        'sortOrder': 'descending'
    }
    
    response = requests.get(base_url, params=params)
    soup = BeautifulSoup(response.content, 'xml')
    
    results = []
    for entry in soup.find_all('entry'):
        title = entry.title.text.strip()
        authors = ', '.join([author.text.strip() for author in entry.find_all('author')])
        summary = entry.summary.text.strip()
        published = format_date(entry.published.text.strip())
        url = entry.id.text.strip()
        
        results.append({
            'title': title,
            'authors': authors,
            'summary': summary,
            'published': published,
            'url': url,
            'citations': 'N/A',
            'citation_url': None,
            'source': 'arXiv'
        })
    
    return results

def search_semantic_scholar(query, year_filter=None, sort_by='relevance', max_results=10):
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    
    params = {
        'query': query,
        'limit': max_results,
        'fields': 'title,authors,abstract,year,citationCount,url'
    }

    if year_filter:
        current_year = datetime.now().year
        params['year'] = f"{current_year - int(year_filter)}-{current_year}"

    response = requests.get(base_url, params=params)
    data = response.json()
    
    results = []
    for paper in data.get('data', []):
        results.append({
            'title': paper.get('title', ''),
            'authors': ', '.join([author.get('name', '') for author in paper.get('authors', [])]),
            'summary': paper.get('abstract', ''),
            'published': paper.get('year', 'N/A'),
            'url': paper.get('url', '#'),
            'citations': paper.get('citationCount', 0),
            'citation_url': f"https://www.semanticscholar.org/paper/{paper.get('paperId')}/citations",
            'source': 'Semantic Scholar'
        })
    
    if sort_by == 'citations':
        results.sort(key=lambda x: x['citations'], reverse=True)
    elif sort_by == 'date':
        results.sort(key=lambda x: x['published'], reverse=True)
    
    return results

def search_google_scholar(query, year_filter=None, sort_by='relevance', max_results=10):
    results = []
    try:
        search_query = scholarly.search_pubs(query)
        count = 0
        
        for paper in search_query:
            if count >= max_results:
                break
                
            # Apply year filter if specified
            if year_filter:
                pub_year = paper.bib.get('year')
                if pub_year and int(pub_year) < datetime.now().year - int(year_filter):
                    continue
            
            results.append({
                'title': paper.bib.get('title', ''),
                'authors': paper.bib.get('author', ''),
                'summary': paper.bib.get('abstract', ''),
                'published': paper.bib.get('year', 'N/A'),
                'url': paper.bib.get('url', '#'),
                'citations': paper.citedby or 0,
                'citation_url': paper.citedby_url,
                'source': 'Google Scholar'
            })
            count += 1
            
        if sort_by == 'citations':
            results.sort(key=lambda x: x['citations'], reverse=True)
        elif sort_by == 'date':
            results.sort(key=lambda x: x['published'], reverse=True)
            
    except Exception as e:
        print(f"Error searching Google Scholar: {str(e)}")
        
    return results

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.get_json()
        query = data.get('query', '')
        sources = data.get('sources', [])
        year_filter = data.get('yearFilter')
        sort_by = data.get('sortBy', 'relevance')
        
        if not query:
            return jsonify({'success': False, 'error': 'No search query provided'})
            
        if not sources:
            return jsonify({'success': False, 'error': 'No sources selected'})
            
        results = []
        
        for source in sources:
            if source == 'arxiv':
                results.extend(search_arxiv(query, year_filter, sort_by))
            elif source == 'semantic_scholar':
                results.extend(search_semantic_scholar(query, year_filter, sort_by))
            elif source == 'google_scholar':
                results.extend(search_google_scholar(query, year_filter, sort_by))
        
        # Final sorting if needed
        if sort_by == 'citations':
            results.sort(key=lambda x: x['citations'] if isinstance(x['citations'], (int, float)) else -1, reverse=True)
        elif sort_by == 'date':
            results.sort(key=lambda x: str(x['published']), reverse=True)
            
        return jsonify({'success': True, 'results': results})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True) 