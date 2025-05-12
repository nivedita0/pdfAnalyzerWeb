from flask import Flask, request, render_template, jsonify, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename
import uuid
import json
from datetime import datetime
import csv
import hashlib
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# Import our enhanced modules
from models.extractor import PDFExtractor
#from models.multi_doc_rag import MultiDocumentRAG
from models.summarizer import PaperSummarizer
from models.citation_network import CitationNetwork
from models.topic_modeling import TopicModeler
from models.table_figure_extractor import TableFigureExtractor
from models.zotero import ZoteroIntegration
from models.langchain_rag import LangChainRAG
from models.rule_based_qa import RuleBasedQA
from models.document_analyzer import DocumentAnalyzer
from models.visualizer import DocumentVisualizer

try:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    print(f"Warning: Failed to download NLTK data: {e}")
    
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['EXTRACT_FOLDER'] = 'extracted'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}
app.config['KB_FOLDER'] = 'knowledge_base'

# Create necessary folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['EXTRACT_FOLDER'], exist_ok=True)
os.makedirs(app.config['KB_FOLDER'], exist_ok=True)

# Initialize services
rag_system = LangChainRAG(llm_model="llama3")
summarizer = PaperSummarizer()
citation_network = CitationNetwork()
topic_modeler = TopicModeler()
document_analyzer = DocumentAnalyzer()
zotero = ZoteroIntegration()
rule_based_qa = RuleBasedQA()
table_figure_extractor = TableFigureExtractor()
document_visualizer = DocumentVisualizer()

# In-memory storage for processed documents
document_store = {}
summaries_store = {}
extracted_figures_tables = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    """Render main application page"""
    return render_template('index.html', 
                          zotero_enabled=zotero.is_configured(),
                          documents=list(document_store.values()))

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle PDF upload and processing"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    if file and allowed_file(file.filename):
        # Generate unique filename
        orig_filename = secure_filename(file.filename)
        filename = f"{uuid.uuid4()}_{orig_filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the PDF
        try:
            # Extract metadata
            extractor = PDFExtractor()
            try:
                metadata = extractor.extract_metadata(filepath)
                if not metadata.get("title"):
                    metadata["title"] = orig_filename
            except Exception as metadata_error:
                print(f"Error extracting metadata: {metadata_error}")
                metadata = {"title": orig_filename}
            
            try:
                # Add to RAG system with memory management
                doc_id = rag_system.add_document(filepath)
                
                rag_system.debug_show_document(doc_id)
                
                # Store document info
                document_store[doc_id] = {
                    "id": doc_id,
                    "path": filepath,
                    "filename": orig_filename,
                    "title": metadata.get("title", orig_filename),
                    "authors": metadata.get("authors", []),
                    "year": metadata.get("year", ""),
                    "doi": metadata.get("doi", ""),
                    "upload_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # Now register AFTER document is created
                rule_based_qa.register_document(doc_id, document_store[doc_id])
                
                return jsonify({
                    'success': True,
                    'message': f'File "{orig_filename}" uploaded and processed successfully',
                    'document_id': doc_id,
                    'document': document_store[doc_id]
                })
            except Exception as process_error:
                print(f"Error processing document: {process_error}")
                return jsonify({'error': f'Error processing document: {str(process_error)}'}), 500
                
        except Exception as e:
            print(f"Upload error: {e}")
            return jsonify({'error': f'Error processing PDF: {str(e)}'}), 500

@app.route('/documents', methods=['GET'])
def get_documents():
    """Get list of all processed documents"""
    return jsonify({
        'success': True,
        'documents': list(document_store.values())
    })

@app.route('/document/<doc_id>', methods=['GET'])
def get_document(doc_id):
    """Get information about a specific document"""
    if doc_id in document_store:
        return jsonify({
            'success': True,
            'document': document_store[doc_id]
        })
    else:
        return jsonify({'error': 'Document not found'}), 404

@app.route('/ask', methods=['POST'])
def ask_question():
    """Answer questions about the loaded documents with improved reliability"""
    data = request.json
    question = data.get('question')
    doc_id = data.get('document_id', None)  # Optional, to restrict to one document
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    try:
        # Log question for debugging
        print(f"Processing question: {question}")
        
        simple_questions = [
            "title", "what is the title", "what is the document",
            "author", "who wrote", "who is the author",
            "when", "year", "publication date"
        ]

         # If it looks like a simple question, try rule-based first
        if any(term in question.lower() for term in simple_questions):
            try:
                # Assume rule_based_qa is already initialized
                rule_response = rule_based_qa.answer_question(question, doc_id)
                
                # If the rule-based QA gave a real answer (not a default response)
                if "I don't have enough information" not in rule_response["answer"]:
                    return jsonify({
                        'success': True,
                        'question': question,
                        'answer': rule_response['answer'],
                        'sources': rule_response.get('sources', [])
                    })
            except Exception as rule_error:
                print(f"Rule-based QA error (continuing to main QA): {rule_error}")
        
        # Use hybrid answer approach
        if hasattr(rag_system, 'hybrid_answer'):
            response = rag_system.hybrid_answer(question)
        else:
            # Fall back to regular answer_question
            response = rag_system.answer_question(question)
        
        # Enhanced error handling
        if 'error' in response.get('answer', '').lower() or not response.get('answer'):
            print("Using fallback response mechanism")
            
            # Get relevant chunks without using LLM
            fallback_answer = "I'm having trouble connecting to the language model. Here's some related information I found:"
            
            # Extract most relevant text from documents based on question
            extractor = PDFExtractor()
            
            # Try to find the document that best matches the question if no doc_id provided
            best_doc_id = doc_id
            if not best_doc_id and document_store:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity
                
                # Create TF-IDF representation of the question
                all_texts = {}
                for id, doc in document_store.items():
                    try:
                        all_texts[id] = extractor.extract_text(doc['path'])[:1000]  # First 1000 chars
                    except:
                        continue
                
                if all_texts:
                    corpus = list(all_texts.values())
                    vectorizer = TfidfVectorizer(stop_words='english')
                    try:
                        tfidf_matrix = vectorizer.fit_transform(corpus + [question])
                        doc_similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])[0]
                        best_match_idx = doc_similarities.argmax()
                        best_doc_id = list(all_texts.keys())[best_match_idx]
                    except:
                        pass
            
            # Use the best document or iterate through all if none found
            if best_doc_id and best_doc_id in document_store:
                try:
                    text = extractor.extract_text(document_store[best_doc_id]['path'])
                    
                    # Find potentially relevant sections
                    import re
                    sections = re.split(r'\n\s*(?:[0-9]+\.)+\s+([A-Z][A-Za-z\s]+)\s*\n|^\s*([A-Z][A-Z\s]+)\s*$', 
                                      text, flags=re.MULTILINE)
                    
                    relevant_text = ""
                    query_terms = set(re.findall(r'\b[a-zA-Z]{3,}\b', question.lower()))
                    query_terms = set([term for term in query_terms if term not in ('the', 'and', 'that', 'what', 'when', 'where', 'who', 'why', 'how')])
                    
                    # Score sections by term overlap
                    for section in sections:
                        section_terms = set(re.findall(r'\b[a-zA-Z]{3,}\b', section.lower()))
                        overlap = len(query_terms.intersection(section_terms))
                        if overlap > 0:
                            relevant_text += section[:500] + "...\n\n"
                            if len(relevant_text) > 1500:
                                break
                    
                    if relevant_text:
                        fallback_answer += f"\n\nFrom document: {document_store[best_doc_id]['title']}\n\n{relevant_text}"
                    else:
                        # Just return start of the document if no relevant sections found
                        fallback_answer += f"\n\nFrom document: {document_store[best_doc_id]['title']}\n\n{text[:1500]}..."
                except Exception as section_error:
                    print(f"Error finding relevant sections: {section_error}")
                    fallback_answer += f"\n\nCould not extract relevant sections due to an error."
            
            response['answer'] = fallback_answer
           
        return jsonify({
            'success': True,
            'question': question,
            'answer': response['answer'],
            'sources': response.get('sources', [])
        })
    except Exception as e:
        print(f"Error answering question: {e}")
        return jsonify({'error': f'Error answering question: {str(e)}'}), 500

@app.route('/summarize/<doc_id>', methods=['GET'])
def summarize_document(doc_id):
    """Generate summary of the document using smart summarization"""
    if doc_id not in document_store:
        return jsonify({'error': 'Document not found'}), 404
        
    try:
        # Get the document path
        doc_path = document_store[doc_id]['path']
        
        # Use smart summarization
        summary_result = summarizer.smart_summarize(doc_path)
            
        return jsonify({
            'success': True,
            'summary': summary_result['summary'],
            'key_points': summary_result.get('key_points', []),
            'method': summary_result.get('method', 'smart')
        })
    except Exception as e:
        print(f"Error generating summary: {e}")
        return jsonify({'error': f'Error generating summary: {str(e)}'}), 500

@app.route('/extract/<doc_id>/figures', methods=['GET'])
def extract_figures(doc_id):
    """Extract figures from a document"""
    if doc_id not in document_store:
        return jsonify({'error': 'Document not found'}), 404
        
    try:
        # Check if we already extracted figures
        if doc_id in extracted_figures_tables and 'figures' in extracted_figures_tables[doc_id]:
            figures = extracted_figures_tables[doc_id]['figures']
        else:
            # Create extraction folder for this document
            extract_dir = os.path.join(app.config['EXTRACT_FOLDER'], doc_id, 'figures')
            os.makedirs(extract_dir, exist_ok=True)
            
            # Extract figures
            doc_path = document_store[doc_id]['path']
            print(f"Extracting figures from {doc_path} to {extract_dir}")  # Debug log
            figures = table_figure_extractor.extract_figures(doc_path, output_dir=extract_dir)
            
            # Update paths to be relative URLs
            for figure in figures:
                if 'path' in figure:
                    # Get the relative path from the extract_dir
                    rel_path = os.path.relpath(figure['path'], app.config['EXTRACT_FOLDER'])
                    figure['url'] = url_for('get_extracted_file', filepath=rel_path)
                    print(f"Figure URL: {figure['url']}")  # Debug log
            
            # Store for future reference
            if doc_id not in extracted_figures_tables:
                extracted_figures_tables[doc_id] = {}
            extracted_figures_tables[doc_id]['figures'] = figures
        
        return jsonify({
            'success': True,
            'document_id': doc_id,
            'figures': figures
        })
    except Exception as e:
        print(f"Error extracting figures: {e}")
        return jsonify({'error': f'Error extracting figures: {str(e)}'}), 500

@app.route('/extract/<doc_id>/tables', methods=['GET'])
def extract_tables(doc_id):
    """Extract tables from a document"""
    if doc_id not in document_store:
        return jsonify({"error": "Document not found"}), 404
        
    try:
        # Check if we already extracted tables
        if doc_id in extracted_figures_tables and 'tables' in extracted_figures_tables[doc_id]:
            tables = extracted_figures_tables[doc_id]['tables']
        else:
            # Create extraction folder for this document
            extract_dir = os.path.join(app.config['EXTRACT_FOLDER'], doc_id, 'tables')
            os.makedirs(extract_dir, exist_ok=True)
            
            # Extract tables
            doc_path = document_store[doc_id]['path']
            print(f"Extracting tables from {doc_path} to {extract_dir}")  # Debug log
            tables = table_figure_extractor.extract_tables(doc_path, output_dir=extract_dir)
            
            # Update paths to be relative URLs
            for table in tables:
                if 'path' in table:
                    # Get the relative path from the extract_dir
                    rel_path = os.path.relpath(table['path'], app.config['EXTRACT_FOLDER'])
                    table['url'] = url_for('get_extracted_file', filepath=rel_path)
                    print(f"Table URL: {table['url']}")  # Debug log
            
            # Store for future reference
            if doc_id not in extracted_figures_tables:
                extracted_figures_tables[doc_id] = {}
            extracted_figures_tables[doc_id]['tables'] = tables
        
        return jsonify({
            'success': True,
            'document_id': doc_id,
            'tables': tables
        })
    except Exception as e:
        print(f"Error extracting tables: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/extracted/<path:filepath>')
def get_extracted_file(filepath):
    """Serve extracted files"""
    try:
        # Convert Windows backslashes to forward slashes
        filepath = filepath.replace('\\', '/')
        print(f"Attempting to serve file: {filepath}")  # Debug log
        
        # Get the directory and filename
        directory = os.path.dirname(os.path.join(app.config['EXTRACT_FOLDER'], filepath))
        filename = os.path.basename(filepath)
        
        print(f"Serving from directory: {directory}")  # Debug log
        print(f"Filename: {filename}")  # Debug log
        
        if os.path.exists(os.path.join(directory, filename)):
            return send_from_directory(directory, filename)
        else:
            print(f"File not found: {os.path.join(directory, filename)}")  # Debug log
            return jsonify({"error": "File not found"}), 404
    except Exception as e:
        print(f"Error serving file {filepath}: {e}")
        return jsonify({"error": "File not found"}), 404

@app.route('/citation-network', methods=['GET'])
def analyze_citations():
    """Generate citation network visualization data"""
    data = request.json
    doc_id = data.get('document_id')
    time_range = data.get('time_range', 'all')
    connection_type = data.get('connection_type', 'citations')
    
    if not doc_id or doc_id not in document_store:
        return jsonify({'error': 'Document not found'}), 404
        
    try:
        # Get citation network data
        network_data = citation_network.generate_network(
            doc_id,
            time_range=time_range,
            connection_type=connection_type
        )
        
        return jsonify({
            'success': True,
            'nodes': network_data['nodes'],
            'edges': network_data['edges']
        })
    except Exception as e:
        print(f"Error generating citation network: {e}")
        return jsonify({'error': f'Error generating citation network: {str(e)}'}), 500

@app.route('/topic-modeling', methods=['GET'])
def analyze_topics():
    """Perform topic modeling across all documents"""
    try:
        # Add all documents to the topic modeler
        for doc_id, doc in document_store.items():
            topic_modeler.add_paper(doc['path'])
        
        # Build topic model
        topic_modeler.build_topic_model()
        
        # Generate visualizations
        dist_vis = topic_modeler.visualize_topic_distribution(
            output_path=os.path.join(app.config['EXTRACT_FOLDER'], 'topic_distribution.html'))
        
        cluster_vis = topic_modeler.visualize_topic_clusters(
            output_path=os.path.join(app.config['EXTRACT_FOLDER'], 'topic_clusters.html'))
        
        # Get keywords for each topic
        keywords = topic_modeler.get_topic_keywords()
        
        return jsonify({
            'success': True,
            'topics_count': len(keywords),
            'topic_keywords': keywords,
            'distribution_visualization': '/extracted/topic_distribution.html',
            'clusters_visualization': '/extracted/topic_clusters.html'
        })
    except Exception as e:
        return jsonify({'error': f'Error analyzing topics: {str(e)}'}), 500

@app.route('/uploads/<path:filename>', methods=['GET'])
def get_uploaded_file(filename):
    """Serve uploaded PDF files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/save-knowledge-base', methods=['POST'])
def save_knowledge_base():
    """Save the current knowledge base for future use"""
    try:
        rag_system.save_knowledge_base(app.config['KB_FOLDER'])
        return jsonify({
            'success': True,
            'message': 'Knowledge base saved successfully',
            'path': app.config['KB_FOLDER']
        })
    except Exception as e:
        return jsonify({'error': f'Error saving knowledge base: {str(e)}'}), 500

@app.route('/load-knowledge-base', methods=['POST'])
def load_knowledge_base():
    """Load a previously saved knowledge base"""
    try:
        rag_system.load_knowledge_base(app.config['KB_FOLDER'])
        return jsonify({
            'success': True,
            'message': 'Knowledge base loaded successfully'
        })
    except Exception as e:
        return jsonify({'error': f'Error loading knowledge base: {str(e)}'}), 500

@app.route('/zotero/collections', methods=['GET'])
def get_zotero_collections():
    """Get Zotero collections"""
    if not zotero.is_configured():
        return jsonify({'error': 'Zotero not configured'}), 400
        
    collections = zotero.get_collections()
    return jsonify(collections)

@app.route('/zotero/add', methods=['POST'])
def add_to_zotero():
    """Add current document to Zotero"""
    data = request.json
    doc_id = data.get('document_id')
    
    if not zotero.is_configured():
        return jsonify({'error': 'Zotero not configured'}), 400
    
    if not doc_id or doc_id not in document_store:
        return jsonify({'error': 'Document not found'}), 404
    
    try:
        # Get document metadata
        doc_metadata = document_store[doc_id]
        
        # Format metadata for Zotero
        metadata = {
            'title': doc_metadata.get('title', ''),
            'authors': doc_metadata.get('authors', []),
            'year': doc_metadata.get('year', ''),
            'doi': doc_metadata.get('doi', ''),
            'url': doc_metadata.get('url', ''),
            'abstract': doc_metadata.get('abstract', ''),
            'journal': doc_metadata.get('journal', ''),
            'volume': doc_metadata.get('volume', ''),
            'issue': doc_metadata.get('issue', ''),
            'pages': doc_metadata.get('pages', '')
        }
        
        # Remove empty values
        metadata = {k: v for k, v in metadata.items() if v}
        
        success = zotero.add_item(metadata)
        if success:
            return jsonify({
                'success': True, 
                'message': 'Added to Zotero successfully',
                'metadata': metadata
            })
        else:
            return jsonify({'error': 'Failed to add to Zotero'}), 500
    except Exception as e:
        print(f"Error adding to Zotero: {str(e)}")
        return jsonify({'error': f'Error adding to Zotero: {str(e)}'}), 500

@app.route('/compare-documents', methods=['POST'])
def compare_documents():
    """Compare two documents for similarity"""
    data = request.json
    doc_id_1 = data.get('document_id_1')
    doc_id_2 = data.get('document_id_2')
    
    if not doc_id_1 or not doc_id_2:
        return jsonify({'error': 'Two document IDs are required'}), 400
        
    if doc_id_1 not in document_store or doc_id_2 not in document_store:
        return jsonify({'error': 'One or both documents not found'}), 404
    
    try:
        # Get document paths
        path1 = document_store[doc_id_1]['path']
        path2 = document_store[doc_id_2]['path']
        
        # Extract text
        extractor = PDFExtractor()
        text1 = extractor.extract_text(path1)
        text2 = extractor.extract_text(path2)
        
        # Use the embedder from the RAG system to compute similarity
        embedder = rag_system.embedder
        embed1 = embedder.encode([text1[:5000]])  # Limit size for practicality
        embed2 = embedder.encode([text2[:5000]])
        
        # Compute cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarity = float(cosine_similarity(embed1, embed2)[0][0])
        
        return jsonify({
            'success': True,
            'document1': document_store[doc_id_1]['title'],
            'document2': document_store[doc_id_2]['title'],
            'similarity_score': similarity,
            'common_topics': similarity > 0.7  # Simple threshold for common topics
        })
    except Exception as e:
        return jsonify({'error': f'Error comparing documents: {str(e)}'}), 500

@app.route('/search', methods=['GET'])
def semantic_search():
    """Perform semantic search across documents"""
    query = request.args.get('q')
    if not query:
        return jsonify({'error': 'No search query provided'}), 400
    
    try:
        results = rag_system.semantic_search(query)
        return jsonify({
            'success': True,
            'query': query,
            'results': results
        })
    except Exception as e:
        return jsonify({'error': f'Error in semantic search: {str(e)}'}), 500

@app.route('/document-analysis/<doc_id>', methods=['GET'])
def get_document_analysis(doc_id):
    """Get comprehensive document analysis including figures, tables, and citations"""
    if doc_id not in document_store:
        return jsonify({'error': 'Document not found'}), 404
        
    try:
        doc_path = document_store[doc_id]['path']
        
        # Create extraction directories
        figures_dir = os.path.join(app.config['EXTRACT_FOLDER'], doc_id, 'figures')
        tables_dir = os.path.join(app.config['EXTRACT_FOLDER'], doc_id, 'tables')
        os.makedirs(figures_dir, exist_ok=True)
        os.makedirs(tables_dir, exist_ok=True)
        
        # Extract figures
        figures = document_analyzer.extract_figures(doc_path, figures_dir)
        
        # Extract tables
        tables = document_analyzer.extract_tables(doc_path, tables_dir)
        
        # Extract citations
        citations = document_analyzer.extract_citations(doc_path)
        
        return jsonify({
            'success': True,
            'figures': figures,
            'tables': tables,
            'citations': citations
        })
    except Exception as e:
        print(f"Error analyzing document: {e}")
        return jsonify({'error': f'Error analyzing document: {str(e)}'}), 500

@app.route('/zotero/test-connection', methods=['GET'])
def test_zotero_connection():
    """Test Zotero API connection and credentials"""
    if not zotero.is_configured():
        return jsonify({
            'success': False,
            'message': 'Zotero is not configured. Please set ZOTERO_API_KEY and ZOTERO_USER_ID'
        }), 400
    
    try:
        # Try to get collections as a test
        collections = zotero.get_collections()
        return jsonify({
            'success': True,
            'message': 'Successfully connected to Zotero',
            'collections': collections
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Failed to connect to Zotero: {str(e)}'
        }), 500

@app.route('/zotero/search', methods=['POST'])
def search_zotero():
    """Search Zotero library"""
    if not zotero.is_configured():
        return jsonify({'error': 'Zotero not configured'}), 400
    
    data = request.json
    query = data.get('query')
    item_type = data.get('item_type')
    
    if not query:
        return jsonify({'error': 'No search query provided'}), 400
    
    try:
        items = zotero.search_items(query=query, item_type=item_type)
        return jsonify({
            'success': True,
            'items': items
        })
    except Exception as e:
        print(f"Error searching Zotero: {e}")
        return jsonify({'error': f'Error searching Zotero: {str(e)}'}), 500

@app.route('/zotero/item/<item_key>', methods=['GET'])
def get_zotero_item(item_key):
    """Get detailed information about a Zotero item"""
    if not zotero.is_configured():
        return jsonify({'error': 'Zotero not configured'}), 400
    
    try:
        item = zotero.get_item_details(item_key)
        if item:
            return jsonify({
                'success': True,
                'item': item
            })
        else:
            return jsonify({'error': 'Item not found'}), 404
    except Exception as e:
        print(f"Error getting Zotero item: {e}")
        return jsonify({'error': f'Error getting Zotero item: {str(e)}'}), 500

@app.route('/visualize/<doc_id>', methods=['GET'])
def visualize_document(doc_id):
    """Generate visualizations for a document"""
    if doc_id not in document_store:
        return jsonify({'error': 'Document not found'}), 404
        
    try:
        # Get the document path
        doc_path = document_store[doc_id]['path']
        
        # Generate visualizations
        results = document_visualizer.generate_all_visualizations(doc_path)
        
        if "error" in results:
            return jsonify({'error': results["error"]}), 500
            
        return jsonify({
            'success': True,
            'document_id': doc_id,
            'visualizations': results['visualizations'],
            'numeric_data': results['numeric_data'],
            'non_numeric_data': results['non_numeric_data']
        })
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        return jsonify({'error': f'Error generating visualizations: {str(e)}'}), 500

@app.route('/visualize/<doc_id>/<viz_type>', methods=['GET'])
def get_visualization(doc_id, viz_type):
    """Get a specific type of visualization for a document"""
    if doc_id not in document_store:
        return jsonify({'error': 'Document not found'}), 404
        
    try:
        # Get the document path
        doc_path = document_store[doc_id]['path']
        
        # Extract text and process data
        text = document_visualizer.extract_text_from_pdf(doc_path)
        _, numeric_data, _ = document_visualizer.extract_and_process_data(text)
        
        if not numeric_data:
            return jsonify({'error': 'No numeric data found for visualization'}), 400
            
        # Generate the specific visualization
        result = document_visualizer.generate_visualization(numeric_data, viz_type)
        
        if "error" in result:
            return jsonify({'error': result["error"]}), 500
            
        # Add explanation
        result['explanation'] = document_visualizer.get_visualization_explanation(viz_type, numeric_data)
        
        return jsonify({
            'success': True,
            'document_id': doc_id,
            'visualization': result
        })
    except Exception as e:
        print(f"Error generating visualization: {e}")
        return jsonify({'error': f'Error generating visualization: {str(e)}'}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Disable reloader to prevent the PyTorch inductor issue
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)