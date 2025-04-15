from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import fitz

def chunk_text(text, size=200):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

def ask_question(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text() for page in doc])
    chunks = chunk_text(text)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedder.encode(chunks)
    
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(embeddings)
    
    qa = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

    question = input("Ask a question about the paper: ")
    q_embed = embedder.encode([question])
    _, top_idxs = index.search(q_embed, k=3)
    context = " ".join([chunks[i] for i in top_idxs[0]])
    answer = qa(question=question, context=context)['answer']
    print(f"\n[Answer]: {answer}")
