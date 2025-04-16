from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import fitz

def chunk_text(text, size=200):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

def build_index(chunks):
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedder.encode(chunks)
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(embeddings)
    return embedder, index, embeddings

def get_top_chunks(question, embedder, chunks, index, k=3):
    q_embed = embedder.encode([question])
    _, top_idxs = index.search(q_embed, k)
    return " ".join([chunks[i] for i in top_idxs[0]])

def ask_question(pdf_path):
    # Step 1: Extract and chunk text
    doc = fitz.open(pdf_path)
    full_text = "\n".join([page.get_text() for page in doc])
    chunks = chunk_text(full_text)

    # Step 2: Build vector index
    embedder, index, _ = build_index(chunks)

    # Step 3: Load QA model
    qa = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

    print("\n🧠 Ask anything about this paper (type 'exit' to quit)\n")
    while True:
        question = input("Q: ")
        if question.lower() in ["exit", "quit"]:
            print("👋 Exiting QA session.\n")
            break

        context = get_top_chunks(question, embedder, chunks, index)
        try:
            answer = qa(question=question, context=context)['answer']
            print(f"A: {answer}\n")
        except Exception as e:
            print(f"⚠️ Could not answer: {e}\n")
