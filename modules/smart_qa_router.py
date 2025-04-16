import fitz
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss

def chunk_text(text, size=200):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

def build_index(chunks, embed_model):
    embeddings = embed_model.encode(chunks)
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(embeddings)
    return index, embeddings

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text() for page in doc])

def get_top_chunks(question, embed_model, chunks, index, k=3):
    q_embed = embed_model.encode([question])
    _, top_idxs = index.search(q_embed, k)
    return [chunks[i] for i in top_idxs[0]]

def run_smart_qa(pdf_path):
    text = extract_text(pdf_path)
    chunks = chunk_text(text)
    
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    index, _ = build_index(chunks, embed_model)

    extractive_qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

    # Load T5 for RAG mode
    t5_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    t5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

    print("\n🔍 Ask anything about the paper (type 'exit' to quit)")
    while True:
        question = input("\nYour Question: ")
        if question.lower() in ["exit", "quit"]:
            break

        mode = input("Choose mode - [1] Fast QA  [2] Smart RAG: ")
        top_chunks = get_top_chunks(question, embed_model, chunks, index)

        if mode.strip() == "1":
            context = " ".join(top_chunks)
            try:
                answer = extractive_qa(question=question, context=context)['answer']
                print("💬 [Fast QA Answer]:", answer)
            except Exception as e:
                print("⚠️ Extractive QA failed:", e)

        elif mode.strip() == "2":
            prompt = f"Answer this based on the following content:\n{''.join(top_chunks)}\n\nQuestion: {question}"
            input_ids = t5_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).input_ids
            output_ids = t5_model.generate(input_ids, max_length=256)
            answer = t5_tokenizer.decode(output_ids[0], skip_special_tokens=True)
            print("🤖 [RAG Answer]:", answer)
        else:
            print("Invalid mode. Choose 1 or 2.")
