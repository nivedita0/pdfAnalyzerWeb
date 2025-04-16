import fitz
import re
from transformers import pipeline

# Load HuggingFace summarizer once
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text() for page in doc])

def split_sections(text):
    section_titles = [
        "abstract", "introduction", "related work", "background",
        "methodology", "methods", "experiment", "results",
        "discussion", "conclusion", "references"
    ]
    
    # Create regex pattern to find section titles
    pattern = re.compile(rf"\n?({'|'.join(section_titles)})\s*\n", re.IGNORECASE)
    matches = list(pattern.finditer(text))

    sections = {}
    for i, match in enumerate(matches):
        section_name = match.group(1).lower()
        start = match.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        sections[section_name] = text[start:end].strip()
    
    return sections

def smart_summarize(pdf_path, user_query):
    text = extract_text(pdf_path)
    sections = split_sections(text)

    # Try to match the query to a section
    for key in sections:
        if key in user_query.lower():
            content = sections[key]
            if len(content) < 100:
                return f"🔍 Section '{key}' is too short or not found clearly."
            try:
                summary = summarizer(content[:2000], max_length=150, min_length=30, do_sample=False)[0]["summary_text"]
                return f"📘 Summary of '{key.title()}':\n{summary}"
            except Exception as e:
                return f"⚠️ Could not summarize due to: {str(e)}"
    
    return "🤖 I couldn’t identify which section you're referring to. Try asking something like 'Summarize the conclusion'."
