from modules.summarizer import summarize_pdf
from modules.citation_extractor import extract_references
from modules.semantic_qa import ask_question
from modules.citation_graph import plot_citation_graph

def main():
    pdf_path = "A_Review_of_Cyber_Threats_to_Medical_Devices_Integration_with_Electronic_Medical_Records.pdf"

    print("\n==== [1] SUMMARIZATION ====")
    summary = summarize_pdf(pdf_path)
    print("\n[Summary Preview]\n", summary[:500], "...\n")

    print("\n==== [2] REFERENCE EXTRACTION ====")
    references = extract_references(pdf_path)
    print(f"\n[Found {len(references)} references]")
    for r in references[:5]:
        print("-", r)

    print("\n==== [3] SEMANTIC SEARCH & QA ====")
    ask_question(pdf_path)

    print("\n==== [4] CITATION GRAPH VISUALIZATION ====")
    dummy_citations = {
        "Paper A": ["Paper B", "Paper C"],
        "Paper B": ["Paper C"],
        "Paper D": ["Paper A", "Paper C"]
    }
    plot_citation_graph(dummy_citations)

if __name__ == "__main__":
    main()
