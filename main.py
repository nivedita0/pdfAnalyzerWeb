from modules.summarizer import summarize_pdf
from modules.citation_extractor import extract_references
from modules.semantic_qa import ask_question
from modules.citation_graph import plot_citation_graph
from modules.reference_enricher import enrich_references
from modules.grobid_client import process_pdf_with_grobid
from modules.citation_graph_builder import build_citation_edges, draw_citation_graph
from modules.sectional_summarizer import smart_summarize
from modules.smart_qa_router import run_smart_qa

def main():
    pdf_path = "A_Review_of_Cyber_Threats_to_Medical_Devices_Integration_with_Electronic_Medical_Records.pdf"

    # print("\n==== [1] SUMMARIZATION ====")
    # summary = summarize_pdf(pdf_path)
    # print("\n[Summary Preview]\n", summary[:500], "...\n")

    # print("\n==== [2] REFERENCE EXTRACTION ====")
    # references = extract_references(pdf_path)
    # print(f"\n[Found {len(references)} references]")
    # for r in references[:5]:
    #     print("-", r)

    # print("\n==== [2] REFERENCE EXTRACTION (GROBID + Crossref) ====")
    # enriched_refs = enrich_references(pdf_path)

    # for ref in enriched_refs[:5]:
    #     print("\nOriginal Title:", ref["original"])
    #     if ref["metadata"]:
    #         print("Enriched Metadata:")
    #         print(" - Title:", ref["metadata"]["title"])
    #         print(" - Authors:", ", ".join(ref["metadata"]["authors"]))
    #         print(" - DOI:", ref["metadata"]["doi"])
    #         print(" - Year:", ref["metadata"]["year"])
    #     else:
    #         print(" - No metadata found.")
        
    # print("\n==== [3] SEMANTIC SEARCH & QA ====")
    # ask_question(pdf_path)

    # print("\n==== [4] CITATION GRAPH VISUALIZATION ====")
    # dummy_citations = {
    #     "Paper A": ["Paper B", "Paper C"],
    #     "Paper B": ["Paper C"],
    #     "Paper D": ["Paper A", "Paper C"]
    # }
    # plot_citation_graph(dummy_citations)

    # print("\n==== [5] BUILD CITATION GRAPH ACROSS PAPERS ====")
    # edges, files = build_citation_edges("../")
    # draw_citation_graph(edges, files)

    # print("\n==== [SMART SUMMARIZER] ====")
    # while True:
    #     q = input("Ask for section summary (or type 'exit'): ")
    #     if q.lower() == "exit":
    #         break
    #     print(smart_summarize("sample.pdf", q))

    print("\n==== [SMART QA ROUTER] ====")
    run_smart_qa(pdf_path)

if __name__ == "__main__":
    main()
