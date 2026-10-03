def format_answer_with_sources(result):
    answer = result["answer"]
    sources = result.get("source_documents", [])

    unique_sources = set()
    for doc in sources:
        page = doc.metadata.get("page", "?")
        source = doc.metadata.get("source", "?")
        unique_sources.add(f"{source} — Page {page}")

    if unique_sources:
        citation = "\n\n**Source:**\n" + "\n".join(f"- {s}" for s in unique_sources)
        return answer + citation
    return answer
