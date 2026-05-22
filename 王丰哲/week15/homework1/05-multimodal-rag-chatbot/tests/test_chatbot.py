from rag_chatbot.core import build_default_chatbot


def test_chatbot_ingests_document_and_returns_grounded_answer_with_citations():
    chatbot = build_default_chatbot()

    ingest = chatbot.ingest_document(
        title="Week15 parser notes",
        source="Week15/02_MinerU.md",
        content="""# MinerU

MinerU transforms complex PDFs and Office documents into structured Markdown and JSON.
It is useful for RAG because it keeps reading order, tables, formulas, and images.

![table parse](images/table.png)

# pdfplumber

pdfplumber extracts native PDF text and tables quickly, but it does not perform OCR or layout reconstruction.
""",
    )

    assert ingest["document_id"].startswith("doc_")
    assert ingest["chunk_count"] >= 2

    response = chatbot.chat("MinerU 适合在 RAG 里做什么？", top_k=2)

    assert "MinerU" in response["answer"]
    assert response["citations"]
    assert response["citations"][0]["source"] == "Week15/02_MinerU.md"
    assert response["retrieved_chunks"]
    assert response["used_modalities"]["text"] is True
    assert response["used_modalities"]["images"] is True


def test_chatbot_search_ranks_semantically_relevant_chunks_first():
    chatbot = build_default_chatbot()
    chatbot.ingest_document(
        title="RAG notes",
        source="rag.md",
        content="""# Embedding
Embedding turns text into vectors for similarity search.

# API
The chat API should return answers with citations.
""",
    )

    results = chatbot.search("How should chat answers cite sources?", top_k=1)

    assert len(results) == 1
    assert results[0]["heading"] == "# API"


def test_chatbot_returns_safe_empty_state_answer_before_ingestion():
    chatbot = build_default_chatbot()

    response = chatbot.chat("有什么资料？")

    assert response["answer"] == "当前知识库为空，请先上传或索引文档。"
    assert response["citations"] == []
    assert response["retrieved_chunks"] == []
