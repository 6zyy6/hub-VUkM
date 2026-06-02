from rag_chatbot.chunking import chunk_markdown


def test_chunk_markdown_preserves_headings_and_image_references():
    markdown = """# MinerU

MinerU can parse complex PDF files into markdown.

![layout result](images/page-1.png)

## pdfplumber

pdfplumber is useful for text-based PDFs but does not rebuild visual layout.
"""

    chunks = chunk_markdown(
        title="parser comparison",
        source="week15.md",
        content=markdown,
        max_chars=120,
        overlap=20,
    )

    assert len(chunks) >= 2
    assert chunks[0].heading == "# MinerU"
    assert chunks[0].source == "week15.md"
    assert chunks[0].images == [
        {"alt": "layout result", "path": "images/page-1.png"}
    ]
    assert any(chunk.heading == "## pdfplumber" for chunk in chunks)
    assert all(chunk.chunk_id.startswith("chk_") for chunk in chunks)
