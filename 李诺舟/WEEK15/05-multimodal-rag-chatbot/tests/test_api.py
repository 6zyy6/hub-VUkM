from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app


def create_client(tmp_path: Path) -> TestClient:
    app = create_app(upload_dir=tmp_path / "uploads")
    return TestClient(app)


def test_upload_document_and_chat_round_trip(tmp_path: Path) -> None:
    client = create_client(tmp_path)

    upload_response = client.post(
        "/upload/document",
        data={"knowledge_base_id": "kb-homework"},
        files={
            "file": (
                "intro.txt",
                "Claude Code can orchestrate parser workers. MinerU focuses on layout-aware PDF parsing.",
                "text/plain",
            )
        },
    )

    assert upload_response.status_code == 200
    payload = upload_response.json()
    assert payload["knowledge_base_id"] == "kb-homework"
    assert payload["status"] == "queued"

    document = client.app.state.store.get_document(payload["document_id"])
    assert document is not None
    assert document.status == "processed"

    chat_response = client.post(
        "/chat",
        json={
            "knowledge_base_id": "kb-homework",
            "question": "谁负责 PDF 版面解析？",
            "top_k": 2,
        },
    )

    assert chat_response.status_code == 200
    result = chat_response.json()
    assert "MinerU" in result["answer"]
    assert result["sources"]
    assert result["sources"][0]["filename"] == "intro.txt"


def test_upload_rejects_unsupported_file_type(tmp_path: Path) -> None:
    client = create_client(tmp_path)

    response = client.post(
        "/upload/document",
        data={"knowledge_base_id": "kb-homework"},
        files={"file": ("script.exe", b"binary", "application/octet-stream")},
    )

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_chat_returns_404_when_kb_has_no_processed_chunks(tmp_path: Path) -> None:
    client = create_client(tmp_path)

    response = client.post(
        "/chat",
        json={"knowledge_base_id": "empty-kb", "question": "hello", "top_k": 1},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "No processed knowledge found for this knowledge base"