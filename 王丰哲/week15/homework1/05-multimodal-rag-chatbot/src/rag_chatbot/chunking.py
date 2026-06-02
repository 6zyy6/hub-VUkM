from __future__ import annotations

import hashlib
import re
from typing import Any

from rag_chatbot.models import Chunk

HEADING_RE = re.compile(r"^(#{1,6}\s+.+)$")
IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


def chunk_markdown(
    *,
    title: str,
    source: str,
    content: str,
    max_chars: int = 900,
    overlap: int = 120,
    metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    """Split markdown into heading-aware chunks and retain image references."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")

    document_id = _stable_id("doc", source, title, content)
    sections = _split_sections(content)
    chunks: list[Chunk] = []

    for heading, section_text in sections:
        for part_index, part in enumerate(_split_with_overlap(section_text, max_chars, overlap)):
            text = part.strip()
            if not text:
                continue
            chunk_id = _stable_id("chk", document_id, heading, str(part_index), text)
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    title=title,
                    source=source,
                    heading=heading,
                    text=text,
                    images=_extract_images(text),
                    metadata=metadata or {},
                )
            )

    return chunks


def _split_sections(content: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, list[str]]] = []
    current_heading = "Document"
    current_lines: list[str] = []

    for line in content.splitlines():
        heading_match = HEADING_RE.match(line.strip())
        if heading_match:
            if current_lines:
                sections.append((current_heading, current_lines))
            current_heading = heading_match.group(1)
            current_lines = [line]
            continue
        current_lines.append(line)

    if current_lines:
        sections.append((current_heading, current_lines))

    return [(heading, "\n".join(lines).strip()) for heading, lines in sections if "\n".join(lines).strip()]


def _split_with_overlap(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            breakpoint = max(text.rfind("\n", start, end), text.rfind(". ", start, end), text.rfind("。", start, end))
            if breakpoint > start + max_chars // 2:
                end = breakpoint + 1
        parts.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)

    return parts


def _extract_images(text: str) -> list[dict[str, str]]:
    return [{"alt": alt, "path": path} for alt, path in IMAGE_RE.findall(text)]


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"
