"""Funciones centrales para un tutor RAG local basado exclusivamente en un PDF."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import re
from typing import Iterable

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class Chunk:
    """Fragmento de texto extraído de una página del PDF."""

    source_id: str
    pdf_page: int
    text: str


def _split_words(text: str, size: int, overlap: int) -> Iterable[str]:
    words = re.findall(r"\S+", text)
    if not words:
        return
    step = max(1, size - overlap)
    for start in range(0, len(words), step):
        piece = " ".join(words[start : start + size]).strip()
        if piece:
            yield piece
        if start + size >= len(words):
            break


def extract_chunks(pdf_bytes: bytes, words_per_chunk: int = 180, overlap: int = 35) -> list[Chunk]:
    """Extrae texto página por página; la numeración citada es la página física del PDF."""
    if not pdf_bytes:
        raise ValueError("El archivo PDF está vacío.")
    reader = PdfReader(BytesIO(pdf_bytes))
    chunks: list[Chunk] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").replace("\x00", " ")
        text = re.sub(r"\s+", " ", text).strip()
        for part_number, part in enumerate(_split_words(text, words_per_chunk, overlap), start=1):
            chunks.append(Chunk(f"P{page_number}-F{part_number}", page_number, part))
    if not chunks:
        raise ValueError("No se pudo extraer texto. El PDF podría ser escaneado y requerir OCR.")
    return chunks


class KnowledgeBase:
    """Índice TF-IDF en memoria: la fuente nunca se envía completa al modelo."""

    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("No hay fragmentos indexables.")
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=180_000,
        )
        self.matrix = self.vectorizer.fit_transform([c.text for c in chunks])

    def search(self, query: str, top_k: int = 4) -> list[tuple[Chunk, float]]:
        query = query.strip()
        if not query:
            return []
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        ranked = scores.argsort()[::-1][: max(1, top_k)]
        return [(self.chunks[i], float(scores[i])) for i in ranked if scores[i] > 0]


def validated_citations(answer: str, allowed_ids: set[str]) -> str:
    """Neutraliza identificadores de cita inventados por el modelo."""
    pattern = re.compile(r"\[(S\d+)\]")

    def replace(match: re.Match[str]) -> str:
        citation = match.group(1)
        return match.group(0) if citation in allowed_ids else "[referencia no verificada]"

    return pattern.sub(replace, answer)


def make_evidence(chunks_and_scores: list[tuple[Chunk, float]]) -> tuple[str, dict[str, Chunk]]:
    """Asigna IDs breves de evidencia para facilitar citas verificables."""
    refs: dict[str, Chunk] = {}
    blocks: list[str] = []
    for index, (chunk, score) in enumerate(chunks_and_scores, start=1):
        source_id = f"S{index}"
        refs[source_id] = chunk
        blocks.append(
            f"[{source_id}] PDF p. {chunk.pdf_page} (fragmento {chunk.source_id}; similitud {score:.3f})\n{chunk.text}"
        )
    return "\n\n---\n\n".join(blocks), refs
