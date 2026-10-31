"""Load and chunk local Markdown documents for the RAG chatbot."""

from hashlib import sha256
from pathlib import Path


CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
KNOWLEDGE_BASE = Path(__file__).resolve().parent / "knowledge_base"


def load_documents(directory: Path) -> list[dict]:
    """Read nonempty .md files in a stable order."""
    directory = Path(directory)

    if not directory.is_dir():
        raise FileNotFoundError(
            f"Knowledge-base directory not found: {directory}"
        )

    paths = sorted(directory.glob("*.md"))

    if not paths:
        raise ValueError(f"No Markdown documents found in: {directory}")

    documents = []

    for path in paths:
        text = path.read_text(encoding="utf-8").strip()

        if not text:
            raise ValueError(f"Empty Markdown document: {path.name}")

        documents.append({
            "source": path.name,
            "text": text,
        })

    return documents


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into deterministic character chunks with overlap."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if not 0 <= overlap < chunk_size:
        raise ValueError(
            "overlap must be nonnegative and smaller than chunk_size"
        )

    text = text.strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        chunks.append(text[start:end])

        if end == len(text):
            break

        start = end - overlap

    return chunks


def prepare_chunks(
    directory: Path,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """Create chunk records with stable source and version metadata."""

    records = []

    for document in load_documents(directory):
        source = document["source"]
        text = document["text"]

        document_hash = sha256(text.encode("utf-8")).hexdigest()

        for index, chunk in enumerate(
            chunk_text(text, chunk_size, overlap)
        ):
            id_input = f"{source}:{index}:{document_hash}"

            chunk_id = sha256(
                id_input.encode("utf-8")
            ).hexdigest()[:24]

            records.append({
                "id": chunk_id,
                "text": chunk,
                "metadata": {
                    "source": source,
                    "chunk_index": index,
                    "document_hash": document_hash,
                },
            })

    return records


if __name__ == "__main__":
    documents = load_documents(KNOWLEDGE_BASE)
    chunks = prepare_chunks(KNOWLEDGE_BASE)

    print(f"Loaded {len(documents)} documents.")
    print(f"Prepared {len(chunks)} chunks.")

    for record in chunks:
        print(
            f"{record['id']}  "
            f"{record['metadata']['source']}  "
            f"#{record['metadata']['chunk_index']}"
        )