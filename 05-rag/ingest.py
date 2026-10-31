"""Load and chunk local Markdown documents for the RAG chatbot."""

from hashlib import sha256
from pathlib import Path
import chromadb
from embeddings import EMBEDDING_MODEL, GeminiEmbedder

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
KNOWLEDGE_BASE = Path(__file__).resolve().parent / "knowledge_base"

CHROMA_PATH = Path(__file__).resolve().parent / "chroma_db"
COLLECTION_NAME = "demo_store_support"
INDEX_BATCH_SIZE = 8

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

def open_collection(database_path=CHROMA_PATH):
    """Open the persistent ChromaDB collection."""

    client = chromadb.PersistentClient(
        path=str(database_path)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
        configuration={
            "hnsw": {
                "space": "cosine",
            }
        },
        metadata={
            "embedding_model": EMBEDDING_MODEL,
        },
    )

    stored_model = (collection.metadata or {}).get(
        "embedding_model"
    )

    if stored_model != EMBEDDING_MODEL:
        raise ValueError(
            "Collection embedding model does not match "
            "the configured Gemini embedding model"
        )

    return collection

def sync_collection(
    collection,
    embedder,
    records,
    batch_size=INDEX_BATCH_SIZE,
):
    """Synchronize prepared chunks with a ChromaDB collection."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    if not records:
        raise ValueError(
            "Refusing to synchronize an empty knowledge base"
        )

    desired_ids = [record["id"] for record in records]

    if len(set(desired_ids)) != len(desired_ids):
        raise ValueError("Duplicate chunk IDs detected")

    # Read existing IDs and text from ChromaDB.
    existing = collection.get(include=["documents"])

    existing_ids = existing["ids"]
    existing_documents = existing["documents"] or []

    if len(existing_ids) != len(existing_documents):
        raise RuntimeError(
            "ChromaDB returned inconsistent document data"
        )

    stored_documents = dict(
        zip(existing_ids, existing_documents)
    )

    # Only embed new or modified chunks.
    pending = [
        record
        for record in records
        if record["id"] not in stored_documents
        or stored_documents[record["id"]] != record["text"]
    ]

    # Generate embeddings and write them in small batches.
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]

        texts = [record["text"] for record in batch]

        vectors = embedder.embed_documents(texts)

        if len(vectors) != len(batch):
            raise RuntimeError(
                "Embedding count does not match batch size"
            )

        collection.upsert(
            ids=[record["id"] for record in batch],
            embeddings=vectors,
            documents=texts,
            metadatas=[
                record["metadata"]
                for record in batch
            ],
        )

    # Remove records that are no longer in the knowledge base.
    obsolete_ids = sorted(
        set(existing_ids) - set(desired_ids)
    )

    # Delete only after new records have been indexed.
    for start in range(0, len(obsolete_ids), batch_size):
        batch = obsolete_ids[start:start + batch_size]

        collection.delete(ids=batch)

    return {
        "prepared": len(records),
        "written": len(pending),
        "unchanged": len(records) - len(pending),
        "deleted": len(obsolete_ids),
        "stored": collection.count(),
    }

if __name__ == "__main__":
    documents = load_documents(KNOWLEDGE_BASE)
    records = prepare_chunks(KNOWLEDGE_BASE)

    collection = open_collection()
    embedder = GeminiEmbedder()

    result = sync_collection(
        collection=collection,
        embedder=embedder,
        records=records,
    )

    print(f"Loaded documents: {len(documents)}")
    print(f"Prepared chunks: {result['prepared']}")
    print(f"Written chunks: {result['written']}")
    print(f"Unchanged chunks: {result['unchanged']}")
    print(f"Deleted chunks: {result['deleted']}")
    print(f"Stored chunks: {result['stored']}")