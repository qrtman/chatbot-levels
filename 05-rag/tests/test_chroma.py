"""Offline tests for ChromaDB indexing and synchronization."""

import sys
import tempfile
import unittest

from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

import chromadb

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1]),
)

from ingest import prepare_chunks, sync_collection


class ChromaTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)

        self.client = chromadb.EphemeralClient()

        self.collection = self.client.create_collection(
            name=f"test_{uuid4().hex[:16]}",
            embedding_function=None,
            configuration={
                "hnsw": {"space": "cosine"}
            },
        )

        self.embedder = Mock()
        self.embedder.embed_documents.side_effect = (
            lambda texts: [
                [float(len(text)), 1.0, 0.25]
                for text in texts
            ]
        )

    def tearDown(self):
        self.temp.cleanup()

    def write_document(self, name, text):
        (self.directory / name).write_text(
            text,
            encoding="utf-8",
        )

    def get_records(self):
        return prepare_chunks(self.directory)

    def test_indexes_new_chunks(self):
        self.write_document(
            "shipping.md",
            "Standard shipping takes 3-5 business days.",
        )

        records = self.get_records()

        result = sync_collection(
            self.collection,
            self.embedder,
            records,
        )

        self.assertEqual(
            self.collection.count(),
            len(records),
        )
        self.assertEqual(result["written"], len(records))

        stored = self.collection.get(
            include=["documents", "metadatas"]
        )

        self.assertEqual(
            stored["documents"][0],
            records[0]["text"],
        )
        self.assertEqual(
            stored["metadatas"][0]["source"],
            "shipping.md",
        )

    def test_second_run_skips_unchanged_chunks(self):
        self.write_document("shipping.md", "Shipping policy")

        records = self.get_records()

        sync_collection(
            self.collection, self.embedder, records
        )

        self.embedder.embed_documents.reset_mock()

        result = sync_collection(
            self.collection, self.embedder, records
        )

        self.assertEqual(result["written"], 0)
        self.assertEqual(result["unchanged"], len(records))

        self.embedder.embed_documents.assert_not_called()

    def test_modified_document_replaces_old_chunks(self):
        self.write_document("shipping.md", "Old shipping policy")

        old_records = self.get_records()

        sync_collection(
            self.collection, self.embedder, old_records
        )

        self.write_document(
            "shipping.md",
            "Updated shipping policy",
        )

        new_records = self.get_records()

        result = sync_collection(
            self.collection, self.embedder, new_records
        )

        stored_ids = set(
            self.collection.get(include=[])["ids"]
        )

        self.assertEqual(
            stored_ids,
            {record["id"] for record in new_records},
        )

        self.assertEqual(
            result["deleted"],
            len(old_records),
        )

    def test_deleted_document_is_removed(self):
        self.write_document("shipping.md", "Shipping policy")
        self.write_document("returns.md", "Returns policy")

        sync_collection(
            self.collection,
            self.embedder,
            self.get_records(),
        )

        (self.directory / "returns.md").unlink()

        result = sync_collection(
            self.collection,
            self.embedder,
            self.get_records(),
        )

        self.assertEqual(result["deleted"], 1)
        self.assertEqual(self.collection.count(), 1)

    def test_empty_records_rejected(self):
        with self.assertRaises(ValueError):
            sync_collection(
                self.collection,
                self.embedder,
                [],
            )

    def test_invalid_batch_size_rejected(self):
        self.write_document("shipping.md", "Shipping policy")

        with self.assertRaises(ValueError):
            sync_collection(
                self.collection,
                self.embedder,
                self.get_records(),
                batch_size=0,
            )

    def test_embedding_failure_preserves_collection(self):
        self.write_document("shipping.md", "Shipping policy")

        self.embedder.embed_documents.side_effect = (
            RuntimeError("API unavailable")
        )

        with self.assertRaises(RuntimeError):
            sync_collection(
                self.collection,
                self.embedder,
                self.get_records(),
            )

        self.assertEqual(self.collection.count(), 0)

    def test_duplicate_chunk_ids_rejected(self):
        self.write_document("shipping.md", "Shipping policy")

        record = self.get_records()[0]

        with self.assertRaises(ValueError):
            sync_collection(
                self.collection,
                self.embedder,
                [record, record],
            )

    def test_repairs_modified_text_with_same_id(self):
        self.write_document("shipping.md", "Shipping policy")

        record = self.get_records()[0]

        self.collection.upsert(
            ids=[record["id"]],
            embeddings=[[1.0, 0.0, 0.25]],
            documents=["Incorrect stored text"],
            metadatas=[record["metadata"]],
        )

        result = sync_collection(
            self.collection,
            self.embedder,
            [record],
        )

        stored = self.collection.get(
            ids=[record["id"]],
            include=["documents"],
        )

        self.assertEqual(result["written"], 1)
        self.assertEqual(
            stored["documents"][0],
            record["text"],
        )

    def test_persistent_storage(self):
        self.write_document("shipping.md", "Shipping policy")

        records = self.get_records()

        with tempfile.TemporaryDirectory() as db_dir:
            name = f"test_{uuid4().hex[:16]}"

            first_client = chromadb.PersistentClient(
                path=db_dir
            )

            collection = first_client.get_or_create_collection(
                name=name,
                embedding_function=None,
            )

            sync_collection(
                collection,
                self.embedder,
                records,
            )

            second_client = chromadb.PersistentClient(
                path=db_dir
            )

            reopened = second_client.get_collection(
                name=name,
                embedding_function=None,
            )

            self.assertEqual(
                reopened.count(),
                len(records),
            )


if __name__ == "__main__":
    unittest.main()