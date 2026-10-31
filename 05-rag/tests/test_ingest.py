"""Offline unit tests for the document preparation stage."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingest import (
    chunk_sections,
    chunk_text,
    load_documents,
    prepare_chunks,
)


class IngestTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write_document(self, filename, text):
        (self.directory / filename).write_text(
            text,
            encoding="utf-8",
        )

    def test_loads_markdown_files_in_sorted_order(self):
        self.write_document("z.md", "Last")
        self.write_document("a.md", "First")
        self.write_document("ignore.txt", "Ignored")

        documents = load_documents(self.directory)

        self.assertEqual(
            [d["source"] for d in documents],
            ["a.md", "z.md"],
        )

    def test_missing_directory_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_documents(self.directory / "missing")

    def test_empty_document_raises(self):
        self.write_document("blank.md", "  \n")

        with self.assertRaises(ValueError):
            load_documents(self.directory)

    def test_no_markdown_files_raises(self):
        with self.assertRaises(ValueError):
            load_documents(self.directory)

    def test_chunks_have_expected_overlap(self):
        chunks = chunk_text(
            "abcdefghij",
            chunk_size=6,
            overlap=2,
        )

        self.assertEqual(chunks, ["abcdef", "efghij"])

    def test_sections_start_at_level_two_headings(self):
        chunks = chunk_sections(
            "# Title\n\nIntro\n\n## First\n\nOne\n\n### Sub\n\nTwo\n\n## Second\n\nThree"
        )

        self.assertEqual(
            chunks,
            [
                "# Title\n\nIntro",
                "## First\n\nOne\n\n### Sub\n\nTwo",
                "## Second\n\nThree",
            ],
        )

    def test_only_long_sections_are_split(self):
        chunks = chunk_sections(
            "## A\nab\n## B\nabcdefghij",
            chunk_size=8,
            overlap=2,
        )

        self.assertEqual(
            chunks,
            ["## A\nab", "## B\nabc", "bcdefghi", "hij"],
        )

    def test_invalid_chunk_settings_raise(self):
        with self.assertRaises(ValueError):
            chunk_text("sample", chunk_size=0, overlap=0)

        with self.assertRaises(ValueError):
            chunk_text("sample", chunk_size=6, overlap=6)

        with self.assertRaises(ValueError):
            chunk_text("sample", chunk_size=6, overlap=-1)

    def test_empty_text_returns_no_chunks(self):
        self.assertEqual(chunk_text("   "), [])

    def test_chunk_ids_and_metadata_are_deterministic(self):
        self.write_document("shipping.md", "abcdefghij")

        first = prepare_chunks(
            self.directory,
            chunk_size=6,
            overlap=2,
        )

        second = prepare_chunks(
            self.directory,
            chunk_size=6,
            overlap=2,
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)

        self.assertEqual(
            first[0]["metadata"]["source"],
            "shipping.md",
        )

        self.assertEqual(
            first[1]["metadata"]["chunk_index"],
            1,
        )

        self.assertEqual(
            len(first[0]["metadata"]["document_hash"]),
            64,
        )

    def test_document_change_changes_ids(self):
        self.write_document("shipping.md", "first version")
        first_id = prepare_chunks(self.directory)[0]["id"]

        self.write_document("shipping.md", "second version")
        second_id = prepare_chunks(self.directory)[0]["id"]

        self.assertNotEqual(first_id, second_id)


if __name__ == "__main__":
    unittest.main()