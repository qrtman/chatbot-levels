"""Run and score real RAG evaluations. Answer quality is manually reviewed."""

import argparse
import csv
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from bot import CLARIFICATION_ANSWER, RAGChatbot, UNSUPPORTED_ANSWER
from embeddings import GeminiEmbedder
from ingest import KNOWLEDGE_BASE, open_collection, prepare_chunks
from retriever import retrieve


BASE_DIR = Path(__file__).resolve().parent
QUESTIONS_PATH = BASE_DIR / "evaluation" / "questions.jsonl"
RESULTS_DIR = BASE_DIR / "evaluation" / "results"
SOURCE_PATTERN = re.compile(r"([A-Za-z0-9_-]+\.md) \(chunk \d+\)")

TARGETS = {
    "retrieval_recall_at_k": 0.90,
    "answer_correctness": 0.90,
    "source_attribution": 1.00,
    "unsupported_handling": 0.90,
    "clarification_handling": 1.00,
}


def load_questions(path=QUESTIONS_PATH):
    """Load and validate the evaluation dataset."""

    cases = []

    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue

            case = json.loads(line)

            required = (
                "id", "type", "question",
                "required_sources", "checklist",
            )

            if not all(key in case for key in required):
                raise ValueError(
                    f"Missing fields at line {line_number}"
                )

            if case["type"] not in {
                "answer", "abstain", "clarify"
            }:
                raise ValueError(
                    f"Unknown type at line {line_number}"
                )

            if not isinstance(case["required_sources"], list):
                raise ValueError(
                    f"Invalid sources at line {line_number}"
                )

            if (
                (case["type"] == "answer")
                != bool(case["required_sources"])
            ):
                raise ValueError(
                    f"Unexpected required_sources at line {line_number}"
                )

            cases.append(case)

    ids = [case["id"] for case in cases]

    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate evaluation case IDs")

    if not cases:
        raise ValueError("Evaluation dataset is empty")

    return cases


def verify_index(collection):
    """Verify ChromaDB matches current knowledge-base documents."""

    expected = {
        record["id"]
        for record in prepare_chunks(KNOWLEDGE_BASE)
    }

    actual = set(
        collection.get(include=[])["ids"]
    )

    if expected != actual:
        raise RuntimeError(
            f"Index is stale: {len(expected - actual)} missing "
            f"and {len(actual - expected)} obsolete chunks. "
            "Run python 05-rag/ingest.py first."
        )

    return len(actual)


def evaluate_case(case, collection, embedder):
    """Run a case through the real chatbot and capture retrieval."""

    # Create independent conversation memory for every case.
    bot = RAGChatbot(
        collection,
        embedder,
        embedder.client,
    )

    trace = {
        "query": None,
        "chunks": [],
    }

    answer = ""
    error = None

    def traced_retrieve(question, collection, embedder, top_k=3):
        chunks = retrieve(
            question,
            collection,
            embedder,
            top_k=top_k,
        )

        trace["query"] = question
        trace["chunks"] = chunks

        return chunks

    try:
        # Run previous turns for contextual questions.
        for prior_question in case.get("context", []):
            bot.chatbot(prior_question)

        # Test the reset behavior where requested.
        if case.get("reset_before_question", False):
            bot.reset_history()

        # Capture the real chunks used by response generation.
        with patch(
            "bot.retrieve",
            side_effect=traced_retrieve,
        ):
            answer = bot.chatbot(case["question"])

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    retrieved = [
        chunk["source"]
        for chunk in trace["chunks"]
    ]

    cited = SOURCE_PATTERN.findall(answer)

    required = set(case["required_sources"])

    if case["type"] == "abstain":
        behavior_ok = answer == UNSUPPORTED_ANSWER

    elif case["type"] == "clarify":
        behavior_ok = (
            answer == CLARIFICATION_ANSWER
            and trace["query"] is None
        )

    else:
        behavior_ok = None

    return {
        **case,
        "retrieval_query": trace["query"],
        "retrieved_sources": retrieved,
        "retrieved_chunks": [
            {
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
                "distance": chunk["distance"],
            }
            for chunk in trace["chunks"]
        ],
        "retrieval_hit_at_k": (
            required.issubset(set(retrieved))
            if case["type"] == "answer"
            else None
        ),
        "cited_sources": cited,
        "gold_sources_cited": (
            required.issubset(set(cited))
            if case["type"] == "answer"
            else None
        ),
        "answer": answer,
        "behavior_ok": behavior_ok,
        "error": error,
    }


def run(args):
    """Execute real evaluation questions and create review files."""

    cases = load_questions()

    if args.ids:
        requested = set(args.ids)

        unknown = requested - {
            case["id"] for case in cases
        }

        if unknown:
            raise ValueError(
                f"Unknown case IDs: {sorted(unknown)}"
            )

        cases = [
            case for case in cases
            if case["id"] in requested
        ]

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")

        cases = cases[:args.limit]

    output = Path(args.output)
    review_path = output.with_suffix(".review.csv")

    if output.exists() or review_path.exists():
        raise FileExistsError(
            "Results already exist; choose another --output path"
        )

    collection = open_collection()

    # Ensure the database reflects the current Markdown files.
    count = verify_index(collection)

    embedder = GeminiEmbedder()

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Evaluating {len(cases)} cases "
        f"against {count} indexed chunks"
    )

    results = []

    knowledge_hashes = {
        path.name: hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(KNOWLEDGE_BASE.glob("*.md"))
    }

    for index, case in enumerate(cases, 1):
        # Stay under per-minute API quotas on long runs.
        if index > 1:
            time.sleep(args.delay)

        result = evaluate_case(
            case,
            collection,
            embedder,
        )

        result["evaluated_at_utc"] = (
            datetime.now(timezone.utc).isoformat()
        )

        result["knowledge_base_sha256"] = knowledge_hashes

        results.append(result)

        print(
            f"{index}/{len(cases)} {case['id']}: "
            f"retrieval={result['retrieval_hit_at_k']} "
            f"error={result['error']}"
        )

    # Save machine-readable results.
    with output.open(
        "w",
        encoding="utf-8",
    ) as stream:
        for row in results:
            stream.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )

    # Create a CSV for human review.
    with review_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        fields = [
            "id",
            "question",
            "type",
            "checklist",
            "answer",
            "retrieved_sources",
            "cited_sources",
            "answer_correct_0_or_1",
            "source_attribution_correct_0_or_1",
            "reviewer_notes",
        ]

        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
        )

        writer.writeheader()

        for row in results:
            writer.writerow({
                "id": row["id"],
                "question": row["question"],
                "type": row["type"],
                "checklist": " | ".join(row["checklist"]),
                "answer": row["answer"],
                "retrieved_sources": " | ".join(
                    row["retrieved_sources"]
                ),
                "cited_sources": " | ".join(
                    row["cited_sources"]
                ),
                "answer_correct_0_or_1": "",
                "source_attribution_correct_0_or_1": "",
                "reviewer_notes": "",
            })

    print(f"Saved: {output}")
    print(f"Review manually: {review_path}")


def ratio(numerator, denominator):
    """Return a metric ratio or None for an empty denominator."""

    return (
        numerator / denominator
        if denominator
        else None
    )


def score(args):
    """Calculate automatic and manually reviewed metrics."""

    results_path = Path(args.results)

    review_path = results_path.with_suffix(
        ".review.csv"
    )

    results = [
        json.loads(line)
        for line in results_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    with review_path.open(
        newline="",
        encoding="utf-8",
    ) as stream:
        review = {
            row["id"]: row
            for row in csv.DictReader(stream)
        }

    answerable = [
        row for row in results
        if row["type"] == "answer"
    ]

    unsupported = [
        row for row in results
        if row["type"] == "abstain"
    ]

    ambiguous = [
        row for row in results
        if row["type"] == "clarify"
    ]

    # Require explicit human grades for answerable questions.
    for row in answerable:
        item = review.get(row["id"])

        fields = (
            "answer_correct_0_or_1",
            "source_attribution_correct_0_or_1",
        )

        if item is None or any(
            item[field] not in {"0", "1"}
            for field in fields
        ):
            raise ValueError(
                f"Invalid grades for {row['id']}. "
                "Correctness and attribution must both be 0 or 1."
            )

        # An error or fallback cannot count as a correct answer.
        if (
            row["error"]
            or row["answer"] in {
                UNSUPPORTED_ANSWER,
                CLARIFICATION_ANSWER,
            }
        ):
            if any(item[field] != "0" for field in fields):
                raise ValueError(
                    f"{row['id']} produced an error or fallback. "
                    "Correctness and attribution must both be 0."
                )

    metrics = {
        "retrieval_recall_at_k": ratio(
            sum(
                # Results saved at top_k=3 use the older key.
                row.get(
                    "retrieval_hit_at_k",
                    row.get("retrieval_hit_at_3"),
                ) is True
                for row in answerable
            ),
            len(answerable),
        ),
        "answer_correctness": ratio(
            sum(
                review[row["id"]]["answer_correct_0_or_1"] == "1"
                for row in answerable
            ),
            len(answerable),
        ),
        "source_attribution": ratio(
            sum(
                review[row["id"]][
                    "source_attribution_correct_0_or_1"
                ] == "1"
                for row in answerable
            ),
            len(answerable),
        ),
        "unsupported_handling": ratio(
            sum(
                row["behavior_ok"] is True
                for row in unsupported
            ),
            len(unsupported),
        ),
        "clarification_handling": ratio(
            sum(
                row["behavior_ok"] is True
                for row in ambiguous
            ),
            len(ambiguous),
        ),
    }

    total_expected = len(load_questions())

    is_full_run = len(results) == total_expected

    report = {
        "cases_evaluated": len(results),
        "cases_in_dataset": total_expected,
        "complete_dataset": is_full_run,
        "error_count": sum(
            bool(row["error"])
            for row in results
        ),
        "metrics": metrics,
        "targets": TARGETS,
    }

    report_path = results_path.with_suffix(
        ".report.json"
    )

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(
        f"Cases: {len(results)}/{total_expected} "
        f"| errors: {report['error_count']}"
    )

    for metric, value in metrics.items():
        if value is None:
            print(
                f"{metric}: N/A (no eligible cases)"
            )
            continue

        passed = value >= TARGETS[metric]

        verdict = (
            "PASS" if passed else "FAIL"
        )

        if not is_full_run:
            verdict = "sample only"

        print(
            f"{metric}: {value:.1%} "
            f"(target {TARGETS[metric]:.0%}) "
            f"{verdict}"
        )

    print(f"Saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate real Demo Store RAG quality"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    run_parser = subparsers.add_parser("run")

    run_parser.add_argument(
        "--limit",
        type=int,
    )

    run_parser.add_argument(
        "--ids",
        nargs="+",
    )

    run_parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Seconds to wait between cases",
    )

    run_parser.add_argument(
        "--output",
        required=True,
    )

    score_parser = subparsers.add_parser("score")

    score_parser.add_argument(
        "--results",
        required=True,
    )

    args = parser.parse_args()

    if args.command == "run":
        run(args)
    else:
        score(args)


if __name__ == "__main__":
    main()