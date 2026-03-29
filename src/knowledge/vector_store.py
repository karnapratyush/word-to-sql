"""Vector Store — ChromaDB-backed semantic search for schema + few-shot examples.

This is the brain of the knowledge layer. It:
1. Embeds all table descriptions, relationship docs, and few-shot examples
   into ChromaDB using the default embedding model (all-MiniLM-L6-v2)
2. At query time, finds the most relevant tables and examples for the
   user's question via cosine similarity search
3. Builds a focused schema context (only relevant tables, not all 7+)

Why this matters: Sending the full schema to the LLM wastes tokens and
can confuse the model. By retrieving only the 3-5 most relevant tables,
we get more accurate SQL generation with lower token costs.

The store is initialized once at startup and kept as a module-level
singleton. ChromaDB runs in-memory (ephemeral mode) by default.

Usage:
    from src.knowledge.vector_store import get_knowledge_store
    store = get_knowledge_store()
    context = store.retrieve_context("What is the average freight cost by carrier?")
    # context.schema_text = focused schema for relevant tables
    # context.few_shot_examples = 3 similar question-to-SQL pairs
"""

import os
from dataclasses import dataclass, field
from functools import lru_cache

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.knowledge.semantic_layer import TABLE_DESCRIPTIONS, RELATIONSHIP_DESCRIPTIONS
from src.knowledge.few_shot_examples import FEW_SHOT_EXAMPLES


# ── Data Classes ─────────────────────────────────────────────────────

@dataclass
class RetrievedContext:
    """Context retrieved from the vector store for a given query.

    This is the output of retrieve_context() and contains everything
    the SQL generator needs to build an accurate prompt.

    Attributes:
        schema_text: Focused schema description (only relevant tables).
        few_shot_text: Similar question-to-SQL pairs formatted for the prompt.
        relevant_tables: Table names that were retrieved by similarity search.
        few_shot_count: Number of few-shot examples included.
        all_table_names: All tables in the database (for reference/fallback).
    """
    schema_text: str
    few_shot_text: str
    relevant_tables: list[str]
    few_shot_count: int
    all_table_names: list[str] = field(default_factory=list)


# ── Knowledge Store Class ────────────────────────────────────────────

class KnowledgeStore:
    """ChromaDB-backed semantic search for schema and few-shot examples.

    Maintains two separate ChromaDB collections:
    - "schema": Table descriptions, column info, relationships, sample values.
      Documents are the natural language descriptions; metadata carries
      structured info (columns, joins, key_values) for prompt building.
    - "few_shot": Question-to-SQL pairs. Documents are the questions;
      metadata carries the corresponding SQL and table list.

    Both collections use cosine similarity for semantic search.
    """

    def __init__(self, persist_directory: str | None = None):
        """Initialize ChromaDB client and populate collections.

        Args:
            persist_directory: Optional path to persist ChromaDB data to disk.
                             If None, uses in-memory (ephemeral) store that
                             is lost on process restart.
        """
        # Choose between persistent (disk) and ephemeral (memory) storage
        if persist_directory:
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.EphemeralClient()

        # Create or get the two collections with cosine similarity metric
        self._schema_collection = self._client.get_or_create_collection(
            name="schema",
            metadata={"hnsw:space": "cosine"},  # Cosine similarity for text embeddings
        )
        self._few_shot_collection = self._client.get_or_create_collection(
            name="few_shot",
            metadata={"hnsw:space": "cosine"},
        )

        # Populate collections if they are empty (first initialization)
        self._populate_schema()
        self._populate_few_shot()

    # ── Collection Population ────────────────────────────────────────

    def _populate_schema(self):
        """Load table descriptions and relationships into the schema collection.

        Only populates if the collection is empty (idempotent).
        Each document's "text" field is embedded for similarity search,
        while "metadata" carries structured info for prompt building.
        """
        if self._schema_collection.count() > 0:
            return  # Already populated — skip to avoid duplicates

        # Combine table descriptions and relationship descriptions
        all_docs = TABLE_DESCRIPTIONS + RELATIONSHIP_DESCRIPTIONS

        self._schema_collection.add(
            ids=[doc["id"] for doc in all_docs],
            documents=[doc["text"] for doc in all_docs],      # Embedded for search
            metadatas=[doc["metadata"] for doc in all_docs],   # Structured data for prompts
        )

    def _populate_few_shot(self):
        """Load few-shot examples into the few_shot collection.

        The question text is embedded so similar user queries retrieve
        relevant SQL examples. The actual SQL is stored in metadata.
        """
        if self._few_shot_collection.count() > 0:
            return  # Already populated — skip to avoid duplicates

        self._few_shot_collection.add(
            ids=[ex["id"] for ex in FEW_SHOT_EXAMPLES],
            documents=[ex["question"] for ex in FEW_SHOT_EXAMPLES],  # Embedded for search
            metadatas=[{
                "sql": ex["sql"],
                "tables": ",".join(ex["tables"]),  # Stored as comma-separated string
                "complexity": ex["complexity"],
            } for ex in FEW_SHOT_EXAMPLES],
        )

    # ── Context Retrieval ────────────────────────────────────────────

    def retrieve_context(
        self,
        query: str,
        n_tables: int = 5,
        n_examples: int = 3,
    ) -> RetrievedContext:
        """Retrieve relevant schema context and few-shot examples for a query.

        This is the main entry point called by the SQL generator. It performs
        two parallel semantic searches and builds a focused context.

        Args:
            query: The user's natural language question.
            n_tables: Max number of table/relationship docs to retrieve.
            n_examples: Max number of few-shot examples to retrieve.

        Returns:
            RetrievedContext with focused schema text and few-shot examples.
        """
        # Step 1: Retrieve the most relevant schema documents (tables + relationships)
        schema_results = self._schema_collection.query(
            query_texts=[query],
            n_results=n_tables,
        )

        # Step 2: Retrieve the most similar few-shot examples
        few_shot_results = self._few_shot_collection.query(
            query_texts=[query],
            n_results=n_examples,
        )

        # Step 3: Build human-readable schema text from retrieved docs
        schema_text = self._build_schema_text(schema_results)
        relevant_tables = self._extract_table_names(schema_results)

        # Step 4: Always append extracted_documents context with live field names.
        # This is always included regardless of similarity score because:
        # - Users can ask about extracted docs in any phrasing
        # - The field list changes at runtime (new uploads add new fields)
        # - The in-memory set (_known_fields) is the source of truth for what
        #   columns actually exist, not the hardcoded expected_fields in prompts.yaml
        extracted_docs_context = self._build_extracted_docs_context()
        if extracted_docs_context:
            schema_text = schema_text + "\n\n" + extracted_docs_context
            if "extracted_documents" not in relevant_tables:
                relevant_tables.append("extracted_documents")
                relevant_tables.sort()

        # Step 5: Build few-shot text for the prompt
        few_shot_text = self._build_few_shot_text(few_shot_results)
        few_shot_count = len(few_shot_results["ids"][0]) if few_shot_results["ids"] else 0

        # Step 6: Get all table names for reference
        all_tables = [
            doc["metadata"]["table"]
            for doc in TABLE_DESCRIPTIONS
        ]

        return RetrievedContext(
            schema_text=schema_text,
            few_shot_text=few_shot_text,
            relevant_tables=relevant_tables,
            few_shot_count=few_shot_count,
            all_table_names=all_tables,
        )

    def _build_extracted_docs_context(self) -> str:
        """Build extracted_documents context using the live known-fields set.

        This is ALWAYS included in the schema context (not dependent on
        similarity search). The known-fields set is the source of truth
        because it reflects the actual fields in the database, which change
        as users upload new document types.

        Returns:
            Multi-line string describing extracted_documents and its JSON fields,
            or empty string if no documents have been uploaded yet.
        """
        try:
            from src.vision.storage import get_known_fields
            fields = get_known_fields()
        except Exception:
            fields = set()

        if not fields:
            return ""

        lines = [
            "ALWAYS AVAILABLE — EXTRACTED DOCUMENTS:",
            "TABLE: extracted_documents",
            "  Regular columns: document_id, document_type, file_name, overall_confidence,",
            "    review_status, linked_shipment_id, upload_timestamp, extraction_model",
            "  ",
            "  JSON columns (query with json_extract(extracted_fields, '$.field_name')):",
            "  The following fields have been extracted from uploaded documents.",
            "  This list is the SOURCE OF TRUTH — it reflects actual data in the database",
            "  and updates automatically when new document types are uploaded.",
            "  ",
        ]

        for field_name in sorted(fields):
            lines.append(f"    - {field_name}")

        lines.extend([
            "  ",
            "  JOIN to shipments: extracted_documents.linked_shipment_id = shipments.shipment_id",
            "  Filter by type: WHERE document_type = 'invoice' | 'bill_of_lading' | 'packing_list' | 'customs_declaration'",
        ])

        return "\n".join(lines)

    # ── Text Builders ────────────────────────────────────────────────

    def _build_schema_text(self, results: dict) -> str:
        """Build human-readable schema text from ChromaDB query results.

        Formats table descriptions with columns, key values, joins, and
        examples. Relationship descriptions include join keys and usage hints.

        Args:
            results: Raw ChromaDB query results dict.

        Returns:
            Multi-line string suitable for injection into an LLM prompt.
        """
        if not results["ids"] or not results["ids"][0]:
            return "(no relevant schema found)"

        parts = []
        seen_tables = set()  # Avoid duplicate table entries

        for i, doc_id in enumerate(results["ids"][0]):
            doc_text = results["documents"][0][i]
            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i] if results.get("distances") else None

            # Format table descriptions with full metadata
            if doc_id.startswith("table_"):
                table_name = metadata.get("table", "")
                if table_name in seen_tables:
                    continue  # Skip duplicate table entries
                seen_tables.add(table_name)

                columns = metadata.get("columns", "")
                key_values = metadata.get("key_values", "")
                joins = metadata.get("joins", "")
                example = metadata.get("example", "")
                row_count = metadata.get("row_count", "?")

                part = f"TABLE: {table_name} ({row_count:,} rows)\n"
                part += f"  Columns: {columns}\n"
                if key_values:
                    part += f"  Values: {key_values}\n"
                if joins:
                    part += f"  Joins: {joins}\n"
                if example:
                    part += f"  Example: {example}\n"
                parts.append(part)

            # Format relationship descriptions with join info
            elif doc_id.startswith("rel_"):
                from_t = metadata.get("from_table", "")
                to_t = metadata.get("to_table", "")
                join_key = metadata.get("join_key", "")
                parts.append(f"RELATIONSHIP: {from_t} → {to_t}\n  JOIN: {join_key}\n  {doc_text}\n")

        return "\n".join(parts)

    def _build_few_shot_text(self, results: dict) -> str:
        """Build few-shot examples text from ChromaDB query results.

        Formats each example as a numbered block with question, SQL, and
        tables used, suitable for injection into the SQL generator prompt.

        Args:
            results: Raw ChromaDB query results dict.

        Returns:
            Multi-line string with numbered examples, or empty string if none.
        """
        if not results["ids"] or not results["ids"][0]:
            return ""

        parts = []
        for i, doc_id in enumerate(results["ids"][0]):
            question = results["documents"][0][i]
            metadata = results["metadatas"][0][i]
            sql = metadata.get("sql", "")
            tables = metadata.get("tables", "")

            parts.append(
                f"Example {i+1}:\n"
                f"  Question: {question}\n"
                f"  SQL: {sql}\n"
                f"  Tables used: {tables}"
            )

        return "\n\n".join(parts)

    def _extract_table_names(self, results: dict) -> list[str]:
        """Extract unique table names from schema query results.

        Collects table names from both table descriptions (metadata.table)
        and relationship descriptions (metadata.from_table, metadata.to_table).

        Args:
            results: Raw ChromaDB query results dict.

        Returns:
            Sorted list of unique table name strings.
        """
        tables = set()
        if results["metadatas"] and results["metadatas"][0]:
            for metadata in results["metadatas"][0]:
                if "table" in metadata:
                    tables.add(metadata["table"])
                if "from_table" in metadata:
                    tables.add(metadata["from_table"])
                if "to_table" in metadata:
                    tables.add(metadata["to_table"])
        return sorted(tables)


# ── Singleton Management ─────────────────────────────────────────────
# The KnowledgeStore is expensive to create (embeddings computation),
# so we maintain a single instance for the lifetime of the process.

_store_instance: KnowledgeStore | None = None


def get_knowledge_store() -> KnowledgeStore:
    """Return the shared KnowledgeStore singleton (created on first call).

    Returns:
        The global KnowledgeStore instance.
    """
    global _store_instance
    if _store_instance is None:
        _store_instance = KnowledgeStore()
    return _store_instance


def reset_knowledge_store():
    """Reset the singleton to None (used in tests to force re-initialization).

    This allows tests to start with a fresh KnowledgeStore without
    interference from previous test runs.
    """
    global _store_instance
    _store_instance = None
