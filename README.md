#   AI Logistics Assistant

An agentic AI system that converts natural language questions into SQL queries against a logistics database and extracts structured data from shipping documents using vision LLMs. Built as a proof of concept for  's logistics platform.

**What it does:**
- **Analytics:** Ask "Which carriers have the highest delay rate?" and get back the SQL, a data table, a natural language answer, and an auto-generated Plotly chart.
- **Document Extraction:** Upload a bill of lading PDF and get structured fields (BL number, consignee, vessel, ports) extracted with per-field confidence scores.
- **Cross-domain Queries:** After extracting documents, query them alongside shipment data: "Show me all extracted invoices linked to delayed shipments."

**Key numbers:** 7 tables, 100K+ rows, 94% SQL accuracy on 150 eval cases, 266 tests passing, 25 sample documents included.

---

## Table of Contents

1. [Setup Instructions](#setup-instructions)
2. [Architecture Overview](#architecture-overview)
3. [Sample Questions](#sample-questions)
4. [Model Evaluation Results](#model-evaluation-results)
5. [Database Schema](#database-schema)
6. [Demo Script](#demo-script)
7. [Known Limitations](#known-limitations)
8. [Production Considerations](#production-considerations)
9. [Tech Stack](#tech-stack)
10. [Project Structure](#project-structure)

---

## Setup Instructions

### Prerequisites

- Python 3.13+ (see `.python-version`)
- API keys for at least one LLM provider (OpenRouter recommended -- single key covers all models)

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd word_to_sql
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```env
OPENROUTER_API_KEY=your-openrouter-key-here    # Required (covers Qwen, Gemini, DeepSeek)
GROQ_API_KEY=your-groq-key-here                # Optional (fallback provider)
GOOGLE_API_KEY=your-google-gemini-key-here      # Optional (direct Gemini access)
LANGFUSE_PUBLIC_KEY=your-langfuse-public-key    # Optional (observability)
LANGFUSE_SECRET_KEY=your-langfuse-secret-key    # Optional (observability)
LANGFUSE_HOST=https://cloud.langfuse.com        # Optional (observability)
```

At minimum, you need `OPENROUTER_API_KEY`. The fallback chain will skip unavailable providers gracefully.

### 3. Initialize the database

```bash
python db/seed_data.py
```

This creates `db/logistics.db` with 7 tables and 100K+ rows of realistic logistics data (customers, carriers, shipments, charges, tracking events, invoices).

### 4. Generate sample documents

```bash
python db/sample_documents.py
```

This generates 25 sample PDFs and images (invoices, bills of lading, packing lists, customs declarations) in `db/samples/` for testing the extraction pipeline.

### 5. Start the application

**Option A: Single command (recommended)**

```bash
bash scripts/start_dev.sh
```

This starts both servers:
- FastAPI API on `http://localhost:8000`
- Streamlit UI on `http://localhost:8501`

**Option B: Start separately (two terminals)**

```bash
# Terminal 1: API server
python run_api.py --port 8000

# Terminal 2: Streamlit UI
streamlit run app/Home.py --server.port 8501 --server.headless true
```

### 6. Run tests

```bash
pytest                    # Run all 266 tests
pytest -x                 # Stop on first failure
pytest tests/test_analytics.py  # Run specific test file
```

### 7. Run model evaluations (optional)

```bash
python eval/run_llm_eval.py         # Benchmark LLMs on 150 test cases
python eval/run_embedding_eval.py   # Benchmark embedding models
```

Results are written to `eval/results/`.

---

## Architecture Overview

### System Diagram

```
                         +-------------------+
                         |   Streamlit UI    |
                         |  (app/Home.py +   |
                         |   pages/*.py)     |
                         +--------+----------+
                                  | HTTP
                                  v
                         +-------------------+
                         |   FastAPI API     |
                         | (src/api/main.py) |
                         +--------+----------+
                                  |
                    +-------------+-------------+
                    |                           |
                    v                           v
          +-----------------+        +-------------------+
          | Analytics       |        | Vision            |
          | Service         |        | Service           |
          +-----------------+        +-------------------+
                    |                           |
                    v                           v
     +------------------------------+  +------------------+
     | Analytics Pipeline           |  | Vision Pipeline  |
     |                              |  |                  |
     | 1. Input Guardrails          |  | 1. PDF/Image     |
     | 2. Planner (intent classify) |  |    Rendering     |
     | 3. Knowledge Retrieval (RAG) |  | 2. Vision LLM   |
     | 4. SQL Generator (LLM)      |  |    Extraction    |
     | 5. Verifier (execute + check)|  | 3. Validation    |
     | 6. Answer Synthesizer (LLM)  |  | 4. Human Review  |
     | 7. Visualizer (Plotly)       |  | 5. Storage       |
     +------------------------------+  +------------------+
                    |                           |
                    v                           v
          +-----------------+        +-------------------+
          | Repository      |        | Document          |
          | (analytics_repo)|        | Repository        |
          +-----------------+        +-------------------+
                    |                           |
                    v                           v
          +-------------------------------------------+
          |        Database Engine Interface           |
          |   (src/repositories/engines/interface.py)  |
          +-------------------------------------------+
                              |
                              v
                    +-------------------+
                    |  SQLite Engine    |
                    |  (db/logistics.db)|
                    +-------------------+
```

### MVCR Pattern

The codebase follows a Model-View-Controller-Repository pattern:

| Layer | Location | Responsibility |
|---|---|---|
| **View** | `app/` (Streamlit) | User interface, chat, document upload |
| **Controller** | `src/api/routers/` (FastAPI) | HTTP handling, request/response mapping |
| **Service** | `src/services/` | Business orchestration, caching |
| **Domain** | `src/analytics/`, `src/vision/` | Core logic (planner, SQL gen, extraction) |
| **Repository** | `src/repositories/` | Data access, query execution |
| **Engine** | `src/repositories/engines/` | Database-specific implementation |

### Analytics Pipeline (Planner-Executor-Verifier)

The analytics pipeline (`src/analytics/agent.py`) processes each query through 7 sequential stages:

1. **Input Guardrails** (`src/guardrails/input_guards.py`): Blocks SQL injection (`DROP TABLE`, `UNION SELECT`), prompt injection (`ignore previous instructions`), empty/oversized inputs. Regex-based, no LLM cost.

2. **Planner** (`src/analytics/planner.py`): LLM classifies intent as `sql_query`, `clarification`, `unanswerable`, or `general`. Unanswerable questions get a helpful response with suggested alternatives instead of a hallucinated SQL query.

3. **Knowledge Retrieval** (`src/knowledge/vector_store.py`): ChromaDB semantic search retrieves the 5 most relevant table descriptions and 3 most similar few-shot SQL examples. Uses `all-MiniLM-L6-v2` embeddings (97.7% recall on our benchmark). This is the RAG component -- the LLM only sees relevant schema, not all 7 tables.

4. **SQL Generator** (`src/analytics/sql_generator.py`): LLM generates a SELECT query with focused schema context, few-shot examples, and conversation history for follow-ups. Retries up to 2 times with error feedback on failure.

5. **Verifier** (`src/analytics/verifier.py`): Output guardrails check for DML/DDL, multi-statement injection, and full table scans. If safe, executes the query read-only against SQLite and returns results.

6. **Answer Synthesizer**: LLM generates a natural language answer from the raw query results. Falls back to basic formatting if the LLM is unavailable.

7. **Visualizer** (`src/analytics/visualizer.py`): Rule-based chart suggestion (no LLM needed). Detects time-series keywords for line charts, distribution keywords for pie charts, otherwise bar charts. Creates Plotly figure dicts.

### LLM Fallback Chain

Each task has an ordered list of models in `config/models.yaml`. The factory (`src/models/llm_factory.py`) tries each model in sequence until one succeeds:

```
SQL Generation:  Qwen3 Coder -> Gemini 2.5 Flash -> Groq Llama 3.3 -> DeepSeek R1 -> Claude Sonnet
Classification:  Qwen3 Coder -> Groq Llama 8B -> Gemini 2.5 Flash
Vision:          Gemini 2.5 Flash -> Google Gemini direct -> GPT-4o-mini -> Groq Llama Vision
Answer:          Qwen3 Coder -> Groq Llama 3.3 -> Gemini 2.5 Flash
```

Free models are tried first. Paid models (`Claude Sonnet` at $3/MTok) are last resort only.

### Engine Plugin System

The repository layer is database-agnostic. `BaseRepository` (`src/repositories/base.py`) delegates to a `DatabaseEngine` interface (`src/repositories/engines/interface.py`). Currently only SQLite is implemented, but adding MySQL or PostgreSQL requires implementing one file with the same interface -- zero changes to business logic.

### Observability

LangFuse integration (`src/tracing.py`) traces every LLM call with inputs, outputs, latency, and token usage. Degrades gracefully: if LangFuse keys are not set, tracing is silently disabled with no code changes.

---

## Sample Questions

### Analytics Flow

**Simple counts and filters:**
```
How many shipments are currently delayed?
Show all shipments from China to the United States
How many customers are in the electronics industry?
```

**Aggregations and cost analysis:**
```
What is the average freight cost by carrier for ocean shipments?
What is the total invoice amount by payment status?
Compare ocean vs air shipping costs
```

**Multi-table joins:**
```
Which carriers have the highest delay rate?
Show me the top 5 routes by shipment volume
What is the average transit time by carrier for delivered shipments?
```

**Time-based analysis:**
```
Monthly shipment count trend for 2024
Show shipment volume by month for the last 6 months
What are the busiest shipping months?
```

**Rankings and comparisons:**
```
Top 10 customers by total freight spend
Which commodities have the most shipments?
Compare average delay days between gold and bronze tier customers
```

### Document Extraction Flow

Upload any of the 25 sample documents from `db/samples/`:
- `invoice_clear_001.pdf` -- a clear commercial invoice
- `bol_clear_001.pdf` -- a bill of lading
- `packing_clear_001.pdf` -- a packing list
- `customs_clear_001.pdf` -- a customs declaration

The system extracts structured fields with per-field confidence scores. Low-confidence fields (below 0.7) are flagged for human review.

### Cross-Domain Queries (Linkage)

After extracting and approving documents, query them alongside shipment data:
```
Show all extracted documents and their linked shipment status
How many invoices have been extracted with confidence above 0.9?
List extracted bills of lading that are linked to delayed shipments
What is the average extraction confidence by document type?
```

---

## Model Evaluation Results

All benchmarks were run against the actual logistics database (7 tables, 100K+ rows) using the 150-case evaluation suite in `eval/test_cases.yaml`.

### LLM Models -- Text-to-SQL

| Model | Provider | Cases | Composite Score | Avg Latency | Cost | Rate Limits |
|---|---|---|---|---|---|---|
| DeepSeek R1 | OpenRouter (free) | 40 | **100.0%** | 34,000ms | $0.00 | None |
| Gemini 2.5 Flash | OpenRouter | 150 | **94.9%** | 2,041ms | ~$0.00 | None |
| Qwen3 Coder | OpenRouter (free) | 150 | **94.1%** | 1,259ms | $0.00 | None |
| Groq Llama 3.3 70B | Groq (free) | 20* | ~98%* | 622ms | $0.00 | Severe |
| Gemini 2.5 Flash | Google direct | 20* | 100%* | 4,815ms | $0.00 | 20 req/day |

*Groq and Google direct could only be tested on 20 cases due to rate limits. Full 150-case runs failed catastrophically (4-29% scores from rate limit failures).

**Why Qwen3 Coder is the primary model:**
- 35% faster than Gemini (1.3s vs 2.0s) -- noticeable in interactive use
- Completely free via OpenRouter (zero credit consumption)
- Better injection handling (60% vs 0% for Gemini)
- 94.1% accuracy -- essentially tied with Gemini's 94.9%

### Category Breakdown

| Category | Qwen3 Coder | Gemini 2.5 Flash | DeepSeek R1 |
|---|---|---|---|
| simple_count (15 cases) | 100% | 100% | 100% |
| simple_filter (15) | 100% | 100% | 100% |
| aggregation (20) | 100% | 100% | 100% |
| two_table_join (20) | 100% | 100% | 100% |
| multi_join (15) | 93% | 100% | 100% |
| time_filter (15) | 100% | 100% | 100% |
| ranking (10) | 100% | 100% | 100% |
| comparison (10) | 100% | 100% | 100% |
| edge_case (10) | 100% | 100% | 100% |
| unanswerable (10) | 60% | 70% | 100% |
| injection (5) | 60% | 0% | 100% |
| ambiguous (5) | 40% | 0% | 100% |

All models score 93-100% on actual SQL generation. The accuracy gap comes from behavior detection (refusing unanswerable questions, handling injection attempts).

### Embedding Models -- Schema Retrieval

| Model | Recall@5 | MRR | Avg Time |
|---|---|---|---|
| all-mpnet-base-v2 | **98.5%** | 0.886 | 15.6ms |
| **all-MiniLM-L6-v2** (selected) | 97.7% | 0.890 | **3.7ms** |
| multi-qa-MiniLM-L6-cos-v1 | 97.7% | 0.890 | 3.8ms |
| paraphrase-MiniLM-L6-v2 | 97.7% | 0.887 | 3.6ms |

Selected `all-MiniLM-L6-v2` (ChromaDB's default): 97.7% recall with 4x faster inference than mpnet. The 0.8% recall gap is 1 test case out of 130.

Full evaluation methodology and raw data: `eval/MODEL_SELECTION.md`, `eval/results/`.

---

## Database Schema

7 tables with realistic logistics data. Foreign keys enforce referential integrity. 26 indexes cover common query patterns.

```
customers (10,000 rows)
  id, customer_code, company_name, industry, country, city, tier,
  contact_email, annual_volume_teu, created_at

carriers (30 rows)
  id, carrier_code, carrier_name, carrier_type, headquarters_country,
  reliability_score, avg_transit_days, cost_per_kg_usd, created_at

shipments (10,000 rows)
  id, shipment_id, customer_id*, carrier_id*, origin_port, origin_country,
  destination_port, destination_country, mode, status, booking_date,
  estimated_departure, actual_departure, estimated_arrival, actual_arrival,
  weight_kg, volume_cbm, container_type, container_count, commodity,
  incoterm, po_number, delay_days, delay_reason, priority, created_at

shipment_charges (29,000 rows)
  id, shipment_id*, charge_type, description, amount_usd, currency,
  exchange_rate, amount_original, invoice_reference, created_at

tracking_events (56,000 rows)
  id, shipment_id*, event_type, event_timestamp, location, location_country,
  description, created_by, is_milestone, created_at

invoices (10,000 rows)
  id, invoice_number, shipment_id*, customer_id*, carrier_id*,
  invoice_date, due_date, subtotal_usd, tax_usd, total_usd, currency,
  payment_status, payment_date, created_at

extracted_documents (dynamic -- grows as documents are uploaded)
  id, document_id, document_type, file_name, upload_timestamp,
  extraction_model, overall_confidence, review_status, reviewed_at,
  extracted_fields (JSON), confidence_scores (JSON), raw_text,
  linked_shipment_id, notes
```

`*` indicates a foreign key. The `extracted_documents` table uses JSON columns for flexible field storage since different document types have different schemas.

### Entity Relationship Summary

```
customers 1──N shipments 1──N shipment_charges
                   │
                   ├──N tracking_events
                   │
                   └──1 invoices
                   │
carriers  1──N shipments
                   │
extracted_documents ──? shipments  (optional linkage via linked_shipment_id)
```

---

## Demo Script

A walkthrough covering both core capabilities in under 2 minutes.

### Pre-demo Setup

Ensure both servers are running:

```bash
bash scripts/start_dev.sh
```

Open `http://localhost:8501` in a browser.

### Part 1: Analytics Flow (45 seconds)

1. **Home Page Health Check.** The landing page shows system status with row counts for all 7 tables. Confirm all tables are populated.

2. **Navigate to Analytics.** Click "Open Analytics Chat" or go to the Analytics page.

3. **Ask a simple question:**
   ```
   How many shipments are currently delayed?
   ```
   Point out: the system shows the generated SQL, a natural language answer, and the raw data table.

4. **Ask a cost analysis question:**
   ```
   What is the average freight cost by carrier for ocean shipments?
   ```
   Point out: a bar chart is auto-generated. The SQL uses a JOIN between shipments, shipment_charges, and carriers.

5. **Ask a follow-up:**
   ```
   Now show me only the top 3
   ```
   Point out: the system uses conversation history to understand "top 3" refers to the previous carrier cost query.

### Part 2: Document Extraction Flow (45 seconds)

6. **Navigate to Document Upload.** Click "Upload Documents" from the home page.

7. **Upload a sample invoice.** Select `db/samples/invoice_clear_001.pdf`. The vision LLM extracts fields: invoice number, vendor, buyer, amounts, dates, line items.

8. **Review confidence scores.** Point out per-field confidence indicators. Fields below 0.7 are flagged for review. Edit any field if needed.

9. **Approve and store.** Click "Approve" to persist the extraction to the `extracted_documents` table.

### Part 3: Linkage Query (30 seconds)

10. **Navigate to Query Documents.** Click "Query Documents" from the home page.

11. **Query extracted data alongside shipments:**
    ```
    Show all extracted documents and their review status
    ```

12. **Cross-domain linkage:**
    ```
    List extracted invoices linked to delayed shipments
    ```
    Point out: this query JOINs `extracted_documents` with `shipments`, connecting the extraction pipeline to the analytics pipeline.

### Wrap-Up

Summarize: natural language in, SQL + answers + charts out. Documents extracted with confidence scores. Both capabilities query the same database, enabling cross-domain analysis. All LLM calls are traced in LangFuse.

---

## Known Limitations

### Accuracy

- **Unanswerable question detection is the weakest point.** The primary model (Qwen3 Coder) scores 60% on unanswerable questions -- it tries to write SQL for questions like "What is the weather?" instead of refusing. DeepSeek R1 handles this perfectly (100%) but is too slow (34s) for interactive use. This is the single biggest accuracy gap.

- **Prompt injection detection scores 60%.** Sophisticated injection attempts can bypass the regex-based input guardrails and the LLM's own judgment. Production deployment would need a dedicated classifier.

- **Multi-join accuracy is 93%, not 100%.** Queries requiring 3+ table joins occasionally produce incorrect SQL on the first attempt, though the retry mechanism usually recovers.

### Performance

- **All processing is synchronous.** A single slow LLM call blocks the entire request. No async/concurrent query support.

- **ChromaDB runs in-memory.** The vector store is rebuilt on every process restart. Not an issue for 7 tables and 30 few-shot examples (takes ~1 second), but would not scale to hundreds of tables.

- **No query caching.** Identical questions hit the LLM every time. Frequently asked questions could be served from a cache.

### Document Extraction

- **Vision extraction accuracy varies by document quality.** Clear, well-formatted documents extract well. Scanned documents with handwriting, stamps, or poor image quality produce lower confidence scores.

- **Only 4 document types supported:** invoice, bill of lading, packing list, customs declaration. Other logistics documents (certificate of origin, letter of credit, etc.) fall to "unknown" type with generic extraction.

- **Document storage is local filesystem.** Uploaded files go to `db/uploads/`. No cloud storage integration.

### Infrastructure

- **SQLite is single-writer.** Concurrent write operations (e.g., two users approving documents simultaneously) would require connection serialization. Read concurrency is fine with WAL mode enabled.

- **No authentication or authorization.** Any user can query any data and upload documents.

- **Rate limits on free LLM tiers.** Groq allows approximately 5-10 requests per minute on free plans. OpenRouter free models have generous limits but are not guaranteed.

---

## Production Considerations

The following changes would be needed to move from POC to production:

**Database Portability.** The engine plugin system (`src/repositories/engines/interface.py`) is already designed for this. Implementing a PostgreSQL or MySQL engine requires one file with the same interface -- zero changes to repositories, services, or domain logic. SQLAlchemy could replace the manual engine implementation for broader driver support.

**Async Processing.** Replace synchronous LLM calls with async equivalents (LangChain supports both). Add a task queue (Celery/Redis) for document extraction, which can take 10-60 seconds.

**Query Caching.** Cache LLM responses keyed by (normalized_query, schema_version). A verified query store of approved SQL patterns would skip the LLM entirely for common questions.

**Row-Level Security.** Add tenant isolation so customers only see their own data. Currently, any query can access all 10K customers' shipments.

**Cloud Document Storage.** Replace local filesystem (`db/uploads/`) with S3 or GCS. The storage layer (`src/vision/storage.py`) is already a separate module for this reason.

**Document-Entity Relationship Table.** Currently, each extracted document has a single `linked_shipment_id` column linking it to one shipment. In production, a dedicated `extracted_document_links` table would support richer relationships:
- One document linked to multiple records (e.g., an invoice referencing 3 shipments)
- Links to multiple tables (shipments, invoices, carriers, customers) not just shipments
- Link confidence tracking (1.0 for exact shipment_ref match, 0.8 for fuzzy carrier name match)
- Link type auditing (matched on shipment_ref vs po_number vs invoice_number)
- This enables queries like "show the carrier and customer details for this BOL" without the user knowing the join path. The auto-linker (`_try_auto_link` in storage.py) already matches against shipments.shipment_id, shipments.po_number, and invoices.invoice_number — the relationship table would formalize and extend this pattern.

**Fine-Tuning for Unanswerable Detection.** The 60% unanswerable detection score is the biggest accuracy gap. Fine-tuning a small model on logistics-specific "answerable vs not" examples could push this to 95%+. Alternatively, a dedicated binary classifier as a pre-filter would be cheaper than fine-tuning.

**Monitoring.** LangFuse tracing is already in place. Add alerting on: model fallback frequency (indicates provider outages), query latency P95, guardrail trigger rates, and extraction confidence distributions.

---

## Tech Stack

| Category | Technology | Purpose |
|---|---|---|
| Language | Python 3.13 | Runtime |
| API | FastAPI + Uvicorn | REST API server |
| UI | Streamlit | Chat interface, document upload |
| Database | SQLite (WAL mode) | Logistics data storage |
| ORM/Access | Raw SQL via engine plugin | Database portability |
| LLM Framework | LangChain | Model abstraction, message formatting |
| LLM Providers | Groq, Google Gemini, OpenRouter | Multi-provider fallback chain |
| Vector Store | ChromaDB + sentence-transformers | Schema retrieval (RAG) |
| Embedding Model | all-MiniLM-L6-v2 | Semantic similarity for table/example retrieval |
| Visualization | Plotly | Auto-generated charts |
| Document Processing | PyMuPDF + ReportLab + Pillow | PDF rendering, image conversion, sample generation |
| Observability | LangFuse | LLM call tracing, latency, token usage |
| Validation | Pydantic v2 | Request/response schemas, type safety |
| Configuration | YAML (PyYAML) | Models, prompts, settings, guardrail rules |
| Testing | pytest + pytest-asyncio | 266 unit/integration tests |
| Evaluation | Custom harness | 150 LLM test cases, 4 embedding benchmarks |

---

## Project Structure

```
word_to_sql/
  app/
    Home.py                          # Streamlit landing page
    pages/
      1_Analytics.py                 # Analytics chat interface
      2_Document_Upload.py           # Document upload + extraction UI
      3_Query_Documents.py           # Cross-domain document queries
    api_client.py                    # HTTP client for Streamlit -> FastAPI
  src/
    api/
      main.py                        # FastAPI app factory
      routers/
        health.py                    # GET /api/health
        analytics.py                 # POST /api/analytics/query
        documents.py                 # POST /api/documents/extract, etc.
    analytics/
      agent.py                       # Pipeline orchestrator (7 steps)
      planner.py                     # Intent classification (LLM)
      sql_generator.py               # NL-to-SQL generation (LLM)
      verifier.py                    # SQL validation + execution
      visualizer.py                  # Chart suggestion + Plotly creation
    vision/
      agent.py                       # Document extraction orchestrator
      extractor.py                   # Vision LLM document extraction
      validator.py                   # Confidence checks, field validation
      storage.py                     # Document persistence
    services/
      analytics_service.py           # Service layer (DI, caching)
      vision_service.py              # Vision service layer
    repositories/
      base.py                        # Engine-agnostic base repository
      analytics_repo.py              # Analytics query execution
      document_repo.py               # Document CRUD operations
      engines/
        interface.py                 # DatabaseEngine protocol
        sqlite.py                    # SQLite implementation
    knowledge/
      semantic_layer.py              # Rich table/column descriptions
      few_shot_examples.py           # Curated question-to-SQL pairs
      vector_store.py                # ChromaDB semantic search
      sql_pitfall_checker.py         # SQL anti-pattern detection
    guardrails/
      input_guards.py                # SQL/prompt injection detection
      output_guards.py               # DML blocking, grounding checks
    models/
      llm_factory.py                 # LLM fallback chain factory
    common/
      schemas.py                     # Pydantic domain models
      exceptions.py                  # Exception hierarchy
      config_loader.py               # YAML config loading
    database.py                      # Database facade functions
    tracing.py                       # LangFuse integration
  config/
    models.yaml                      # LLM providers, fallback chains
    prompts.yaml                     # All LLM prompt templates
    settings.yaml                    # Guardrails, thresholds, DB config
  db/
    schema.sql                       # DDL for all 7 tables + indexes
    seed_data.py                     # Generate 100K+ rows of test data
    sample_documents.py              # Generate 25 sample PDFs/images
    samples/                         # Generated sample documents
    uploads/                         # User-uploaded documents
  eval/
    MODEL_SELECTION.md               # Model evaluation decision record
    run_llm_eval.py                  # LLM benchmark harness (150 cases)
    run_embedding_eval.py            # Embedding benchmark harness
    test_cases.yaml                  # 150 eval test cases
    results/                         # Benchmark reports and raw data
  tests/                             # 266 pytest tests
  scripts/
    start_dev.sh                     # Start both servers
  run_api.py                         # Uvicorn entry point
  requirements.txt                   # Python dependencies
  pytest.ini                         # Test configuration
  .env.example                       # Environment variable template
  .python-version                    # Python 3.13
```
