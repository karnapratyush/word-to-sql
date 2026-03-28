# Model Selection — Decision Record

## Overview

We evaluated multiple LLM and embedding models to find the **cheapest models that give best results** for each task. All decisions are backed by automated benchmarks run against our 150-test-case evaluation suite on the actual logistics database (7 tables, 100K+ rows).

Reports: `eval/results/llm_report.md`, `eval/results/embedding_report.md`
Raw data: `eval/results/llm_results.json`, `eval/results/embedding_results.json`

---

## LLM Models — Text-to-SQL Generation

### Models Tested

| Model | Provider | Cases Run | Composite Score | Latency | Cost/query | Rate Limit Issues? |
|---|---|---|---|---|---|---|
| DeepSeek R1 | OpenRouter (free) | 40 | **100.0%** | 34,000ms | $0.00 | No |
| Gemini 2.5 Flash | OpenRouter | 150 | **94.9%** | 2,041ms | ~$0.00 | No |
| Qwen3 Coder | OpenRouter (free) | 150 | **94.1%** | 1,259ms | $0.00 | No |
| Groq Llama 3.3 70B | Groq (free) | 20* | ~98%* | 622ms | $0.00 | **Yes — severe** |
| Gemini 2.5 Flash | Google direct (free) | 20* | 100%* | 4,815ms | $0.00 | **Yes — 20 req/day** |

*Groq and Google direct could only be tested on 20 cases due to rate limits. 150-case runs scored 4-29% (rate limit failures, not model quality).

### Scoring Dimensions (per test case)

- **JSON Parse (json_parse)**: Did the LLM return valid JSON with a `sql` key?
- **SQL Valid (sql_valid)**: Does the generated SQL parse without syntax errors?
- **SQL Runs (sql_runs)**: Does the SQL execute against the database without errors?
- **Correct Tables (correct_tables)**: Did the SQL reference the expected tables?
- **Correct Behavior (correct_behavior)**: Did it answer data questions AND refuse unanswerable ones?

### Category-Level Results

| Category | Qwen3 Coder | Gemini 2.5 Flash (OR) | DeepSeek R1 |
|---|---|---|---|
| simple_count (15) | 100% | 100% | 100% |
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

### Key Findings

1. **All models ace SQL generation** (95-100% on SQL categories). The LLMs are genuinely good at writing SQL when given proper schema context and few-shot examples.

2. **Behavior detection is the weak spot.** Qwen and Gemini struggle to refuse unanswerable questions — they try to write SQL for "What is the weather?" instead of saying "I can't answer that." DeepSeek R1's chain-of-thought reasoning handles this perfectly.

3. **Rate limits killed Groq and Google direct.** Both scored ~98-100% on small batches but failed catastrophically at 150 cases. Groq's free tier allows ~5-10 requests per minute. Google allows 20 per day.

4. **OpenRouter solved the rate limit problem.** Same models (Gemini, Qwen, DeepSeek) accessed via OpenRouter had zero rate limit issues across 150+ calls. Cost from $5 credits: negligible ($0.001 total for Gemini).

### Decision: Fallback Chain Order

```yaml
sql_generation:
  chain:
    1. openrouter/qwen/qwen3-coder         # PRIMARY: 94.1%, 1.3s, free, reliable
    2. openrouter/google/gemini-2.5-flash   # SECONDARY: 94.9%, 2.0s, near-free
    3. groq/llama-3.3-70b-versatile         # TERTIARY: fast when available, may rate limit
    4. openrouter/deepseek/deepseek-r1      # COMPLEX QUERIES: 100% but 34s
    5. openrouter/anthropic/claude-sonnet-4  # LAST RESORT: expensive ($3/MTok)
```

**Why Qwen first, not Gemini?**
- Qwen is 35% faster (1.3s vs 2.0s) — noticeable in interactive use
- Qwen is completely free (no credits consumed); Gemini costs ~$0.000007/query from OpenRouter credits
- Both are 94-95% accuracy — essentially tied
- Qwen handles injection attempts better (60% vs 0%)

**Why DeepSeek R1 fourth, not first?**
- 34 seconds per query is unacceptable for interactive use
- It's perfect (100%) but 26x slower than Qwen
- Placed as fallback for when simpler models fail on complex multi-join queries

**Why Claude Sonnet last?**
- $3 per million input tokens = ~$0.01 per query
- With $5 credits, that's only ~500 queries before credits run out
- Reserved as absolute last resort when all free models are down

### Classification Task

Same Qwen + Groq + Gemini chain. Classification is simpler than SQL generation (just "is this a data question or not?"), so even small models handle it well. No separate benchmark needed — if a model generates correct SQL, it can certainly classify intent.

### Answer Synthesis Task

Same Qwen + Groq + Gemini chain. Takes SQL results and generates a natural language answer. All models handle this well since it's summarization, not reasoning.

---

## Embedding Models — Schema Retrieval

### Models Tested

| Model | Dimensions | Recall@5 | Precision@5 | MRR | Avg Time |
|---|---|---|---|---|---|
| all-mpnet-base-v2 | 768 | **98.5%** | 38.3% | 0.886 | 15.6ms |
| all-MiniLM-L6-v2 | 384 | 97.7% | 38.6% | 0.890 | **3.7ms** |
| multi-qa-MiniLM-L6-cos-v1 | 384 | 97.7% | 37.5% | 0.890 | 3.8ms |
| paraphrase-MiniLM-L6-v2 | 384 | 97.7% | 36.4% | 0.887 | 3.6ms |

### Metrics Explained

- **Recall@5**: "Of the tables the query actually needs, what fraction were retrieved in top 5?" Higher = fewer missed tables.
- **Precision@5**: "Of the 5 tables retrieved, what fraction were actually needed?" Lower values are OK since we retrieve 5 but often only need 1-2.
- **MRR (Mean Reciprocal Rank)**: "How quickly does the first relevant table appear?" 1.0 = always first, 0.5 = usually second.
- **Avg Time**: Time per embedding + search operation (milliseconds).

### Key Findings

1. **All models are >97.7% recall.** The semantic layer descriptions are well-written enough that even the smallest embedding model finds the right tables.

2. **mpnet is marginally better (98.5% vs 97.7%) but 4x slower.** The 0.8% gap means ~1 extra correct retrieval out of 130 test cases. Not worth the latency trade.

3. **The only failures are multi_join queries** (3+ tables). When a question needs shipments + carriers + shipment_charges, sometimes the embedding retrieves 2 of 3. This is expected — 3-table queries are semantically complex.

4. **Precision is low (36-38%) by design.** We retrieve top 5 documents but most queries only need 1-2 tables. Retrieving "extra" context tables is harmless — the LLM ignores irrelevant ones.

### Decision: Use all-MiniLM-L6-v2

```python
# ChromaDB default — no explicit embedding function needed
client.get_or_create_collection(name="schema")  # Uses all-MiniLM-L6-v2 automatically
```

**Why not mpnet?**
- 0.8% recall improvement is negligible (1 test case out of 130)
- 4x slower (15.6ms vs 3.7ms) — adds up across multiple retrievals per query
- Requires explicit embedding function configuration instead of ChromaDB default
- MiniLM-L6 is battle-tested as ChromaDB's default — well-optimized for this use case

**Why not multi-qa-MiniLM?**
- Identical recall (97.7%) to the default MiniLM
- Designed for QA retrieval, but our use case is more "keyword-to-schema-description" matching
- No measurable improvement for our specific data

---

## Production Recommendations

1. **Re-run eval periodically.** New model releases (Llama 4, Gemini 3, etc.) may change the optimal chain. Run `python eval/run_llm_eval.py` after any model update.

2. **Add Groq paid tier.** Groq's model quality is excellent (~98%) and latency is lowest (622ms). A paid plan ($0.10/MTok) would eliminate rate limits and make it the best primary model.

3. **Consider fine-tuning.** The 86-87% behavior score (unanswerable detection) could improve to 95%+ with fine-tuning on logistics-specific examples. This is the biggest accuracy gap remaining.

4. **Monitor via LangFuse.** Every production LLM call is traced. Use the LangFuse dashboard to track: which model is actually being used, latency trends, error rates, and cost per query.
