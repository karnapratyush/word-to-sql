"""Knowledge layer — semantic schema registry + few-shot examples.

Uses ChromaDB vector store for:
1. Rich table/column descriptions with business context and sample values
2. Cross-table examples using golden customers (consistent data stories)
3. Few-shot question→SQL pairs for accuracy improvement

At query time: embed the user's question → retrieve relevant tables + examples → build focused LLM context.
"""
