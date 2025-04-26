# Migration Notes: Document Splitting Removal

With the upgrade to a 1 M-token LLM, we can index and process entire documents without chunking. This document outlines the key changes and deprecated behaviors.

## 1. New Configuration
- **USE_WHOLE_DOCUMENTS** (env var, default `true`): when `true`, bypasses chunking in `FileProcessor` and indexes full document text.

## 2. Backend Updates
- **FileProcessor** (`file_processor.py`): wraps chunking logic in `USE_WHOLE_DOCUMENTS`; concatenates full text into a single `DocumentIn`.
- **RetrieverTool** (`tools.py`): added `queryType` argument (`"hybrid"` vs `"simple"`); uses `vector_store.semantic_hybrid_search` by default; supports `vector_store.search` when `queryType="simple"`.
- **Agent Workflow** (`agent.py`): removed local reranking step; Azure AI Search semantic ranking is the default; summary retriever no longer blocks multi-file selection and allows up to 50 files.

## 3. Frontend Changes
- **Sidebar Guidance** (`file-sidebar.tsx`): updated help text to allow multi-document summarization.

## 4. Deprecated/Removed
- Single-document-only summarization guard.
- Hard-coded chunk size and overlap settings.

## 5. Next Steps & Testing
1. Verify retrieval and summarization with multiple large documents.
2. Run smoke tests for file processing, retrieval, and summary generation.
3. Monitor memory and performance impact.
