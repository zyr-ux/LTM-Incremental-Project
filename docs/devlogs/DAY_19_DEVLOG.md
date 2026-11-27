# Day 19 Devlog: RAG Pipeline — Grounded Q&A

## Objective
Answer customer and staff questions with retrieval-grounded LLM responses using the Day-18 FAISS knowledge base, cutting hallucination and returning supporting passages for verification.

## What Was Built
- **`src/rag_assistant.py` (`RAGAssistant`)**:
  - Wired `RetailKnowledgeBase.search()` into a LangChain `ChatPromptTemplate | RunnableLambda | StrOutputParser` chain.
  - Crafted `RAG_SYSTEM_PROMPT` to instruct the LLM to answer strictly from retrieved context, cite `[source]` filenames, and refuse to guess when context is missing.
  - Added `min_score` filtering and a `NO_CONTEXT_ANSWER` fallback when no relevant passages are found.
  - Returns `answer`, `sources`, `passages`, and `grounded` so staff can verify every reply.
- **`src/assistant_service.py` (`RetailAssistantModule`)**:
  - Replaced the Day-14 ungrounded `SupportChatbot` with `RAGAssistant`.
  - Updated `chat()` to include `sources`, `passages`, and `grounded` in its response payload.
- **`tests/test_rag_assistant.py`**:
  - Added unit tests for grounded Q&A, supporting passage return, missing-context anti-hallucination, low-score filtering, and `RetailAssistantModule` integration.

## Verification
- Verified syntax and structure of `src/rag_assistant.py`, `src/assistant_service.py`, and `tests/test_rag_assistant.py`.
