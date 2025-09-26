# RAG System Improvements Plan

## ✅ Step-by-step Application Plan

### 1. Query Strategy Updates
- [ ] Combine ChromaDB + PostgreSQL full-text (`tsvector`) in your `get_rag_context()`
- [ ] Add synonym expansion using NLTK/WordNet or custom rules
- [ ] Index sentence-level and paragraph-level chunks in Chroma

### 2. Embedding Optimizations
- [ ] Split texts by sentence using `nltk.sent_tokenize()` or `spacy`
- [ ] Add metadata (timestamp, phase, role) to each ChromaDB embedding
- [ ] Cache frequent queries using Redis with a TTL (e.g., `functools.lru_cache` or manual store)

### 3. Memory Improvements
- [ ] Use recency weights in similarity scoring (can be done in your re-ranking step)
- [ ] Cluster embeddings by topic (e.g., KMeans on embedding space) during background processing
- [ ] Score importance per memory (use custom logic or classify with LLM)

### 4. Speed Boosting
- [ ] Apply the SQL indexes via Alembic or manually
- [ ] Switch Chroma insert/search to batch mode
- [ ] Enable memory-mapping in Chroma (`persist_directory`, `settings={'persist': True, 'mmapped': True}`)

### 5. Caching Layer
- [ ] Add Redis for:
  - RAG result cache
  - Tool output cache
  - Semantic query cache (cosine sim > 0.95)
- [ ] Cache last N messages in `app.state` or memory buffer

### 6. Tool System Improvements
- [ ] Stream LLM output while tools run (`asyncio.TaskGroup`)
- [ ] Buffer tool results in memory, write to DB at end
- [ ] Pass tool outputs forward via `tool_context` dict

### 7. Prompt System Enhancements
- [ ] Dynamically adjust RAG token budget
- [ ] Add summarization to old messages (use `GPT-3.5` or similar locally if available)
- [ ] Track `memory_state` and inject into `_build_system_prompt()`

### 8. Architecture Upgrades
- [ ] Run tool/memory tasks in background using `asyncio.create_task()` or Celery
- [ ] Add lifecycle policies (delete old, keep important, summarize old convos)
- [ ] Log timing for all major actions (retrieval, embed, tool call)
