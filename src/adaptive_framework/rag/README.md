# RAG Demo Package

## Architecture (v2.0 §2.5)

```
Document Text
        ↓
   IChunker        ← chunking strategy (fixed_size | sentence | semantic)
        ↓
   IEmbedder       ← sentence-transformers or equivalent
        ↓
   IVectorStore    ← ChromaDB | FAISS | Qdrant
        ↓
   Query Engine
```

## Status: Phase 5 Placeholder

## Configuration

`configs/rag.yaml`