"""Generation module for Phase 4.5 RAG answer synthesis."""

from adaptive_framework.rag.generation.answer import RAGAnswer, SourceReference
from adaptive_framework.rag.generation.context_builder import (
    BuiltContext,
    ContextBuilder,
    EvidenceBlock,
)
from adaptive_framework.rag.generation.fake_llm_provider import FakeLLMProvider
from adaptive_framework.rag.generation.llm_factory import create_llm_provider
from adaptive_framework.rag.generation.llm_response import LLMResponse
from adaptive_framework.rag.generation.clinical_rag_orchestrator import (
    ClinicalRAGOrchestrator,
)
from adaptive_framework.rag.generation.prompt_builder import BuiltPrompt, PromptBuilder
from adaptive_framework.rag.generation.rag_service import RAGService
from adaptive_framework.rag.generation.real_llm_provider import RealLLMProvider

__all__ = [
    "EvidenceBlock",
    "BuiltContext",
    "ContextBuilder",
    "BuiltPrompt",
    "PromptBuilder",
    "LLMResponse",
    "FakeLLMProvider",
    "RealLLMProvider",
    "create_llm_provider",
    "SourceReference",
    "RAGAnswer",
    "RAGService",
    "ClinicalRAGOrchestrator",
]
