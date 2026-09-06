"""Orchestration service for the complete end-to-end RAG answer pipeline."""

from __future__ import annotations

import logging
import time
from typing import Any

from adaptive_framework.core.exceptions import (
    ContextBuildError,
    LLMProviderError,
    LLMTimeoutError,
    PromptBuildError,
)
from adaptive_framework.rag.generation.answer import RAGAnswer, SourceReference
from adaptive_framework.rag.generation.context_builder import ContextBuilder
from adaptive_framework.rag.generation.prompt_builder import PromptBuilder
from adaptive_framework.rag.interfaces.i_llm_provider import ILLMProvider
from adaptive_framework.rag.interfaces.i_retrieval_engine import IRetrievalEngine
from adaptive_framework.rag.retrieval.retrieval_result import RetrievalResult

logger = logging.getLogger(__name__)


class RAGService:
    """Orchestrates query answering across retrieval, context bounding, prompt construction, and LLM inference."""

    def __init__(
        self,
        retrieval_engine: IRetrievalEngine,
        context_builder: ContextBuilder,
        prompt_builder: PromptBuilder,
        llm_provider: ILLMProvider,
    ) -> None:
        """Initialise RAGService.

        Args:
            retrieval_engine: Engine executing vector/hybrid queries.
            context_builder: Component bounding and deduplicating chunks.
            prompt_builder: Component structuring prompts with prompt injection defenses.
            llm_provider: Component interfacing with the LLM backend.
        """
        self.retrieval_engine = retrieval_engine
        self.context_builder = context_builder
        self.prompt_builder = prompt_builder
        self.llm_provider = llm_provider

    def answer(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float = 0.0,
        metadata_filters: dict[str, Any] | None = None,
    ) -> RAGAnswer:
        """Process a query end-to-end and return a grounded RAGAnswer.

        Handles edge cases (empty query, no results, engine errors, LLM errors) gracefully.

        Args:
            query: User search query.
            top_k: Optional maximum retrieval results to fetch.
            min_score: Minimum similarity score filter.
            metadata_filters: Optional metadata filtering criteria.

        Returns:
            RAGAnswer object populated with answer text, evidence, and provenance.
        """
        total_start = time.perf_counter()

        # Handle empty query edge case
        if not query or not query.strip():
            total_elapsed = time.perf_counter() - total_start
            return RAGAnswer(
                query=query,
                answer_text="Empty query provided. Please ask a valid question.",
                evidence=(),
                source_references=(),
                retrieval_metrics={},
                provider_name=self.llm_provider.get_provider_name(),
                model_name=self.llm_provider.get_model_name(),
                retrieval_latency_s=0.0,
                context_latency_s=0.0,
                prompt_latency_s=0.0,
                generation_latency_s=0.0,
                total_latency_s=total_elapsed,
                context_chars=0,
                num_retrieved=0,
                generation_status="empty_query",
            )

        # 1. Retrieval stage
        retrieval_start = time.perf_counter()
        results: list[RetrievalResult] = []
        retrieval_metrics: dict[str, Any] = {}
        try:
            results = self.retrieval_engine.retrieve(
                query=query.strip(),
                top_k=top_k,
                min_score=min_score,
                metadata_filters=metadata_filters,
            )
            retrieval_metrics = self.retrieval_engine.get_metrics()
        except Exception as e:
            logger.error(f"Retrieval engine failed for query: {query}: {e}")
            retrieval_elapsed = time.perf_counter() - retrieval_start
            total_elapsed = time.perf_counter() - total_start
            return RAGAnswer(
                query=query,
                answer_text="An error occurred during evidence retrieval.",
                evidence=(),
                source_references=(),
                retrieval_metrics={},
                provider_name=self.llm_provider.get_provider_name(),
                model_name=self.llm_provider.get_model_name(),
                retrieval_latency_s=retrieval_elapsed,
                context_latency_s=0.0,
                prompt_latency_s=0.0,
                generation_latency_s=0.0,
                total_latency_s=total_elapsed,
                context_chars=0,
                num_retrieved=0,
                generation_status="retrieval_error",
            )

        retrieval_latency = time.perf_counter() - retrieval_start

        # 2. Context Building stage
        context_start = time.perf_counter()
        try:
            context = self.context_builder.build(results)
        except Exception as e:
            logger.error(f"Context builder failed: {e}")
            context_elapsed = time.perf_counter() - context_start
            total_elapsed = time.perf_counter() - total_start
            return RAGAnswer(
                query=query,
                answer_text="An error occurred while preparing evidence context.",
                evidence=tuple(results),
                source_references=(),
                retrieval_metrics=retrieval_metrics,
                provider_name=self.llm_provider.get_provider_name(),
                model_name=self.llm_provider.get_model_name(),
                retrieval_latency_s=retrieval_latency,
                context_latency_s=context_elapsed,
                prompt_latency_s=0.0,
                generation_latency_s=0.0,
                total_latency_s=total_elapsed,
                context_chars=0,
                num_retrieved=len(results),
                generation_status="context_error",
            )
        context_latency = time.perf_counter() - context_start

        # Handle zero retrieved evidence
        if not context.evidence_blocks:
            total_elapsed = time.perf_counter() - total_start
            return RAGAnswer(
                query=query,
                answer_text="No relevant documents found to answer this question.",
                evidence=(),
                source_references=(),
                retrieval_metrics=retrieval_metrics,
                provider_name=self.llm_provider.get_provider_name(),
                model_name=self.llm_provider.get_model_name(),
                retrieval_latency_s=retrieval_latency,
                context_latency_s=context_latency,
                prompt_latency_s=0.0,
                generation_latency_s=0.0,
                total_latency_s=total_elapsed,
                context_chars=0,
                num_retrieved=0,
                generation_status="no_results",
            )

        # 3. Prompt Building stage
        prompt_start = time.perf_counter()
        try:
            built_prompt = self.prompt_builder.build(context, query)
        except Exception as e:
            logger.error(f"Prompt builder failed: {e}")
            prompt_elapsed = time.perf_counter() - prompt_start
            total_elapsed = time.perf_counter() - total_start
            return RAGAnswer(
                query=query,
                answer_text="An error occurred while constructing the prompt.",
                evidence=tuple(context.retained_results),
                source_references=(),
                retrieval_metrics=retrieval_metrics,
                provider_name=self.llm_provider.get_provider_name(),
                model_name=self.llm_provider.get_model_name(),
                retrieval_latency_s=retrieval_latency,
                context_latency_s=context_latency,
                prompt_latency_s=prompt_elapsed,
                generation_latency_s=0.0,
                total_latency_s=total_elapsed,
                context_chars=context.total_chars,
                num_retrieved=len(results),
                generation_status="prompt_error",
            )
        prompt_latency = time.perf_counter() - prompt_start

        # 4. LLM Generation stage
        generation_start = time.perf_counter()
        try:
            llm_response = self.llm_provider.generate(built_prompt)
            answer_text = llm_response.text
            status = "ok"
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            generation_elapsed = time.perf_counter() - generation_start
            total_elapsed = time.perf_counter() - total_start
            return RAGAnswer(
                query=query,
                answer_text="An error occurred during answer generation.",
                evidence=tuple(context.retained_results),
                source_references=(),
                retrieval_metrics=retrieval_metrics,
                provider_name=self.llm_provider.get_provider_name(),
                model_name=self.llm_provider.get_model_name(),
                retrieval_latency_s=retrieval_latency,
                context_latency_s=context_latency,
                prompt_latency_s=prompt_latency,
                generation_latency_s=generation_elapsed,
                total_latency_s=total_elapsed,
                context_chars=context.total_chars,
                num_retrieved=len(results),
                generation_status="llm_error",
            )
        generation_latency = time.perf_counter() - generation_start

        # Construct distinct source references
        seen_sources: set[tuple[str, str, tuple[int, ...]]] = set()
        source_refs: list[SourceReference] = []
        for r in context.retained_results:
            key = (r.document_id, r.source_file, r.page_numbers)
            if key not in seen_sources:
                seen_sources.add(key)
                source_refs.append(
                    SourceReference(
                        document_id=r.document_id,
                        source_file=r.source_file,
                        page_numbers=r.page_numbers,
                    )
                )

        total_latency = time.perf_counter() - total_start

        return RAGAnswer(
            query=query,
            answer_text=answer_text,
            evidence=tuple(context.retained_results),
            source_references=tuple(source_refs),
            retrieval_metrics=retrieval_metrics,
            provider_name=self.llm_provider.get_provider_name(),
            model_name=self.llm_provider.get_model_name(),
            retrieval_latency_s=retrieval_latency,
            context_latency_s=context_latency,
            prompt_latency_s=prompt_latency,
            generation_latency_s=generation_latency,
            total_latency_s=total_latency,
            context_chars=context.total_chars,
            num_retrieved=len(results),
            generation_status=status,
        )
