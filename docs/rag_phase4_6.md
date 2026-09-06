# Phase 4.6 -- Real LLM Provider Integration

This document details the design, architecture, configuration, and verification of the Real LLM Provider layer in the Adaptive Distributed Framework (v2.0).

---

## 1. Architecture Overview

Phase 4.6 integrates production LLM APIs behind the existing `ILLMProvider` interface established in Phase 4.5. The design preserves the complete decoupled RAG pipeline:

```
Doctor/User Question
        |
        v
 RetrievalEngine
        |
        v
BGE Query Embedding
        |
        v
FAISS Top-K Retrieval
        |
        v
 ContextBuilder (chunk deduplication & bounding)
        |
        v
  PromptBuilder (prompt injection isolation)
        |
        v
   ILLMProvider
   /          \
  /            \
FakeLLMProvider RealLLMProvider (Gemini / OpenAI / Groq / Ollama)
                    |
                    v
                 Real LLM
                    |
                    v
               LLMResponse
                    |
                    v
                RAGAnswer
              /           \
             /             \
      Answer Text       Provenance
                           |
                           v
                    Document + Page
```

---

## 2. Why This Abstraction Exists

The `ILLMProvider` interface is a core application of the Strategy and Provider patterns:

1. **Vendor Independence**: Neither `RAGService`, `ContextBuilder`, nor `PromptBuilder` has any direct dependency on Google Gemini, OpenAI, Groq, or Anthropic SDKs.
2. **Infrastructure Decoupling**: LLM APIs are infrastructure dependencies subject to network variability, quota constraints, and API revisions. The abstraction insulates core RAG logic.
3. **Pluggable Providers**: Switching from offline testing to live cloud generation requires changing only configuration (`provider: fake` vs `provider: gemini` or `provider: openai`), with zero source code edits.

---

## 3. Why Tests Use FakeLLMProvider

Unit tests and CI suites must be:
- **100% Deterministic**: Ground truth assertions must never fail due to model drift, non-zero temperature, or prompt formatting variations.
- **100% Offline**: Running `pytest` requires zero external network access and zero API keys.
- **Fast and Cost-Free**: Automated tests run in seconds without incurring vendor API costs or consuming rate limit quotas.

The `FakeLLMProvider` parses the retrieved evidence blocks in the prompt and synthesizes predictable answers with provenance citations, allowing complete validation of the answer pipeline without network access.

---

## 4. RealLLMProvider Implementation Details

`RealLLMProvider` is implemented in `adaptive_framework.rag.generation.real_llm_provider` and supports two major REST protocols via `httpx`:

1. **Google Gemini REST API**:
   - Endpoint: `https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent`
   - Authentication: `x-goog-api-key` request header.
   - Payload: Structured `system_instruction`, `contents` array with user messages, and `generationConfig` (temperature, maxOutputTokens).
   - Default Model: `gemini-1.5-flash`.

2. **OpenAI-Compatible REST API**:
   - Endpoint: `{base_url}/chat/completions` (supports OpenAI, Groq, Ollama, LocalAI, vLLM).
   - Authentication: `Authorization: Bearer {api_key}` header.
   - Payload: `messages` array (`system` and `user` roles), `model`, `temperature`, `max_tokens`.
   - Default Model: `gpt-4o-mini`.

---

## 5. Credential and Secret Management

Security rules enforced by `RealLLMProvider`:
- **No Hard-Coded Keys**: Credentials are never hard-coded in code, YAML configurations, test files, or documentation.
- **Environment Variables**: Keys are resolved at runtime via environment variables:
  - `GEMINI_API_KEY` for Google Gemini.
  - `OPENAI_API_KEY` for OpenAI / Groq.
- **No Leakage**: Secrets are never printed in console output, written to logs, or embedded in exception error messages.
- **Explicit Failure**: If the required environment variable is missing, a clear `LLMProviderError` is raised with instructions on how to set the variable.

---

## 6. Error Handling and Exception Hierarchy

All provider failures are mapped cleanly into the framework exception hierarchy:

| Condition | Underlying Cause | Framework Exception |
|---|---|---|
| Missing Credential | Environment variable unset | `LLMProviderError` |
| Invalid Backend | Unsupported provider name | `LLMProviderError` |
| Authentication Error | HTTP 401 / 403 | `LLMProviderError` |
| Rate Limit Reached | HTTP 429 | `LLMProviderError` |
| Network Timeout | `httpx.TimeoutException` | `LLMTimeoutError` |
| Network Failure | `httpx.ConnectError` | `LLMProviderError` |
| Provider Server Error | HTTP 500 / 502 / 503 | `LLMGenerationError` |
| Malformed Response | Non-JSON or invalid schema | `LLMGenerationError` |
| Empty Completion | Zero-length text returned | `LLMGenerationError` |

---

## 7. Configuration Example

In `configs/rag.yaml`:

```yaml
rag:
  enabled: true

  # Phase 4.5 & 4.6: LLM Generation configuration
  generation:
    # Provider: "fake" (default for CI/offline) | "real" | "gemini" | "openai"
    provider: "fake"
    model: "gemini-1.5-flash"
    temperature: 0.0
    max_tokens: 512
    api_key_env_var: "GEMINI_API_KEY"
    timeout_seconds: 30.0
    # Optional custom base URL for OpenAI-compatible endpoints:
    # base_url: "https://api.groq.com/openai/v1"
```

To switch between providers:
- Offline development / CI: `provider: "fake"`
- Google Gemini: `provider: "gemini"` with `$env:GEMINI_API_KEY="<key>"`
- OpenAI: `provider: "openai"` with `$env:OPENAI_API_KEY="<key>"`

---

## 8. Clinical Grounding and Safety Positioning

The Adaptive Distributed Framework is an experimental biomedical and clinical document processing prototype.

Important safety guidelines:
1. **Provenance Enforcement**: Every answer generated by `RAGService` includes full source references (`document_id`, `source_file`, `page_numbers`).
2. **Untrusted Evidence Boundary**: Retrieved document text is strictly isolated inside `--- BEGIN UNTRUSTED RETRIEVED EVIDENCE ---` delimiters to neutralize potential prompt injection in untrusted PDFs.
3. **Grounding Instruction**: The system prompt instructs the model to rely solely on retrieved evidence and state "Insufficient evidence to answer this question" when facts are not present.
4. **Clinical Role**: This system is designed for evidence retrieval and summarization assistance; it is not an autonomous medical diagnostic or prescription system. Clinical decisions remain the sole responsibility of qualified healthcare professionals.

---

## 9. End-to-End Demonstration

To run the end-to-end demonstration script:

```bash
# Offline deterministic demonstration:
.venv/Scripts/python scripts/test_real_rag.py "What findings are documented in Soumya Das's report?" --provider fake

# Live real LLM generation (with GEMINI_API_KEY set in environment):
.venv/Scripts/python scripts/test_real_rag.py "What findings are documented in Soumya Das's report?" --provider gemini

# Live OpenAI generation (with OPENAI_API_KEY set in environment):
.venv/Scripts/python scripts/test_real_rag.py "What findings are documented in Soumya Das's report?" --provider openai
```
