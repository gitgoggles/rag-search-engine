# RAG Search Engine

## Re-ranking LLM

Individual re-ranking uses an OpenAI-compatible chat completions API. Defaults:

```dotenv
RERANK_BASE_URL=https://llama.rucknet.uk/v1
RERANK_MODEL=Qwen3.6-35B-A3B-UD-Q4_K_M
```

Override these in `.env` or your environment to switch servers or models.
No API key is needed for this server. Set `RERANK_API_KEY` only if your
provider requires authentication; otherwise the client uses a dummy key.
Query enhancement still uses OpenRouter and requires `OPENROUTER_API_KEY`.

```sh
uv run cli/hybrid_search_cli.py rrf-search "family movie about bears in the woods" --rerank-method individual --limit 3
```
