# ChatGPT Conversation Extractor

Extract and analyze ChatGPT history using **streaming ingestion + durable on-disk state**.

This example is designed for the real constraint you described: the export can be **huge** (e.g. 1GB / “millions of tokens”), while the model you run is effectively limited to **~50k usable tokens**.

## Features

- **True streaming JSON parsing** for top-level export arrays (no `json.load()` memory bomb)
- **SQLite index** (single file) for queryable retrieval across large history
- **Token-bounded analysis**: only pull relevant slices into the LLM
- **Externalized state**: planning/log files persisted to disk (task_plan/findings/progress)

## Installation

```bash
# From repo root (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Set API key (example: Anthropic)
export ANTHROPIC_API_KEY='sk-ant-...'
```

## Usage

### 1) Index the export (streaming)

```bash
# Build a durable SQLite index from a large export (no LLM calls)
uv run python examples/chatgpt-extractor/extract.py /path/to/conversations.json \
  --db extractions.db \
  --verbatim-only
```

### 2) Ask questions with an RLM agent (planning files + code execution)

```bash
uv run python examples/chatgpt-extractor/agent_query.py \
  --db extractions.db \
  --workspace runs/query-001 \
  --question "Find conversations about MCP + filesystem context engineering in Oct–Nov 2025 and summarize the key decisions." \
  --start-date 2025-10-01 \
  --end-date 2025-11-30 \
  --topics "MCP,file system,context engineering" \
  --limit 50
```

### 3) Optional: LLM extraction during ingestion

```bash
uv run python examples/chatgpt-extractor/extract.py /path/to/conversations.json \
  --db extractions.db \
  --model haiku \
  --topics "MCP,RLM" \
  --date 2025-11-12
```

## Architecture

See `examples/chatgpt-extractor/ARCHITECTURE.md` for the concrete “filesystem-as-memory + RLM compute + planning files” design.

## Notes

- **Share URL fetching**: `chatgpt.com/share/...` often blocks automated fetching. The reliable path is using the local export JSON.
- **Legacy scripts**: `chatgpt_extractor.py` and `fs_extractor.py` exist for historical experiments and use a copied `examples/chatgpt-extractor/rlm/` implementation. The recommended path is `extract.py` + `agent_query.py`.

## Output Files

Each extraction creates a folder with:

| File | Description |
|------|-------------|
| `full_extraction.md` | Complete markdown report with all extracted data |
| `key_points.md` | Summary of key points and action items |
| `verbatim.txt` | Raw conversation text |
| `extraction.json` | Structured JSON data for programmatic use |
| `code_snippets.md` | All code from the conversation (if any) |

## How It Works

The system avoids context-window failure by keeping large data on disk:

- The export is **streamed** into SQLite (index).
- The RLM agent **executes code** to retrieve only relevant slices.
- Planning files keep the agent aligned across long runs.

## Model Pricing

| Model | Input | Output | Speed |
|-------|-------|--------|-------|
| Haiku | $0.80/1M | $4.00/1M | Fastest |
| Sonnet | $3.00/1M | $15.00/1M | Balanced |
| Opus | $15.00/1M | $75.00/1M | Most capable |

For most extractions, Haiku is recommended as it provides excellent results at minimal cost.

## Example

```bash
$ python chatgpt_extractor.py "https://chatgpt.com/share/abc123..."

 Fetching: https://chatgpt.com/share/abc123...
   Title: Building a REST API with FastAPI
   Messages: 24

============================================================
 EXTRACTING CONVERSATION WITH RLM
============================================================
Title: Building a REST API with FastAPI
Messages: 24
Model: claude-haiku-4-5-20250929
============================================================

------------------------------------------------------------
 EXTRACTION COMPLETE
------------------------------------------------------------
Key Points: 8
Topics: 5
Action Items: 3
Code Snippets: 6
Questions: 12
------------------------------------------------------------
Tokens Used: 15,432
Cost: $0.0234
------------------------------------------------------------

 Saved to: extractions/20260109_building-a-rest-api-with-fastapi/
   - full_extraction.md
   - key_points.md
   - verbatim.txt
   - extraction.json
   - code_snippets.md
```

## Troubleshooting

### "Failed to fetch URL"

ChatGPT share links must be public. Make sure you've created a share link from the conversation.

### "No messages found"

Some ChatGPT pages may use different HTML structures. Try exporting your conversation as JSON and using the `--json` flag.

### API Key Issues

Make sure your Anthropic API key is set:

```bash
export ANTHROPIC_API_KEY='sk-ant-api03-...'
```
