## Goals & Constraints

We want a system that can answer **narrow questions** about a **huge ChatGPT export** (e.g. ~1GB JSON, effectively ~10M tokens), while the “base model” we run is limited to **~50,000 usable tokens**.

### Key constraint

- We cannot “fit” the export into model context. The model must **interact with data on disk** and only pull in what it needs.

This matches the “filesystem as memory” / “dynamic context discovery” idea: store large context as files, and load only relevant slices when needed ([LangChain blog](https://blog.langchain.com/how-agents-can-use-filesystems-for-context-engineering/), [Cursor blog](https://cursor.com/blog/dynamic-context-discovery)).

---

## Minimal, Robust Architecture (No Over-Engineering)

### Core principle

- **Disk is memory. Context window is cache.**

So the system is split into two simple stages:

1. **Index (deterministic, streaming)**: Convert the monolithic JSON export into a queryable “environment” on disk.
2. **Analyze (agentic, token-bounded)**: Use RLM code execution + targeted retrieval + (optional) sub-LLM calls to answer a specific question over a specific time window.

---

## Components

### 1) Source of truth (immutable)

- `conversations.json` (or whatever the export file is called)

### 2) Durable index (queryable)

- `extractions.db` (SQLite)
  - Conversations table: ids, titles, timestamps
  - Messages table: role/content, conversation_id, timestamps

This turns “1GB blob” into “queryable store”. It’s still just a file.

### 3) Externalized agent state (persistent files)

A workspace directory (per run or per query):

- `task_plan.md` (goal, phases, current status)
- `findings.md` (discoveries / extracted insights)
- `progress.md` (append-only log; errors; checkpoints)

This is the “planning-with-files” pattern, but the important implementation detail for **this repo**:

> `LocalREPL` runs code in a temp directory and deletes it at the end of a completion.  
> So planning files must be written to an **explicit workspace path outside the REPL temp dir**.

We do this by passing `workspace_dir` through `context` and using `setup_code` to define helpers that always write to that path.

---

## Data Flow

### Stage A: Index (streaming)

1. Stream the top-level JSON array using `rlm.utils.streaming_json.iter_json_array_items()`
2. For each conversation:
   - Extract metadata (title, id, create_time)
   - Extract messages (role, content, timestamps)
3. Insert into SQLite in batches (transaction per N conversations)
4. Commit often so crashes don’t lose much work

**Result**: you can now query “topics in date range” without rescanning 1GB in memory.

### Stage B: Analyze (RLM agent over the index)

1. Build a small `context` dict:
   - `db_path` (absolute path)
   - `workspace_dir` (absolute path)
   - `query_spec` (date range + topics + desired output)
2. Create `RLM(custom_system_prompt=..., environment_kwargs={setup_code: ...})`
3. Root LM writes small `repl` code that:
   - Reads/updates `task_plan.md`
   - Runs SQL queries to retrieve only relevant conversations/messages
   - Chunks retrieved text to stay under ~50k tokens
   - Uses `llm_query` / `llm_query_batched` only on selected chunks for semantic work
   - Writes summaries/findings to `findings.md`, logs to `progress.md`
4. Final output is a path to produced artifacts (markdown/json), or a final report string.

This is “externalized compute”: the model reasons by writing code; the code pulls data from disk; the disk holds the memory.

---

## Why This Works With a “Shitty” 50k-Token Model

### The model never sees the full dataset

It only sees:

- a small plan file excerpt
- small query parameters
- tiny slices of retrieved conversations
- summaries/buffers that it wrote to disk

### The system is resilient

- Crashes don’t destroy state (SQLite + append-only logs + planning files)
- You can restart and continue because state is externalized

---

## Implementation Phases (Concrete Tasks)

### Phase 0 — Make “streaming” real (DONE)

- Implement a real streaming iterator (no `json.load`) for top-level JSON arrays.
- Replace the fake streaming in existing extractors.

### Phase 1 — Index-first workflow

- Add an explicit `--index-only` mode that only builds the DB (no LLM calls).
- Add `--resume` that skips already-seen conversation ids (idempotent index build).

### Phase 2 — Agent (RLM) query mode

- Add an RLM-powered CLI that:
  - reads `task_plan.md`
  - queries the DB
  - calls sub-LLMs only for semantic extraction
  - writes `findings.md`/`progress.md`

### Phase 3 — Token / cost controls

- Max conversations per run
- Max chars per chunk
- Max sub-LLM calls and per-call char budget
- Store prompts/responses on disk for auditability

