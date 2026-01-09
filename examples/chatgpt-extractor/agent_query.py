#!/usr/bin/env python3
"""
RLM-powered query agent over an indexed ChatGPT export (SQLite).

Why this exists:
- The base model is token-limited (~50k usable tokens).
- The ChatGPT export can be ~1GB / ~10M tokens.
- So the model should NOT "read the export" directly; it should run code that queries an index.
- Agent state (plan/findings/progress) must persist on disk outside the REPL temp dir.

Workflow (minimal):
1) Build an index DB once (e.g. `python examples/chatgpt-extractor/extract.py export.json --verbatim-only`)
2) Ask questions over time windows/topics using this RLM agent:
   `python examples/chatgpt-extractor/agent_query.py --db extractions.db --workspace runs/q1 --question "..."`
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

# Add the repo root to path if needed (for running examples directly)
_repo_root = Path(__file__).parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from rlm.core.rlm import RLM


AGENT_SYSTEM_PROMPT = """You are an agent operating in a Python REPL environment (via RLM).

Hard constraints:
- Your usable context window is ~50,000 tokens. Assume anything large must stay on disk.

- DO NOT load large JSON exports into memory. Query the SQLite DB instead.
- Keep REPL prints small. Persist important information to disk (planning files).

You have a `context` dict with:
- context["db_path"]: absolute path to a SQLite database of ChatGPT conversations/messages
- context["workspace_dir"]: absolute path to a persistent workspace directory
- context["query_spec"]: structured filters (may be empty)

Your REPL environment also provides:
- llm_query(prompt, model=None)
- llm_query_batched(prompts, model=None)
- FINAL(...) / FINAL_VAR(...)

Persistence rules (critical):
- The REPL current working directory is a TEMP directory and will be deleted.
- Any persistent files MUST be written under context["workspace_dir"] using ABSOLUTE paths.

Workflow:
1) In the first REPL step, call ensure_planning_files().
2) Read task_plan.md (path: task_plan_path()) to orient.
3) Query the DB for relevant conversations/messages using db_query_* helpers.
4) For semantic work, chunk text and use llm_query/llm_query_batched.
5) Write results to findings.md and progress.md.
6) Return the final answer via FINAL(...) or point to the generated report file.
"""


def _build_setup_code() -> str:
    # NOTE: This runs inside LocalREPL after `context` is loaded.
    return r"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now().isoformat()


def workspace_dir() -> Path:
    return Path(context["workspace_dir"]).expanduser().resolve()


def task_plan_path() -> Path:
    return workspace_dir() / "task_plan.md"


def findings_path() -> Path:
    return workspace_dir() / "findings.md"


def progress_path() -> Path:
    return workspace_dir() / "progress.md"


def ensure_planning_files() -> None:
    wd = workspace_dir()
    wd.mkdir(parents=True, exist_ok=True)

    if not task_plan_path().exists():
        task_plan_path().write_text(
            "# Task Plan: ChatGPT History Query\n\n"
            f"> Created: {_now()}\n\n"
            "## Goal\n"
            "- Answer a specific question over a time/topic slice of the indexed ChatGPT export.\n\n"
            "## Phases\n"
            "- [ ] Phase 1: Validate DB + inspect schema\n"
            "- [ ] Phase 2: Retrieve candidate conversations (date/topic)\n"
            "- [ ] Phase 3: Semantic extraction (llm_query)\n"
            "- [ ] Phase 4: Synthesize final answer + write report\n\n"
            "## Current Status\n"
            f"**Status:** INITIALIZED\n"
            f"**Last Updated:** {_now()}\n",
            encoding="utf-8",
        )

    if not findings_path().exists():
        findings_path().write_text(
            "# Findings\n\n"
            f"> Created: {_now()}\n\n"
            "## Notes\n\n",
            encoding="utf-8",
        )

    if not progress_path().exists():
        progress_path().write_text(
            "# Progress Log\n\n"
            f"> Created: {_now()}\n\n",
            encoding="utf-8",
        )


def log_progress(event: str, details: str = "") -> None:
    line = f"- [{_now()}] {event}"
    if details:
        line += f" — {details}"
    progress_path().write_text(progress_path().read_text(encoding="utf-8") + line + "\n", encoding="utf-8")


def append_findings(text: str) -> None:
    findings_path().write_text(findings_path().read_text(encoding="utf-8") + text + "\n", encoding="utf-8")


def db_path() -> Path:
    return Path(context["db_path"]).expanduser().resolve()


def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path()))
    conn.row_factory = sqlite3.Row
    return conn


def db_list_tables() -> list[str]:
    conn = db_connect()
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        return [r["name"] for r in rows]
    finally:
        conn.close()


def db_query_conversations(
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    topic_substrings: list[str] | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    topic_substrings = topic_substrings or []
    conn = db_connect()
    try:
        where = []
        params: list[Any] = []

        if start_date:
            where.append("date >= ?")
            params.append(start_date)
        if end_date:
            where.append("date <= ?")
            params.append(end_date)

        # Cheap first-pass filter: title/topics JSON string (if populated) and message contents.
        # NOTE: This is intentionally simple; the goal is to narrow candidates, not perfect retrieval.
        if topic_substrings:
            subclauses = []
            for t in topic_substrings:
                subclauses.append("(title LIKE ? OR topics LIKE ? OR id IN (SELECT conversation_id FROM messages WHERE content LIKE ?))")
                like = f"%{t}%"
                params.extend([like, like, like])
            where.append("(" + " OR ".join(subclauses) + ")")

        query = "SELECT id, title, date, message_count, word_count FROM conversations"
        if where:
            query += " WHERE " + " AND ".join(where)
        query += " ORDER BY date DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def db_get_messages(conversation_id: str) -> list[dict[str, Any]]:
    conn = db_connect()
    try:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id ASC",
            (conversation_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def format_conversation_for_llm(conversation_id: str, *, max_chars: int = 40_000) -> str:
    msgs = db_get_messages(conversation_id)
    chunks: list[str] = []
    total = 0
    for m in msgs:
        part = f"[{m.get('role','').upper()}]: {m.get('content','')}\n\n"
        if total + len(part) > max_chars:
            break
        chunks.append(part)
        total += len(part)
    return "".join(chunks).strip()
"""


def _backend_kwargs_for(backend: str, model_name: str) -> dict[str, Any]:
    if backend == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for backend=anthropic")
        return {"api_key": api_key, "model_name": model_name}

    if backend in {"openai", "openrouter", "vercel", "vllm"}:
        # OpenAIClient can read API key from environment based on base_url.
        # We keep kwargs minimal and let the client resolve env vars.
        return {"api_key": None, "model_name": model_name}

    raise ValueError(f"Unsupported backend: {backend}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RLM agent query over a ChatGPT export SQLite DB")
    parser.add_argument("--db", required=True, help="Path to SQLite DB created by extract.py")
    parser.add_argument(
        "--workspace",
        required=True,
        help="Directory for persistent planning files + outputs (task_plan.md, findings.md, progress.md)",
    )
    parser.add_argument("--backend", default="anthropic", help="RLM backend (anthropic/openai/...)")
    parser.add_argument("--model", default="claude-haiku-4-5-20250929", help="Model name")
    parser.add_argument("--question", required=True, help="The question/task to answer over your history")
    parser.add_argument("--start-date", help="Filter conversations from this date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="Filter conversations up to this date (YYYY-MM-DD)")
    parser.add_argument("--topics", help="Comma-separated topic substrings for filtering")
    parser.add_argument("--limit", type=int, default=50, help="Max candidate conversations to retrieve")
    parser.add_argument("--max-iterations", type=int, default=10, help="Max RLM iterations")
    args = parser.parse_args()

    db_path = str(Path(args.db).expanduser().resolve())
    workspace_dir = str(Path(args.workspace).expanduser().resolve())

    topic_substrings = []
    if args.topics:
        topic_substrings = [t.strip() for t in args.topics.split(",") if t.strip()]

    context = {
        "db_path": db_path,
        "workspace_dir": workspace_dir,
        "query_spec": {
            "start_date": args.start_date,
            "end_date": args.end_date,
            "topic_substrings": topic_substrings,
            "limit": args.limit,
        },
    }

    rlm = RLM(
        backend=args.backend,
        backend_kwargs=_backend_kwargs_for(args.backend, args.model),
        environment="local",
        environment_kwargs={"setup_code": _build_setup_code()},
        max_iterations=args.max_iterations,
        custom_system_prompt=AGENT_SYSTEM_PROMPT,
        verbose=True,
    )

    completion = rlm.completion(prompt=context, root_prompt=args.question)
    print("\n=== FINAL ANSWER ===\n")
    print(completion.response)


if __name__ == "__main__":
    main()