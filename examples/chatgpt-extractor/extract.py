#!/usr/bin/env python3
"""
PRODUCTION-READY ChatGPT Export Extractor

The simplest, most robust solution for extracting insights from ChatGPT exports.

Features:
- Works with ANY model (Anthropic, OpenAI, DeepSeek, Ollama, OpenRouter)
- Handles 1GB+ files
- Smart chunking for limited context windows (50K tokens)
- Extracts key points, action items, summaries
- SQLite storage + Markdown export
- Full observability with progress tracking
- Checkpoint/resume capability

Usage:
    # Basic extraction (uses ANTHROPIC_API_KEY by default)
    python extract.py conversations.json

    # With specific model
    python extract.py conversations.json --model haiku
    python extract.py conversations.json --model gpt-4o-mini
    python extract.py conversations.json --model deepseek

    # Filter by date
    python extract.py conversations.json --date 2025-01-08

    # Filter by topics
    python extract.py conversations.json --topics "VNC,MCP,RLM"

    # Verbatim only (no LLM calls)
    python extract.py conversations.json --verbatim-only

    # Query existing database
    python extract.py --db extractions.db --search "action items"
    python extract.py --db extractions.db --list
    python extract.py --db extractions.db --report
"""

import os
import sys
import re
import json
import sqlite3
import hashlib
import argparse
import time
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Generator
from dataclasses import dataclass, asdict


# ============================================================================
# CONFIGURATION
# ============================================================================

# Model configurations - add more as needed
MODELS = {
    # Anthropic
    "haiku": {
        "provider": "anthropic",
        "model": "claude-haiku-4-5-20250929",
        "max_tokens": 4096,
        "context_window": 200000,
    },
    "sonnet": {
        "provider": "anthropic",
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 8192,
        "context_window": 200000,
    },
    # OpenAI
    "gpt-4o-mini": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "max_tokens": 4096,
        "context_window": 128000,
    },
    "gpt-4o": {
        "provider": "openai",
        "model": "gpt-4o",
        "max_tokens": 4096,
        "context_window": 128000,
    },
    # DeepSeek
    "deepseek": {
        "provider": "deepseek",
        "model": "deepseek-chat",
        "max_tokens": 4096,
        "context_window": 64000,
    },
    # Local Ollama
    "ollama": {
        "provider": "ollama",
        "model": "llama3.2",
        "max_tokens": 4096,
        "context_window": 32000,
    },
}

# Default topics to filter for
DEFAULT_TOPICS = [
    "computer use", "VNC", "file system", "MCP", "RLM",
    "agent", "claude", "cursor", "context engineering",
    "anthropic", "model context protocol"
]

# Chunk size for processing (leave room for prompt + response)
CHUNK_SIZE_CHARS = 40000  # ~10K tokens, safe for 50K context


# ============================================================================
# SIMPLE LLM CLIENT - Works with any provider
# ============================================================================

class LLMClient:
    """
    Universal LLM client that works with any provider via HTTP.
    No complex SDKs needed - just urllib.
    """

    def __init__(self, model_name: str = "haiku"):
        if model_name not in MODELS:
            # Try to use as-is
            self.config = {
                "provider": "anthropic",
                "model": model_name,
                "max_tokens": 4096,
                "context_window": 50000,
            }
        else:
            self.config = MODELS[model_name]

        self.provider = self.config["provider"]
        self.model = self.config["model"]
        self.max_tokens = self.config["max_tokens"]
        self.context_window = self.config["context_window"]

        # Get API key
        self.api_key = self._get_api_key()

        # Track usage
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0

    def _get_api_key(self) -> str:
        """Get API key for the provider."""
        key_map = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "ollama": None,  # No key needed
        }

        env_var = key_map.get(self.provider)
        if env_var is None:
            return ""

        key = os.getenv(env_var, "")
        if not key:
            print(f"  Warning: {env_var} not set")
        return key

    def complete(self, prompt: str, system: str = "") -> str:
        """
        Send a completion request to the LLM.
        Returns the response text.
        """
        self.total_calls += 1

        if self.provider == "anthropic":
            return self._anthropic_complete(prompt, system)
        elif self.provider == "openai":
            return self._openai_complete(prompt, system)
        elif self.provider == "deepseek":
            return self._deepseek_complete(prompt, system)
        elif self.provider == "ollama":
            return self._ollama_complete(prompt, system)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _anthropic_complete(self, prompt: str, system: str) -> str:
        """Call Anthropic API."""
        url = "https://api.anthropic.com/v1/messages"

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

        data = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            data["system"] = system

        return self._http_post(url, headers, data)

    def _openai_complete(self, prompt: str, system: str) -> str:
        """Call OpenAI API."""
        url = "https://api.openai.com/v1/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        data = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }

        return self._http_post(url, headers, data, openai_format=True)

    def _deepseek_complete(self, prompt: str, system: str) -> str:
        """Call DeepSeek API (OpenAI-compatible)."""
        url = "https://api.deepseek.com/v1/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        data = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }

        return self._http_post(url, headers, data, openai_format=True)

    def _ollama_complete(self, prompt: str, system: str) -> str:
        """Call local Ollama API."""
        url = "http://localhost:11434/api/generate"

        headers = {"Content-Type": "application/json"}

        full_prompt = f"{system}\n\n{prompt}" if system else prompt

        data = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
        }

        return self._http_post(url, headers, data, ollama_format=True)

    def _http_post(self, url: str, headers: dict, data: dict,
                   openai_format: bool = False, ollama_format: bool = False) -> str:
        """Make HTTP POST request and parse response."""
        try:
            request = urllib.request.Request(
                url,
                data=json.dumps(data).encode('utf-8'),
                headers=headers,
                method='POST'
            )

            with urllib.request.urlopen(request, timeout=120) as response:
                result = json.loads(response.read().decode('utf-8'))

            # Parse response based on format
            if ollama_format:
                return result.get("response", "")
            elif openai_format:
                # Track tokens
                usage = result.get("usage", {})
                self.total_input_tokens += usage.get("prompt_tokens", 0)
                self.total_output_tokens += usage.get("completion_tokens", 0)
                return result["choices"][0]["message"]["content"]
            else:
                # Anthropic format
                usage = result.get("usage", {})
                self.total_input_tokens += usage.get("input_tokens", 0)
                self.total_output_tokens += usage.get("output_tokens", 0)
                return result["content"][0]["text"]

        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8') if e.fp else ""
            raise RuntimeError(f"API error {e.code}: {error_body[:500]}")
        except Exception as e:
            raise RuntimeError(f"Request failed: {e}")

    def get_usage(self) -> dict:
        """Get usage statistics."""
        return {
            "calls": self.total_calls,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
        }


# ============================================================================
# DATABASE - SQLite for simplicity
# ============================================================================

class Database:
    """SQLite database for storing extractions."""

    def __init__(self, db_path: str = "extractions.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        """Create database schema."""
        cursor = self.conn.cursor()

        # Conversations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT,
                date TEXT,
                message_count INTEGER,
                word_count INTEGER,
                topics TEXT,
                summary TEXT,
                source_file TEXT,
                processed_at TEXT
            )
        """)

        # Messages
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT,
                role TEXT,
                content TEXT,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
        """)

        # Key Points
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS key_points (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT,
                point TEXT,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
        """)

        # Action Items
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS action_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT,
                item TEXT,
                completed INTEGER DEFAULT 0,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
        """)

        # Progress/Checkpoints
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                event TEXT,
                details TEXT
            )
        """)

        # Indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_conv_date ON conversations(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id)")

        self.conn.commit()

    def save_conversation(self, conv_id: str, title: str, date: str,
                          messages: List[dict], topics: List[str],
                          summary: str, source_file: str):
        """Save a conversation with all its data."""
        cursor = self.conn.cursor()

        word_count = sum(len(m.get('content', '').split()) for m in messages)

        # Upsert conversation
        cursor.execute("""
            INSERT OR REPLACE INTO conversations
            (id, title, date, message_count, word_count, topics, summary, source_file, processed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (conv_id, title, date, len(messages), word_count,
              json.dumps(topics), summary, source_file, datetime.now().isoformat()))

        # Replace messages
        cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
        for msg in messages:
            cursor.execute("""
                INSERT INTO messages (conversation_id, role, content)
                VALUES (?, ?, ?)
            """, (conv_id, msg.get('role', ''), msg.get('content', '')))

        self.conn.commit()

    def save_key_points(self, conv_id: str, points: List[str]):
        """Save key points for a conversation."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM key_points WHERE conversation_id = ?", (conv_id,))
        for point in points:
            cursor.execute("""
                INSERT INTO key_points (conversation_id, point)
                VALUES (?, ?)
            """, (conv_id, point))
        self.conn.commit()

    def save_action_items(self, conv_id: str, items: List[str]):
        """Save action items for a conversation."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM action_items WHERE conversation_id = ?", (conv_id,))
        for item in items:
            cursor.execute("""
                INSERT INTO action_items (conversation_id, item)
                VALUES (?, ?)
            """, (conv_id, item))
        self.conn.commit()

    def log_progress(self, event: str, details: str = ""):
        """Log progress event."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO progress (timestamp, event, details)
            VALUES (?, ?, ?)
        """, (datetime.now().isoformat(), event, details))
        self.conn.commit()

    def get_checkpoint(self) -> Optional[str]:
        """Get last processed conversation ID."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT details FROM progress
            WHERE event = 'PROCESSED'
            ORDER BY id DESC LIMIT 1
        """)
        row = cursor.fetchone()
        return row[0] if row else None

    def list_conversations(self) -> List[dict]:
        """List all conversations."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, title, date, message_count, word_count
            FROM conversations ORDER BY date DESC
        """)
        return [dict(row) for row in cursor.fetchall()]

    def search(self, query: str) -> List[dict]:
        """Search messages."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT m.*, c.title as conversation_title
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            WHERE m.content LIKE ?
            LIMIT 50
        """, (f"%{query}%",))
        return [dict(row) for row in cursor.fetchall()]

    def get_all_key_points(self) -> List[dict]:
        """Get all key points."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT k.point, c.title, c.date
            FROM key_points k
            JOIN conversations c ON k.conversation_id = c.id
            ORDER BY c.date DESC
        """)
        return [dict(row) for row in cursor.fetchall()]

    def get_all_action_items(self) -> List[dict]:
        """Get all action items."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT a.item, a.completed, c.title, c.date
            FROM action_items a
            JOIN conversations c ON a.conversation_id = c.id
            ORDER BY c.date DESC
        """)
        return [dict(row) for row in cursor.fetchall()]

    def get_stats(self) -> dict:
        """Get database statistics."""
        cursor = self.conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM conversations")
        convs = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM messages")
        msgs = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM key_points")
        kps = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM action_items")
        ais = cursor.fetchone()[0]

        cursor.execute("SELECT SUM(word_count) FROM conversations")
        words = cursor.fetchone()[0] or 0

        return {
            "conversations": convs,
            "messages": msgs,
            "key_points": kps,
            "action_items": ais,
            "total_words": words,
        }

    def close(self):
        """Close connection."""
        self.conn.close()


# ============================================================================
# JSON PARSER
# ============================================================================

def parse_chatgpt_export(file_path: str) -> Generator[dict, None, None]:
    """
    Parse ChatGPT export JSON file.
    Yields conversations one at a time.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, list):
        for conv in data:
            yield conv
    elif isinstance(data, dict):
        if 'mapping' in data:
            yield data
        elif 'conversations' in data:
            for conv in data['conversations']:
                yield conv


def extract_messages(conv: dict) -> List[dict]:
    """Extract messages from a conversation."""
    messages = []
    mapping = conv.get('mapping', {})

    if mapping:
        nodes = []
        for node_data in mapping.values():
            message = node_data.get('message')
            if message:
                role = message.get('author', {}).get('role', 'unknown')
                content_data = message.get('content', {})
                parts = content_data.get('parts', [])
                content = '\n'.join(str(p) for p in parts if p)
                create_time = message.get('create_time', 0) or 0

                if role in ['user', 'assistant'] and content.strip():
                    nodes.append({
                        'time': create_time,
                        'role': role,
                        'content': content.strip()
                    })

        nodes.sort(key=lambda x: x['time'])
        messages = [{'role': n['role'], 'content': n['content']} for n in nodes]

    return messages


def get_conversation_date(conv: dict) -> str:
    """Get date from conversation."""
    create_time = conv.get('create_time')
    if isinstance(create_time, (int, float)) and create_time > 0:
        return datetime.fromtimestamp(create_time).strftime("%Y-%m-%d")
    return "unknown"


def matches_topics(conv: dict, topics: List[str]) -> tuple[bool, List[str]]:
    """Check if conversation matches any topics."""
    if not topics:
        return True, []

    matched = []

    # Check title
    title = conv.get('title', '').lower()

    # Check messages
    mapping = conv.get('mapping', {})
    all_content = title + " "
    for node_data in mapping.values():
        message = node_data.get('message', {})
        content_data = message.get('content', {})
        parts = content_data.get('parts', [])
        all_content += ' '.join(str(p) for p in parts if p).lower() + " "

    for topic in topics:
        if topic.lower() in all_content:
            matched.append(topic)

    return len(matched) > 0, matched


# ============================================================================
# CHUNKING - For limited context windows
# ============================================================================

def chunk_conversation(messages: List[dict], max_chars: int = CHUNK_SIZE_CHARS) -> List[str]:
    """
    Chunk a conversation into pieces that fit in context window.
    Returns list of text chunks.
    """
    chunks = []
    current_chunk = ""

    for msg in messages:
        role = msg.get('role', 'unknown').upper()
        content = msg.get('content', '')
        formatted = f"[{role}]: {content}\n\n"

        if len(current_chunk) + len(formatted) > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = formatted
        else:
            current_chunk += formatted

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# ============================================================================
# EXTRACTION - The actual LLM calls
# ============================================================================

EXTRACTION_PROMPT = """Analyze this conversation excerpt and extract:

1. KEY POINTS: List the most important insights, facts, or conclusions (3-10 points)
2. ACTION ITEMS: List any tasks, todos, or next steps mentioned (if any)
3. SUMMARY: One paragraph summary of what was discussed

Format your response EXACTLY like this:
KEY POINTS:
- Point 1
- Point 2
...

ACTION ITEMS:
- Item 1
- Item 2
...

SUMMARY:
Your summary here.

If there are no action items, write "ACTION ITEMS: None"

CONVERSATION:
{conversation}"""


def extract_insights(llm: LLMClient, messages: List[dict], title: str) -> dict:
    """
    Extract insights from a conversation using LLM.
    Handles chunking for long conversations.
    """
    chunks = chunk_conversation(messages)

    all_key_points = []
    all_action_items = []
    summaries = []

    for i, chunk in enumerate(chunks):
        prompt = EXTRACTION_PROMPT.format(conversation=chunk)

        try:
            response = llm.complete(prompt)

            # Parse response
            kp = extract_section(response, "KEY POINTS:")
            ai = extract_section(response, "ACTION ITEMS:")
            summary = extract_section(response, "SUMMARY:")

            all_key_points.extend(kp)
            if ai and ai != ["None"]:
                all_action_items.extend(ai)
            if summary:
                summaries.append(summary[0] if isinstance(summary, list) else summary)

        except Exception as e:
            print(f"      Warning: Chunk {i+1} failed: {e}")

    # If multiple chunks, synthesize
    final_summary = " ".join(summaries) if summaries else f"Conversation about: {title}"

    return {
        "key_points": list(set(all_key_points)),  # Dedupe
        "action_items": list(set(all_action_items)),
        "summary": final_summary[:1000],  # Truncate if too long
    }


def extract_section(text: str, header: str) -> List[str]:
    """Extract a section from LLM response."""
    # Find the section
    pattern = rf'{re.escape(header)}\s*(.*?)(?=\n\n[A-Z]|\Z)'
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)

    if not match:
        return []

    section = match.group(1).strip()

    # Check for "None"
    if section.lower() in ['none', 'n/a', 'none.']:
        return []

    # Parse bullet points
    items = re.findall(r'[-*•]\s*(.+?)(?=\n[-*•]|\Z)', section, re.DOTALL)

    if items:
        return [item.strip() for item in items if item.strip()]

    # If no bullets, return as single item
    return [section] if section else []


# ============================================================================
# MARKDOWN EXPORT
# ============================================================================

def export_to_markdown(db: Database, output_dir: str = "output"):
    """Export all data to markdown files."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    stats = db.get_stats()

    # Master index
    index_content = f"""# ChatGPT Export Extraction

> Generated: {datetime.now().isoformat()}
> Database: {db.db_path}

## Statistics

| Metric | Value |
|--------|-------|
| Conversations | {stats['conversations']} |
| Messages | {stats['messages']} |
| Key Points | {stats['key_points']} |
| Action Items | {stats['action_items']} |
| Total Words | {stats['total_words']:,} |

---

## All Action Items

"""

    for item in db.get_all_action_items():
        status = "[x]" if item['completed'] else "[ ]"
        index_content += f"- {status} {item['item']} *(from: {item['title'][:30]})*\n"

    index_content += "\n---\n\n## All Key Points\n\n"

    # Group by conversation
    by_conv = {}
    for kp in db.get_all_key_points():
        title = kp['title']
        if title not in by_conv:
            by_conv[title] = []
        by_conv[title].append(kp['point'])

    for title, points in by_conv.items():
        index_content += f"\n### {title[:50]}\n\n"
        for point in points:
            index_content += f"- {point}\n"

    index_content += "\n---\n\n## Conversations\n\n"

    for conv in db.list_conversations():
        index_content += f"- [{conv['title'][:50]}]({conv['date']}_{conv['id'][:8]}.md) ({conv['message_count']} msgs)\n"

    (output / "index.md").write_text(index_content)

    # Individual conversation files
    cursor = db.conn.cursor()
    for conv in db.list_conversations():
        conv_id = conv['id']

        # Get full conversation data
        cursor.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,))
        full_conv = dict(cursor.fetchone())

        cursor.execute("SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id", (conv_id,))
        messages = cursor.fetchall()

        cursor.execute("SELECT point FROM key_points WHERE conversation_id = ?", (conv_id,))
        key_points = [row[0] for row in cursor.fetchall()]

        cursor.execute("SELECT item FROM action_items WHERE conversation_id = ?", (conv_id,))
        action_items = [row[0] for row in cursor.fetchall()]

        content = f"""# {full_conv['title']}

> **Date:** {full_conv['date']}
> **Messages:** {full_conv['message_count']}
> **Words:** {full_conv['word_count']:,}
> **Topics:** {full_conv['topics']}

---

## Summary

{full_conv['summary'] or 'No summary available.'}

---

## Key Points

"""
        for point in key_points:
            content += f"- {point}\n"

        if not key_points:
            content += "*No key points extracted.*\n"

        content += "\n---\n\n## Action Items\n\n"

        for item in action_items:
            content += f"- [ ] {item}\n"

        if not action_items:
            content += "*No action items.*\n"

        content += "\n---\n\n## Full Conversation\n\n"

        for msg in messages:
            role = "**USER:**" if msg[0] == 'user' else "**ASSISTANT:**"
            content += f"\n{role}\n\n{msg[1]}\n\n---\n"

        filename = f"{full_conv['date']}_{conv_id[:8]}.md"
        (output / filename).write_text(content)

    print(f"  Exported to: {output}/")
    return output


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_extraction(
    source_file: str,
    db_path: str = "extractions.db",
    model_name: str = "haiku",
    target_date: Optional[str] = None,
    topics: Optional[List[str]] = None,
    verbatim_only: bool = False,
    output_dir: str = "output",
):
    """
    Main extraction pipeline.
    """
    print(f"\n{'='*60}")
    print(" CHATGPT EXPORT EXTRACTOR")
    print(f"{'='*60}")
    print(f"Source: {source_file}")
    print(f"Model: {model_name}")
    print(f"Date filter: {target_date or 'all'}")
    print(f"Topic filter: {', '.join(topics[:3]) if topics else 'all'}...")
    print(f"Verbatim only: {verbatim_only}")
    print(f"{'='*60}\n")

    # Initialize
    db = Database(db_path)
    db.log_progress("START", f"source={source_file}")

    llm = None
    if not verbatim_only:
        try:
            llm = LLMClient(model_name)
            print(f"  LLM: {llm.provider}/{llm.model}")
        except Exception as e:
            print(f"  Warning: Could not init LLM: {e}")
            print(f"  Falling back to verbatim-only mode")
            verbatim_only = True

    # Get checkpoint for resume
    last_processed = db.get_checkpoint()
    skip_until_found = last_processed is not None

    # Process conversations
    processed = 0
    skipped = 0
    errors = 0

    for conv in parse_chatgpt_export(source_file):
        conv_id = conv.get('id', hashlib.md5(str(conv).encode()).hexdigest()[:16])
        title = conv.get('title', 'Untitled')
        conv_date = get_conversation_date(conv)

        # Resume logic
        if skip_until_found:
            if conv_id == last_processed:
                skip_until_found = False
            continue

        # Date filter
        if target_date and conv_date != target_date:
            skipped += 1
            continue

        # Topic filter
        if topics:
            matches, matched = matches_topics(conv, topics)
            if not matches:
                skipped += 1
                continue
        else:
            matched = []

        # Extract messages
        messages = extract_messages(conv)
        if not messages:
            skipped += 1
            continue

        processed += 1
        print(f"  [{processed}] {title[:45]}... ({len(messages)} msgs)")

        # Extract insights with LLM
        summary = ""
        key_points = []
        action_items = []

        if llm and not verbatim_only:
            try:
                insights = extract_insights(llm, messages, title)
                summary = insights.get('summary', '')
                key_points = insights.get('key_points', [])
                action_items = insights.get('action_items', [])
                print(f"      KPs: {len(key_points)}, Actions: {len(action_items)}")
            except Exception as e:
                print(f"      Error: {e}")
                errors += 1

        # Save to database
        db.save_conversation(
            conv_id=conv_id,
            title=title,
            date=conv_date,
            messages=messages,
            topics=matched,
            summary=summary,
            source_file=source_file,
        )

        if key_points:
            db.save_key_points(conv_id, key_points)
        if action_items:
            db.save_action_items(conv_id, action_items)

        db.log_progress("PROCESSED", conv_id)

    # Export to markdown
    print(f"\n  Exporting to markdown...")
    export_to_markdown(db, output_dir)

    # Summary
    stats = db.get_stats()
    usage = llm.get_usage() if llm else {}

    print(f"\n{'='*60}")
    print(" EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")
    print(f"Errors: {errors}")
    print(f"\nDatabase: {db_path}")
    print(f"  Conversations: {stats['conversations']}")
    print(f"  Messages: {stats['messages']}")
    print(f"  Key Points: {stats['key_points']}")
    print(f"  Action Items: {stats['action_items']}")

    if usage:
        print(f"\nLLM Usage:")
        print(f"  Calls: {usage['calls']}")
        print(f"  Tokens: {usage['total_tokens']:,}")

    print(f"\nOutput: {output_dir}/")
    print(f"{'='*60}")

    db.close()

    return {
        "processed": processed,
        "skipped": skipped,
        "errors": errors,
        "stats": stats,
    }


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Production-ready ChatGPT Export Extractor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract.py conversations.json
  python extract.py conversations.json --model haiku --date 2025-01-08
  python extract.py conversations.json --topics "VNC,MCP,RLM"
  python extract.py conversations.json --verbatim-only

  python extract.py --db extractions.db --list
  python extract.py --db extractions.db --search "action"
  python extract.py --db extractions.db --report

Models: haiku, sonnet, gpt-4o-mini, gpt-4o, deepseek, ollama
        """
    )

    parser.add_argument("source", nargs="?", help="ChatGPT export JSON file")
    parser.add_argument("--model", "-m", default="haiku", help="Model to use")
    parser.add_argument("--db", default="extractions.db", help="Database file")
    parser.add_argument("--date", "-d", help="Filter by date (YYYY-MM-DD)")
    parser.add_argument("--topics", "-t", help="Comma-separated topics to filter")
    parser.add_argument("--output", "-o", default="output", help="Output directory")
    parser.add_argument("--verbatim-only", action="store_true", help="Skip LLM extraction")

    # Query options
    parser.add_argument("--list", action="store_true", help="List conversations")
    parser.add_argument("--search", help="Search messages")
    parser.add_argument("--report", action="store_true", help="Generate report")
    parser.add_argument("--stats", action="store_true", help="Show statistics")

    args = parser.parse_args()

    # Query mode
    if args.list or args.search or args.report or args.stats:
        db = Database(args.db)

        if args.stats:
            stats = db.get_stats()
            print(f"\nDatabase: {args.db}")
            for k, v in stats.items():
                print(f"  {k}: {v:,}" if isinstance(v, int) else f"  {k}: {v}")

        if args.list:
            print(f"\nConversations in {args.db}:")
            for conv in db.list_conversations():
                print(f"  [{conv['date']}] {conv['title'][:50]} ({conv['message_count']} msgs)")

        if args.search:
            print(f"\nSearch results for '{args.search}':")
            for r in db.search(args.search)[:10]:
                print(f"  [{r['role']}] {r['content'][:100]}...")

        if args.report:
            export_to_markdown(db, args.output)

        db.close()
        return

    # Extraction mode
    if not args.source:
        parser.print_help()
        print("\nError: Please provide a source file or use query options")
        sys.exit(1)

    if not os.path.exists(args.source):
        print(f"Error: File not found: {args.source}")
        sys.exit(1)

    # Parse topics
    topics = None
    if args.topics:
        topics = [t.strip() for t in args.topics.split(",")]

    # Run extraction
    run_extraction(
        source_file=args.source,
        db_path=args.db,
        model_name=args.model,
        target_date=args.date,
        topics=topics,
        verbatim_only=args.verbatim_only,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
