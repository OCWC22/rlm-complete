"""
File-System RLM Extractor for ChatGPT Conversations

The SIMPLEST possible solution for processing ChatGPT exports:
- SQLite database (single file, no server)
- File system traversal
- RLM processing with any model
- Observable progress tracking

Inspired by:
- Cursor's "Everything is a File" approach
- LangChain's filesystem context engineering
- "Everything is Context" paper (arXiv:2512.05470)

Usage:
    # Process all ChatGPT exports in a directory
    python fs_extractor.py /path/to/exports/

    # Process with specific model
    python fs_extractor.py /path/to/exports/ --model haiku

    # Verbatim only (no RLM, just extract and store)
    python fs_extractor.py /path/to/exports/ --verbatim-only

    # Query the database
    python fs_extractor.py --query "action items"
    python fs_extractor.py --list-conversations
"""

import os
import sys
import re
import json
import sqlite3
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Generator
from dataclasses import dataclass, field

# Add the rlm module to path
sys.path.insert(0, str(Path(__file__).parent))


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class Message:
    """A single message in a conversation."""
    role: str
    content: str
    timestamp: Optional[str] = None


@dataclass
class Conversation:
    """A ChatGPT conversation."""
    id: str
    title: str
    messages: List[Message]
    create_time: Optional[str] = None
    source_file: str = ""


@dataclass
class ProcessedConversation:
    """Result of processing a conversation with RLM."""
    conversation_id: str
    title: str
    message_count: int
    key_points: List[str]
    action_items: List[str]
    topics: List[str]
    summary: str
    verbatim: str
    model_used: str
    cost_usd: float
    processed_at: str


# ============================================================================
# SQLITE DATABASE - THE SIMPLEST STORAGE
# ============================================================================

class ConversationDB:
    """
    SQLite database for storing processed conversations.

    Single file, no server, portable, queryable.
    """

    def __init__(self, db_path: str = "conversations.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        """Create database schema."""
        cursor = self.conn.cursor()

        # Conversations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT,
                message_count INTEGER,
                summary TEXT,
                verbatim TEXT,
                model_used TEXT,
                cost_usd REAL,
                source_file TEXT,
                create_time TEXT,
                processed_at TEXT
            )
        """)

        # Messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT,
                role TEXT,
                content TEXT,
                timestamp TEXT,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
        """)

        # Key points table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS key_points (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT,
                point TEXT,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
        """)

        # Action items table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS action_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT,
                item TEXT,
                completed INTEGER DEFAULT 0,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
        """)

        # Topics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS topics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT,
                topic TEXT,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
        """)

        # Create indexes for fast queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_keypoints_conv ON key_points(conversation_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_actions_conv ON action_items(conversation_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_topics_conv ON topics(conversation_id)")

        # Full-text search on messages
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                content,
                conversation_id,
                content='messages',
                content_rowid='id'
            )
        """)

        self.conn.commit()

    def save_conversation(self, conv: Conversation):
        """Save a raw conversation (before RLM processing)."""
        cursor = self.conn.cursor()

        # Insert conversation
        cursor.execute("""
            INSERT OR REPLACE INTO conversations (id, title, message_count, source_file, create_time)
            VALUES (?, ?, ?, ?, ?)
        """, (conv.id, conv.title, len(conv.messages), conv.source_file, conv.create_time))

        # Delete old messages
        cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conv.id,))

        # Insert messages
        for msg in conv.messages:
            cursor.execute("""
                INSERT INTO messages (conversation_id, role, content, timestamp)
                VALUES (?, ?, ?, ?)
            """, (conv.id, msg.role, msg.content, msg.timestamp))

        self.conn.commit()

    def save_processed(self, result: ProcessedConversation):
        """Save RLM processing results."""
        cursor = self.conn.cursor()

        # Update conversation with processed data
        cursor.execute("""
            UPDATE conversations
            SET summary = ?, verbatim = ?, model_used = ?, cost_usd = ?, processed_at = ?
            WHERE id = ?
        """, (result.summary, result.verbatim, result.model_used, result.cost_usd,
              result.processed_at, result.conversation_id))

        # Delete old extracted data
        cursor.execute("DELETE FROM key_points WHERE conversation_id = ?", (result.conversation_id,))
        cursor.execute("DELETE FROM action_items WHERE conversation_id = ?", (result.conversation_id,))
        cursor.execute("DELETE FROM topics WHERE conversation_id = ?", (result.conversation_id,))

        # Insert key points
        for point in result.key_points:
            cursor.execute("""
                INSERT INTO key_points (conversation_id, point)
                VALUES (?, ?)
            """, (result.conversation_id, point))

        # Insert action items
        for item in result.action_items:
            cursor.execute("""
                INSERT INTO action_items (conversation_id, item)
                VALUES (?, ?)
            """, (result.conversation_id, item))

        # Insert topics
        for topic in result.topics:
            cursor.execute("""
                INSERT INTO topics (conversation_id, topic)
                VALUES (?, ?)
            """, (result.conversation_id, topic))

        self.conn.commit()

    def get_conversation(self, conv_id: str) -> Optional[dict]:
        """Get a conversation by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def list_conversations(self) -> List[dict]:
        """List all conversations."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, title, message_count, processed_at, cost_usd
            FROM conversations
            ORDER BY create_time DESC
        """)
        return [dict(row) for row in cursor.fetchall()]

    def search_messages(self, query: str) -> List[dict]:
        """Full-text search across all messages."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT m.*, c.title as conversation_title
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            WHERE m.content LIKE ?
            LIMIT 50
        """, (f"%{query}%",))
        return [dict(row) for row in cursor.fetchall()]

    def get_all_action_items(self, completed: Optional[bool] = None) -> List[dict]:
        """Get all action items across all conversations."""
        cursor = self.conn.cursor()
        query = """
            SELECT a.*, c.title as conversation_title
            FROM action_items a
            JOIN conversations c ON a.conversation_id = c.id
        """
        if completed is not None:
            query += f" WHERE a.completed = {1 if completed else 0}"
        query += " ORDER BY a.id DESC"
        cursor.execute(query)
        return [dict(row) for row in cursor.fetchall()]

    def get_all_key_points(self) -> List[dict]:
        """Get all key points across all conversations."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT k.*, c.title as conversation_title
            FROM key_points k
            JOIN conversations c ON k.conversation_id = c.id
            ORDER BY k.id DESC
        """)
        return [dict(row) for row in cursor.fetchall()]

    def get_stats(self) -> dict:
        """Get database statistics."""
        cursor = self.conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM conversations")
        conv_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM messages")
        msg_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM key_points")
        kp_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM action_items")
        ai_count = cursor.fetchone()[0]

        cursor.execute("SELECT SUM(cost_usd) FROM conversations")
        total_cost = cursor.fetchone()[0] or 0

        return {
            "conversations": conv_count,
            "messages": msg_count,
            "key_points": kp_count,
            "action_items": ai_count,
            "total_cost_usd": total_cost
        }

    def close(self):
        """Close database connection."""
        self.conn.close()


# ============================================================================
# FILE SYSTEM SCANNER
# ============================================================================

class FileScanner:
    """
    Scans directories for ChatGPT export files.

    Supports:
    - Single conversation JSON files
    - ChatGPT data export format (conversations.json)
    - Directories with multiple exports
    """

    def __init__(self, path: str):
        self.path = Path(path)

    def scan(self) -> Generator[Conversation, None, None]:
        """Scan and yield conversations."""
        if self.path.is_file():
            yield from self._parse_file(self.path)
        elif self.path.is_dir():
            yield from self._scan_directory()
        else:
            raise ValueError(f"Invalid path: {self.path}")

    def _scan_directory(self) -> Generator[Conversation, None, None]:
        """Recursively scan directory for JSON files."""
        for json_file in self.path.rglob("*.json"):
            try:
                yield from self._parse_file(json_file)
            except Exception as e:
                print(f"  Warning: Failed to parse {json_file}: {e}")

    def _parse_file(self, file_path: Path) -> Generator[Conversation, None, None]:
        """Parse a JSON file and yield conversations."""
        print(f"  Scanning: {file_path.name}")

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Handle different formats
        if isinstance(data, list):
            # List of conversations (ChatGPT export format)
            for conv_data in data:
                conv = self._parse_conversation(conv_data, str(file_path))
                if conv:
                    yield conv
        elif isinstance(data, dict):
            # Single conversation or wrapped format
            if 'mapping' in data:
                # Single conversation with mapping
                conv = self._parse_conversation(data, str(file_path))
                if conv:
                    yield conv
            elif 'conversations' in data:
                # Wrapped format
                for conv_data in data['conversations']:
                    conv = self._parse_conversation(conv_data, str(file_path))
                    if conv:
                        yield conv
            else:
                # Try as single conversation
                conv = self._parse_conversation(data, str(file_path))
                if conv:
                    yield conv

    def _parse_conversation(self, data: dict, source_file: str) -> Optional[Conversation]:
        """Parse a single conversation from JSON data."""
        conv_id = data.get('id') or data.get('conversation_id') or str(hash(json.dumps(data, sort_keys=True)))[:16]
        title = data.get('title', 'Untitled Conversation')
        create_time = data.get('create_time')

        if create_time and isinstance(create_time, (int, float)):
            create_time = datetime.fromtimestamp(create_time).isoformat()

        messages = []

        # Try mapping format (ChatGPT export)
        mapping = data.get('mapping', {})
        if mapping:
            messages = self._parse_mapping(mapping)

        # Try messages array format
        if not messages:
            msg_array = data.get('messages', [])
            for msg_data in msg_array:
                role = msg_data.get('role', msg_data.get('author', {}).get('role', 'unknown'))
                content = msg_data.get('content', '')

                if isinstance(content, dict):
                    parts = content.get('parts', [])
                    content = '\n'.join(str(p) for p in parts if p)

                if role in ['user', 'assistant'] and content.strip():
                    messages.append(Message(role=role, content=content.strip()))

        if not messages:
            return None

        return Conversation(
            id=conv_id,
            title=title,
            messages=messages,
            create_time=create_time,
            source_file=source_file
        )

    def _parse_mapping(self, mapping: dict) -> List[Message]:
        """Parse ChatGPT mapping format."""
        messages = []
        nodes = []

        for node_id, node_data in mapping.items():
            message = node_data.get('message')
            if message:
                role = message.get('author', {}).get('role', 'unknown')
                content_data = message.get('content', {})
                parts = content_data.get('parts', [])
                content = '\n'.join(str(p) for p in parts if p)

                if role in ['user', 'assistant'] and content.strip():
                    create_time = message.get('create_time', 0) or 0
                    timestamp = None
                    if create_time:
                        timestamp = datetime.fromtimestamp(create_time).isoformat()
                    nodes.append((create_time, role, content.strip(), timestamp))

        # Sort by creation time
        nodes.sort(key=lambda x: x[0])

        for _, role, content, timestamp in nodes:
            messages.append(Message(role=role, content=content, timestamp=timestamp))

        return messages


# ============================================================================
# RLM PROCESSOR
# ============================================================================

class RLMProcessor:
    """
    Processes conversations using RLM.

    Can use any model: Haiku (fast/cheap), Sonnet, Opus.
    """

    MODELS = {
        "haiku": "claude-haiku-4-5-20250929",
        "sonnet": "claude-sonnet-4-5-20250929",
        "opus": "claude-opus-4-5-20251101",
    }

    def __init__(
        self,
        model: str = "haiku",
        max_iterations: int = 10,
        budget_limit: float = 0.50,
        enable_logging: bool = False,
    ):
        self.model = self.MODELS.get(model, model)
        self.max_iterations = max_iterations
        self.budget_limit = budget_limit
        self.enable_logging = enable_logging

        # Import RLM
        from rlm.rlm_repl import RLM_REPL
        from rlm.utils.llm import reset_cost_tracker
        self.RLM_REPL = RLM_REPL
        self.reset_cost_tracker = reset_cost_tracker

    def process(self, conv: Conversation) -> ProcessedConversation:
        """Process a conversation with RLM."""
        print(f"    Processing with RLM ({self.model})...")

        # Prepare context
        context = self._prepare_context(conv)

        # Create query
        query = """Analyze this conversation and extract:

1. **KEY POINTS**: List 5-15 key insights, conclusions, or important information discussed.

2. **ACTION ITEMS**: Extract any todos, next steps, or actionable items mentioned.

3. **TOPICS**: List the main topics/subjects discussed.

4. **SUMMARY**: Write a 2-3 sentence summary of what this conversation was about.

Be thorough and extract everything important. Format your response with clear sections."""

        # Reset cost tracker and create RLM
        self.reset_cost_tracker()
        rlm = self.RLM_REPL(
            model=self.model,
            recursive_model=self.model,
            provider="anthropic",
            enable_logging=self.enable_logging,
            max_iterations=self.max_iterations,
            budget_limit=self.budget_limit,
        )

        # Process
        try:
            result = rlm.completion(context=context, query=query)
        except RuntimeError as e:
            if "Budget" in str(e):
                result = f"[Budget exceeded] {str(e)}"
            else:
                raise

        # Get cost
        cost_summary = rlm.cost_summary()

        # Parse result
        return self._parse_result(conv, result, cost_summary)

    def _prepare_context(self, conv: Conversation) -> str:
        """Prepare conversation as RLM context."""
        lines = [
            f"CONVERSATION: {conv.title}",
            f"ID: {conv.id}",
            f"MESSAGES: {len(conv.messages)}",
            "",
            "=" * 60,
            ""
        ]

        for i, msg in enumerate(conv.messages, 1):
            role_label = "USER" if msg.role == "user" else "ASSISTANT"
            lines.append(f"[{role_label}] (Message #{i})")
            lines.append("-" * 40)
            lines.append(msg.content)
            lines.append("")
            lines.append("=" * 60)
            lines.append("")

        return "\n".join(lines)

    def _parse_result(self, conv: Conversation, result: str, cost_summary: dict) -> ProcessedConversation:
        """Parse RLM result into structured data."""
        key_points = self._extract_list(result, r'KEY POINTS?:?\n(.*?)(?=\n\n[A-Z#*]|\Z)')
        action_items = self._extract_list(result, r'ACTION ITEMS?:?\n(.*?)(?=\n\n[A-Z#*]|\Z)')
        topics = self._extract_list(result, r'TOPICS?:?\n(.*?)(?=\n\n[A-Z#*]|\Z)')

        # Extract summary
        summary_match = re.search(r'SUMMARY:?\n(.*?)(?=\n\n[A-Z#*]|\Z)', result, re.DOTALL | re.IGNORECASE)
        summary = summary_match.group(1).strip() if summary_match else result[:500]

        # Build verbatim
        verbatim = "\n\n".join([
            f"[{msg.role.upper()}]: {msg.content}"
            for msg in conv.messages
        ])

        return ProcessedConversation(
            conversation_id=conv.id,
            title=conv.title,
            message_count=len(conv.messages),
            key_points=key_points,
            action_items=action_items,
            topics=topics,
            summary=summary,
            verbatim=verbatim,
            model_used=self.model,
            cost_usd=cost_summary.get('total_cost_usd', 0),
            processed_at=datetime.now().isoformat()
        )

    def _extract_list(self, text: str, pattern: str) -> List[str]:
        """Extract a bulleted list from text."""
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if not match:
            return []

        section = match.group(1)
        items = re.findall(r'(?:^|\n)\s*(?:[-*\d.]+)\s*(.+?)(?=\n|$)', section)
        return [item.strip() for item in items if item.strip()]


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class FSExtractor:
    """
    Main pipeline for file-system based extraction.

    Combines:
    - File scanning
    - RLM processing
    - SQLite storage
    - Observable progress
    """

    def __init__(
        self,
        db_path: str = "conversations.db",
        model: str = "haiku",
        max_iterations: int = 10,
        budget_limit: float = 0.50,
        enable_logging: bool = False,
        verbatim_only: bool = False,
    ):
        self.db = ConversationDB(db_path)
        self.verbatim_only = verbatim_only

        if not verbatim_only:
            self.processor = RLMProcessor(
                model=model,
                max_iterations=max_iterations,
                budget_limit=budget_limit,
                enable_logging=enable_logging,
            )
        else:
            self.processor = None

    def process_path(self, path: str) -> dict:
        """Process all conversations in a path."""
        scanner = FileScanner(path)

        print(f"\n Scanning: {path}")
        print("=" * 60)

        processed = 0
        skipped = 0
        total_cost = 0.0

        for conv in scanner.scan():
            print(f"\n  [{processed + skipped + 1}] {conv.title[:50]}...")
            print(f"      Messages: {len(conv.messages)}")

            # Save raw conversation
            self.db.save_conversation(conv)

            if not self.verbatim_only and self.processor:
                try:
                    result = self.processor.process(conv)
                    self.db.save_processed(result)
                    total_cost += result.cost_usd
                    print(f"      Key points: {len(result.key_points)}")
                    print(f"      Action items: {len(result.action_items)}")
                    print(f"      Cost: ${result.cost_usd:.4f}")
                    processed += 1
                except Exception as e:
                    print(f"      Error: {e}")
                    skipped += 1
            else:
                # Verbatim only - create minimal result
                result = ProcessedConversation(
                    conversation_id=conv.id,
                    title=conv.title,
                    message_count=len(conv.messages),
                    key_points=[],
                    action_items=[],
                    topics=[],
                    summary="Verbatim extraction only.",
                    verbatim="\n\n".join([f"[{m.role.upper()}]: {m.content}" for m in conv.messages]),
                    model_used="N/A",
                    cost_usd=0,
                    processed_at=datetime.now().isoformat()
                )
                self.db.save_processed(result)
                processed += 1

        # Print summary
        print("\n" + "=" * 60)
        print(" EXTRACTION COMPLETE")
        print("=" * 60)
        stats = self.db.get_stats()
        print(f"Conversations: {stats['conversations']}")
        print(f"Messages: {stats['messages']}")
        print(f"Key points: {stats['key_points']}")
        print(f"Action items: {stats['action_items']}")
        print(f"Total cost: ${total_cost:.4f}")
        print(f"Database: {self.db.db_path}")
        print("=" * 60)

        return {
            "processed": processed,
            "skipped": skipped,
            "total_cost": total_cost,
            "stats": stats
        }

    def generate_report(self, output_path: str = "report.md"):
        """Generate a markdown report of all extractions."""
        stats = self.db.get_stats()
        conversations = self.db.list_conversations()
        all_action_items = self.db.get_all_action_items(completed=False)
        all_key_points = self.db.get_all_key_points()

        md = f"""# ChatGPT Conversation Extraction Report

> Generated: {datetime.now().isoformat()}
> Database: {self.db.db_path}

## Summary

| Metric | Value |
|--------|-------|
| Conversations | {stats['conversations']} |
| Messages | {stats['messages']} |
| Key Points | {stats['key_points']} |
| Action Items | {stats['action_items']} |
| Total Cost | ${stats['total_cost_usd']:.4f} |

---

## All Action Items ({len(all_action_items)})

"""
        for item in all_action_items:
            md += f"- [ ] {item['item']} *(from: {item['conversation_title'][:30]})*\n"

        md += f"""
---

## All Key Points ({len(all_key_points)})

"""
        # Group by conversation
        by_conv = {}
        for kp in all_key_points:
            title = kp['conversation_title']
            if title not in by_conv:
                by_conv[title] = []
            by_conv[title].append(kp['point'])

        for title, points in by_conv.items():
            md += f"\n### {title}\n\n"
            for point in points:
                md += f"- {point}\n"

        md += f"""
---

## Conversations ({stats['conversations']})

| Title | Messages | Cost | Processed |
|-------|----------|------|-----------|
"""
        for conv in conversations:
            title = (conv['title'] or 'Untitled')[:40]
            md += f"| {title} | {conv['message_count']} | ${conv['cost_usd'] or 0:.4f} | {conv['processed_at'][:10] if conv['processed_at'] else 'No'} |\n"

        md += "\n\n---\n\n*Generated by FS-RLM Extractor*\n"

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md)

        print(f"\n Report saved: {output_path}")
        return output_path

    def close(self):
        """Close database connection."""
        self.db.close()


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=" File-System RLM Extractor for ChatGPT Conversations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all exports in a directory
  python fs_extractor.py /path/to/chatgpt-exports/

  # Use specific model
  python fs_extractor.py /path/to/exports/ --model sonnet

  # Verbatim only (no RLM processing)
  python fs_extractor.py /path/to/exports/ --verbatim-only

  # Query the database
  python fs_extractor.py --query "action items"
  python fs_extractor.py --list-conversations
  python fs_extractor.py --all-actions
  python fs_extractor.py --report

Models:
  haiku   - Claude Haiku 4.5 (fastest, cheapest)
  sonnet  - Claude Sonnet 4.5 (balanced)
  opus    - Claude Opus 4.5 (most capable)
        """
    )

    parser.add_argument("path", nargs="?", help="Path to ChatGPT export file or directory")
    parser.add_argument("--model", "-m", default="haiku", help="Model to use (default: haiku)")
    parser.add_argument("--db", default="conversations.db", help="Database file (default: conversations.db)")
    parser.add_argument("--max-iterations", type=int, default=10, help="Max RLM iterations")
    parser.add_argument("--budget", type=float, default=0.50, help="Max cost per conversation")
    parser.add_argument("--verbatim-only", action="store_true", help="Only extract verbatim (no RLM)")
    parser.add_argument("--logging", action="store_true", help="Enable detailed logging")

    # Query options
    parser.add_argument("--query", "-q", help="Search messages for a term")
    parser.add_argument("--list-conversations", action="store_true", help="List all conversations")
    parser.add_argument("--all-actions", action="store_true", help="Show all action items")
    parser.add_argument("--all-keypoints", action="store_true", help="Show all key points")
    parser.add_argument("--report", action="store_true", help="Generate markdown report")
    parser.add_argument("--stats", action="store_true", help="Show database statistics")

    args = parser.parse_args()

    # Handle query modes (no processing needed)
    if args.query or args.list_conversations or args.all_actions or args.all_keypoints or args.report or args.stats:
        db = ConversationDB(args.db)

        if args.stats:
            stats = db.get_stats()
            print(f"\n Database: {args.db}")
            print("=" * 40)
            for key, value in stats.items():
                if 'cost' in key:
                    print(f"{key}: ${value:.4f}")
                else:
                    print(f"{key}: {value}")

        if args.list_conversations:
            print(f"\n Conversations in {args.db}:")
            print("=" * 60)
            for conv in db.list_conversations():
                title = (conv['title'] or 'Untitled')[:40]
                processed = "Yes" if conv['processed_at'] else "No"
                print(f"  [{conv['id'][:8]}] {title} ({conv['message_count']} msgs, processed: {processed})")

        if args.query:
            print(f"\n Search results for: '{args.query}'")
            print("=" * 60)
            results = db.search_messages(args.query)
            for r in results[:20]:
                print(f"\n  [{r['role'].upper()}] in {r['conversation_title'][:30]}:")
                content = r['content'][:200] + "..." if len(r['content']) > 200 else r['content']
                print(f"    {content}")

        if args.all_actions:
            print(f"\n All Action Items:")
            print("=" * 60)
            for item in db.get_all_action_items():
                status = "[x]" if item['completed'] else "[ ]"
                print(f"  {status} {item['item']}")
                print(f"      from: {item['conversation_title'][:40]}")

        if args.all_keypoints:
            print(f"\n All Key Points:")
            print("=" * 60)
            for kp in db.get_all_key_points()[:50]:
                print(f"  - {kp['point']}")
                print(f"    from: {kp['conversation_title'][:40]}")

        if args.report:
            extractor = FSExtractor(db_path=args.db, verbatim_only=True)
            extractor.generate_report()
            extractor.close()

        db.close()
        return

    # Process mode - need a path
    if not args.path:
        parser.print_help()
        print("\n Error: Please provide a path to process or use a query option.")
        sys.exit(1)

    # Check API key if not verbatim-only
    if not args.verbatim_only and not os.getenv("ANTHROPIC_API_KEY"):
        print(" ERROR: ANTHROPIC_API_KEY not set!")
        print("   Run: export ANTHROPIC_API_KEY='sk-ant-...'")
        print("   Or use --verbatim-only to skip RLM processing")
        sys.exit(1)

    # Process
    extractor = FSExtractor(
        db_path=args.db,
        model=args.model,
        max_iterations=args.max_iterations,
        budget_limit=args.budget,
        enable_logging=args.logging,
        verbatim_only=args.verbatim_only,
    )

    try:
        extractor.process_path(args.path)
        extractor.generate_report()
    finally:
        extractor.close()


if __name__ == "__main__":
    main()
