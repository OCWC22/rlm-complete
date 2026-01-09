"""
ChatGPT Conversation Extractor with RLM

This tool extracts and synthesizes ChatGPT conversations from shared links.
It uses the RLM (Recursive Language Model) framework to:
1. Fetch the conversation from a ChatGPT shared URL
2. Extract EVERY message verbatim
3. Summarize and synthesize key points
4. Generate structured output

Works with any model: Claude Haiku (fast/cheap), Sonnet, or custom.

Usage:
    python chatgpt_extractor.py "https://chatgpt.com/share/..."
    python chatgpt_extractor.py "https://chatgpt.com/share/..." --model haiku
    python chatgpt_extractor.py "https://chatgpt.com/share/..." --output results/
"""

import os
import sys
import re
import json
import html
import argparse
from datetime import datetime
from typing import Optional, List, Dict, Tuple
from pathlib import Path
from dataclasses import dataclass, field, asdict
from urllib.parse import urlparse
import urllib.request
import urllib.error

# Add the rlm module to path
sys.path.insert(0, str(Path(__file__).parent))

from rlm.rlm_repl import RLM_REPL
from rlm.utils.llm import CostTracker, reset_cost_tracker


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

MODELS = {
    # Anthropic Claude models
    "haiku": "claude-haiku-4-5-20250929",
    "haiku-4": "claude-haiku-4-5-20250929",
    "sonnet": "claude-sonnet-4-5-20250929",
    "sonnet-4": "claude-sonnet-4-5-20250929",
    "opus": "claude-opus-4-5-20251101",
    "opus-4": "claude-opus-4-5-20251101",
    # OpenAI models (if using OpenAI backend)
    "gpt-4": "gpt-4",
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
}


# ============================================================================
# CHATGPT CONVERSATION FETCHING
# ============================================================================

@dataclass
class ChatMessage:
    """A single message in a ChatGPT conversation."""
    role: str  # 'user' or 'assistant'
    content: str
    index: int


@dataclass
class ChatGPTConversation:
    """A complete ChatGPT conversation."""
    url: str
    title: str
    messages: List[ChatMessage]
    fetched_at: str
    raw_html: Optional[str] = None

    def to_text(self) -> str:
        """Convert conversation to plain text format."""
        lines = [
            f"# {self.title}",
            f"URL: {self.url}",
            f"Fetched: {self.fetched_at}",
            f"Messages: {len(self.messages)}",
            "",
            "=" * 60,
            ""
        ]

        for msg in self.messages:
            role_label = "USER" if msg.role == "user" else "ASSISTANT"
            lines.append(f"[{role_label}] (Message #{msg.index})")
            lines.append("-" * 40)
            lines.append(msg.content)
            lines.append("")
            lines.append("=" * 60)
            lines.append("")

        return "\n".join(lines)

    def get_stats(self) -> dict:
        """Get conversation statistics."""
        user_msgs = [m for m in self.messages if m.role == "user"]
        assistant_msgs = [m for m in self.messages if m.role == "assistant"]

        total_chars = sum(len(m.content) for m in self.messages)
        total_words = sum(len(m.content.split()) for m in self.messages)

        return {
            "total_messages": len(self.messages),
            "user_messages": len(user_msgs),
            "assistant_messages": len(assistant_msgs),
            "total_characters": total_chars,
            "total_words": total_words,
        }


class ChatGPTFetcher:
    """Fetches and parses ChatGPT shared conversations."""

    def __init__(self):
        self.user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"

    def fetch(self, url: str) -> ChatGPTConversation:
        """
        Fetch a ChatGPT conversation from a shared URL.

        Supports:
        - https://chat.openai.com/share/[uuid]
        - https://chatgpt.com/share/[uuid]
        """
        print(f"\n Fetching: {url}")

        # Validate URL
        parsed = urlparse(url)
        if not any(domain in parsed.netloc for domain in ['chat.openai.com', 'chatgpt.com']):
            raise ValueError(f"Invalid ChatGPT URL. Expected chat.openai.com or chatgpt.com, got: {parsed.netloc}")

        # Fetch the HTML
        try:
            request = urllib.request.Request(
                url,
                headers={
                    'User-Agent': self.user_agent,
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                }
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                html_content = response.read().decode('utf-8')
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Failed to fetch URL (HTTP {e.code}): {e.reason}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to connect: {e.reason}")

        # Parse the conversation
        conversation = self._parse_html(html_content, url)
        conversation.raw_html = html_content

        print(f"   Title: {conversation.title}")
        print(f"   Messages: {len(conversation.messages)}")

        return conversation

    def _parse_html(self, html_content: str, url: str) -> ChatGPTConversation:
        """Parse ChatGPT conversation from HTML."""

        # Try to extract the JSON data embedded in the page
        # ChatGPT embeds conversation data in a script tag

        # Method 1: Look for __NEXT_DATA__ JSON
        next_data_match = re.search(
            r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>',
            html_content,
            re.DOTALL
        )

        if next_data_match:
            try:
                data = json.loads(next_data_match.group(1))
                return self._parse_next_data(data, url, html_content)
            except json.JSONDecodeError:
                pass

        # Method 2: Look for embedded conversation JSON
        json_matches = re.findall(
            r'(?:window\.__INITIAL_STATE__|"serverResponse")\s*[:=]\s*(\{.*?\});',
            html_content,
            re.DOTALL
        )

        for match in json_matches:
            try:
                data = json.loads(match)
                return self._parse_state_json(data, url)
            except json.JSONDecodeError:
                continue

        # Method 3: Parse HTML structure directly
        return self._parse_html_structure(html_content, url)

    def _parse_next_data(self, data: dict, url: str, raw_html: str) -> ChatGPTConversation:
        """Parse from Next.js __NEXT_DATA__ JSON."""

        # Navigate to the conversation data
        try:
            # The structure varies, try common paths
            props = data.get('props', {})
            page_props = props.get('pageProps', {})

            # Try different data paths
            server_response = page_props.get('serverResponse', {})
            conversation_data = server_response.get('data', page_props.get('data', {}))

            # Get title
            title = (
                conversation_data.get('title') or
                page_props.get('title') or
                'ChatGPT Conversation'
            )

            # Get messages
            messages = []
            linear_conversation = conversation_data.get('linear_conversation', [])
            mapping = conversation_data.get('mapping', {})

            if linear_conversation:
                for i, item in enumerate(linear_conversation):
                    msg_data = item if isinstance(item, dict) else mapping.get(item, {})
                    message = msg_data.get('message', {})
                    if message:
                        role = message.get('author', {}).get('role', 'unknown')
                        content_data = message.get('content', {})
                        parts = content_data.get('parts', [])
                        content = '\n'.join(str(p) for p in parts if p)

                        if role in ['user', 'assistant'] and content.strip():
                            messages.append(ChatMessage(
                                role=role,
                                content=content.strip(),
                                index=len(messages) + 1
                            ))

            elif mapping:
                # Parse from mapping structure
                messages = self._parse_mapping(mapping)

            if not messages:
                # Fallback to HTML parsing
                return self._parse_html_structure(raw_html, url)

            return ChatGPTConversation(
                url=url,
                title=title,
                messages=messages,
                fetched_at=datetime.now().isoformat()
            )

        except Exception as e:
            print(f"   Warning: Failed to parse __NEXT_DATA__: {e}")
            return self._parse_html_structure(raw_html, url)

    def _parse_mapping(self, mapping: dict) -> List[ChatMessage]:
        """Parse messages from ChatGPT mapping structure."""
        messages = []

        # Build a tree and traverse it
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
                    nodes.append((create_time, role, content.strip()))

        # Sort by creation time
        nodes.sort(key=lambda x: x[0])

        for i, (_, role, content) in enumerate(nodes):
            messages.append(ChatMessage(
                role=role,
                content=content,
                index=i + 1
            ))

        return messages

    def _parse_state_json(self, data: dict, url: str) -> ChatGPTConversation:
        """Parse from server state JSON."""
        messages = []
        title = data.get('title', 'ChatGPT Conversation')

        # Try common data structures
        conv_data = data.get('conversation', data)
        items = conv_data.get('messages', conv_data.get('items', []))

        for i, item in enumerate(items):
            role = item.get('role', item.get('author', {}).get('role', 'unknown'))
            content = item.get('content', item.get('text', ''))

            if isinstance(content, dict):
                parts = content.get('parts', [])
                content = '\n'.join(str(p) for p in parts)

            if role in ['user', 'assistant'] and content.strip():
                messages.append(ChatMessage(
                    role=role,
                    content=content.strip(),
                    index=len(messages) + 1
                ))

        return ChatGPTConversation(
            url=url,
            title=title,
            messages=messages,
            fetched_at=datetime.now().isoformat()
        )

    def _parse_html_structure(self, html_content: str, url: str) -> ChatGPTConversation:
        """Parse conversation directly from HTML structure."""
        messages = []

        # Extract title from <title> tag
        title_match = re.search(r'<title>([^<]+)</title>', html_content)
        title = html.unescape(title_match.group(1)) if title_match else 'ChatGPT Conversation'
        title = title.replace(' | ChatGPT', '').strip()

        # Look for message containers
        # ChatGPT uses various class names, try common patterns
        patterns = [
            # User messages
            (r'<div[^>]*data-message-author-role="user"[^>]*>(.*?)</div>', 'user'),
            (r'<div[^>]*class="[^"]*user[^"]*"[^>]*>(.*?)</div>', 'user'),
            # Assistant messages
            (r'<div[^>]*data-message-author-role="assistant"[^>]*>(.*?)</div>', 'assistant'),
            (r'<div[^>]*class="[^"]*assistant[^"]*"[^>]*>(.*?)</div>', 'assistant'),
            # Markdown content
            (r'<div[^>]*class="[^"]*markdown[^"]*"[^>]*>(.*?)</div>', 'assistant'),
        ]

        for pattern, default_role in patterns:
            matches = re.findall(pattern, html_content, re.DOTALL | re.IGNORECASE)
            for match in matches:
                # Clean HTML tags
                content = re.sub(r'<[^>]+>', '', match)
                content = html.unescape(content).strip()

                if content and len(content) > 10:  # Filter noise
                    messages.append(ChatMessage(
                        role=default_role,
                        content=content,
                        index=len(messages) + 1
                    ))

        # Deduplicate (same content might be captured multiple times)
        seen = set()
        unique_messages = []
        for msg in messages:
            key = (msg.role, msg.content[:100])
            if key not in seen:
                seen.add(key)
                msg.index = len(unique_messages) + 1
                unique_messages.append(msg)

        return ChatGPTConversation(
            url=url,
            title=title,
            messages=unique_messages,
            fetched_at=datetime.now().isoformat()
        )

    def from_text(self, text: str, title: str = "ChatGPT Conversation") -> ChatGPTConversation:
        """Load a conversation from raw text (copy-pasted)."""
        print(f"\n Loading from text input...")

        messages = []

        # Try to parse common copy-paste formats
        # Format 1: "You said:" / "ChatGPT said:"
        # Format 2: "User:" / "Assistant:"
        # Format 3: Just alternating blocks

        patterns = [
            (r'(?:You said:|User:)\s*(.*?)(?=(?:ChatGPT said:|Assistant:|You said:|User:|$))', 'user'),
            (r'(?:ChatGPT said:|Assistant:)\s*(.*?)(?=(?:You said:|User:|ChatGPT said:|Assistant:|$))', 'assistant'),
        ]

        for pattern, role in patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            for match in matches:
                content = match.strip()
                if content and len(content) > 5:
                    messages.append(ChatMessage(
                        role=role,
                        content=content,
                        index=len(messages) + 1
                    ))

        # If no structured format found, try to split by double newlines
        if not messages:
            blocks = re.split(r'\n\n+', text)
            for i, block in enumerate(blocks):
                block = block.strip()
                if block and len(block) > 10:
                    # Alternate between user and assistant
                    role = 'user' if i % 2 == 0 else 'assistant'
                    messages.append(ChatMessage(
                        role=role,
                        content=block,
                        index=len(messages) + 1
                    ))

        print(f"   Title: {title}")
        print(f"   Messages: {len(messages)}")

        return ChatGPTConversation(
            url="text://manual-input",
            title=title,
            messages=messages,
            fetched_at=datetime.now().isoformat()
        )

    def from_json(self, json_path: str) -> ChatGPTConversation:
        """Load a conversation from a JSON file (ChatGPT export format)."""
        print(f"\n Loading from JSON: {json_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Handle both single conversation and export format
        if isinstance(data, list):
            # Export format: list of conversations
            if not data:
                raise ValueError("Empty conversation list")
            data = data[0]  # Use first conversation

        title = data.get('title', 'ChatGPT Conversation')
        mapping = data.get('mapping', {})

        messages = self._parse_mapping(mapping)

        print(f"   Title: {title}")
        print(f"   Messages: {len(messages)}")

        return ChatGPTConversation(
            url=f"file://{json_path}",
            title=title,
            messages=messages,
            fetched_at=datetime.now().isoformat()
        )


# ============================================================================
# RLM-BASED EXTRACTION AND SYNTHESIS
# ============================================================================

@dataclass
class ExtractionResult:
    """Result of extracting and synthesizing a conversation."""
    conversation: ChatGPTConversation
    verbatim_extraction: str
    key_points: List[str]
    summary: str
    topics: List[str]
    action_items: List[str]
    code_snippets: List[str]
    questions_asked: List[str]
    model_used: str
    cost_usd: float
    tokens_used: int
    extraction_time: str


class ConversationExtractor:
    """
    Uses RLM to extract and synthesize ChatGPT conversations.

    The RLM approach allows processing of very long conversations
    by chunking and recursively analyzing with sub-LLMs.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20250929",
        max_iterations: int = 10,
        budget_limit: float = 0.50,
        enable_logging: bool = True,
    ):
        self.model = MODELS.get(model, model)  # Resolve model alias
        self.max_iterations = max_iterations
        self.budget_limit = budget_limit
        self.enable_logging = enable_logging

    def _create_rlm(self) -> RLM_REPL:
        """Create a fresh RLM instance."""
        return RLM_REPL(
            model=self.model,
            recursive_model=self.model,
            provider="anthropic",
            enable_logging=self.enable_logging,
            max_iterations=self.max_iterations,
            budget_limit=self.budget_limit,
        )

    def extract(self, conversation: ChatGPTConversation) -> ExtractionResult:
        """
        Extract and synthesize the conversation using RLM.

        This performs:
        1. Verbatim extraction of all messages
        2. Key point synthesis
        3. Topic identification
        4. Action item extraction
        5. Code snippet extraction
        """
        print("\n" + "=" * 60)
        print(" EXTRACTING CONVERSATION WITH RLM")
        print("=" * 60)
        print(f"Title: {conversation.title}")
        print(f"Messages: {len(conversation.messages)}")
        print(f"Model: {self.model}")
        print("=" * 60)

        # Prepare the context
        context = self._prepare_context(conversation)

        # Create extraction query
        query = self._create_extraction_query()

        # Reset cost tracker and create RLM
        reset_cost_tracker()
        rlm = self._create_rlm()

        # Run extraction
        start_time = datetime.now()
        try:
            raw_result = rlm.completion(context=context, query=query)
        except RuntimeError as e:
            if "Budget" in str(e):
                raw_result = f"[Budget exceeded] {str(e)}"
            else:
                raise

        end_time = datetime.now()

        # Get cost summary
        cost_summary = rlm.cost_summary()

        # Parse the result
        result = self._parse_result(
            raw_result,
            conversation,
            cost_summary,
            start_time.isoformat()
        )

        # Print summary
        self._print_summary(result, cost_summary)

        return result

    def _prepare_context(self, conversation: ChatGPTConversation) -> str:
        """Prepare the conversation as context for RLM."""
        stats = conversation.get_stats()

        header = f"""==============================================================================
CHATGPT CONVERSATION DATA
==============================================================================
TITLE: {conversation.title}
URL: {conversation.url}
FETCHED: {conversation.fetched_at}
TOTAL_MESSAGES: {stats['total_messages']}
USER_MESSAGES: {stats['user_messages']}
ASSISTANT_MESSAGES: {stats['assistant_messages']}
TOTAL_WORDS: {stats['total_words']}
TOTAL_CHARACTERS: {stats['total_characters']}
==============================================================================

"""
        return header + conversation.to_text()

    def _create_extraction_query(self) -> str:
        """Create the extraction query for RLM."""
        return """Analyze this ChatGPT conversation and provide a COMPREHENSIVE extraction with:

1. **VERBATIM EXTRACTION**: Go through EVERY single message in the conversation and extract the EXACT content verbatim. Do not summarize or paraphrase - copy the exact text.

2. **KEY POINTS**: Synthesize the conversation into 5-15 key points that capture the essential information, insights, and conclusions discussed.

3. **TOPICS**: List all the main topics/subjects discussed in the conversation.

4. **ACTION ITEMS**: Extract any action items, to-dos, or next steps mentioned.

5. **CODE SNIPPETS**: If any code was shared, extract all code snippets with their context.

6. **QUESTIONS ASKED**: List all questions the user asked.

Format your response as a structured document with clear sections for each of the above. Be thorough - the goal is to capture EVERYTHING discussed in this conversation so nothing is lost.

Start by examining the full conversation, then chunk it intelligently if needed, and compile the complete extraction."""

    def _parse_result(
        self,
        raw_result: str,
        conversation: ChatGPTConversation,
        cost_summary: dict,
        extraction_time: str
    ) -> ExtractionResult:
        """Parse the RLM result into structured format."""

        # Extract sections from the result
        key_points = self._extract_list_section(raw_result, r"KEY POINTS?:?\n(.*?)(?=\n\n[A-Z#*]|\Z)")
        topics = self._extract_list_section(raw_result, r"TOPICS?:?\n(.*?)(?=\n\n[A-Z#*]|\Z)")
        action_items = self._extract_list_section(raw_result, r"ACTION ITEMS?:?\n(.*?)(?=\n\n[A-Z#*]|\Z)")
        questions = self._extract_list_section(raw_result, r"QUESTIONS? ASKED:?\n(.*?)(?=\n\n[A-Z#*]|\Z)")

        # Extract code snippets
        code_snippets = re.findall(r'```(?:\w+)?\n(.*?)```', raw_result, re.DOTALL)

        # Extract verbatim section (usually the largest section)
        verbatim_match = re.search(
            r'VERBATIM EXTRACTION:?\n(.*?)(?=\n\n(?:KEY POINTS?|TOPICS?|#)|\Z)',
            raw_result,
            re.DOTALL | re.IGNORECASE
        )
        verbatim = verbatim_match.group(1).strip() if verbatim_match else raw_result

        # Create summary from key points
        summary = " ".join(key_points[:5]) if key_points else "No summary generated."

        return ExtractionResult(
            conversation=conversation,
            verbatim_extraction=verbatim,
            key_points=key_points,
            summary=summary,
            topics=topics,
            action_items=action_items,
            code_snippets=code_snippets,
            questions_asked=questions,
            model_used=self.model,
            cost_usd=cost_summary.get('total_cost_usd', 0),
            tokens_used=cost_summary.get('total_tokens', 0),
            extraction_time=extraction_time
        )

    def _extract_list_section(self, text: str, pattern: str) -> List[str]:
        """Extract a bulleted/numbered list from text."""
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if not match:
            return []

        section = match.group(1)

        # Parse list items (bullet points, numbers, dashes)
        items = re.findall(r'(?:^|\n)\s*(?:[-*\d.]+)\s*(.+?)(?=\n|$)', section)

        return [item.strip() for item in items if item.strip()]

    def _print_summary(self, result: ExtractionResult, cost_summary: dict):
        """Print extraction summary."""
        print("\n" + "-" * 60)
        print(" EXTRACTION COMPLETE")
        print("-" * 60)
        print(f"Key Points: {len(result.key_points)}")
        print(f"Topics: {len(result.topics)}")
        print(f"Action Items: {len(result.action_items)}")
        print(f"Code Snippets: {len(result.code_snippets)}")
        print(f"Questions: {len(result.questions_asked)}")
        print("-" * 60)
        print(f"Tokens Used: {result.tokens_used:,}")
        print(f"Cost: ${result.cost_usd:.4f}")
        print("-" * 60)


# ============================================================================
# OUTPUT GENERATION
# ============================================================================

class OutputGenerator:
    """Generates various output formats from extraction results."""

    def __init__(self, output_dir: str = "extractions"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, result: ExtractionResult) -> dict:
        """Save extraction results to multiple formats."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        title_slug = self._slugify(result.conversation.title)
        base_name = f"{timestamp}_{title_slug}"

        # Create output directory for this extraction
        extraction_dir = self.output_dir / base_name
        extraction_dir.mkdir(parents=True, exist_ok=True)

        # Save files
        files = {}

        # 1. Full markdown report
        md_path = extraction_dir / "full_extraction.md"
        md_content = self._generate_markdown(result)
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        files['markdown'] = str(md_path)

        # 2. Key points summary
        summary_path = extraction_dir / "key_points.md"
        summary_content = self._generate_summary(result)
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_content)
        files['summary'] = str(summary_path)

        # 3. Verbatim transcript
        verbatim_path = extraction_dir / "verbatim.txt"
        with open(verbatim_path, 'w', encoding='utf-8') as f:
            f.write(result.conversation.to_text())
        files['verbatim'] = str(verbatim_path)

        # 4. JSON data
        json_path = extraction_dir / "extraction.json"
        json_data = self._to_json(result)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2)
        files['json'] = str(json_path)

        # 5. Code snippets (if any)
        if result.code_snippets:
            code_path = extraction_dir / "code_snippets.md"
            code_content = self._generate_code_file(result)
            with open(code_path, 'w', encoding='utf-8') as f:
                f.write(code_content)
            files['code'] = str(code_path)

        print(f"\n Saved to: {extraction_dir}/")
        for name, path in files.items():
            print(f"   - {Path(path).name}")

        return files

    def _slugify(self, text: str) -> str:
        """Convert text to a safe filename."""
        text = re.sub(r'[^\w\s-]', '', text.lower())
        text = re.sub(r'[\s_]+', '-', text)
        return text[:50]

    def _generate_markdown(self, result: ExtractionResult) -> str:
        """Generate full markdown report."""
        conv = result.conversation
        stats = conv.get_stats()

        md = f"""# {conv.title}

> **Extracted:** {result.extraction_time}
> **URL:** {conv.url}
> **Model:** `{result.model_used}`

---

## Conversation Statistics

| Metric | Value |
|--------|-------|
| Total Messages | {stats['total_messages']} |
| User Messages | {stats['user_messages']} |
| Assistant Messages | {stats['assistant_messages']} |
| Total Words | {stats['total_words']:,} |
| Extraction Cost | ${result.cost_usd:.4f} |
| Tokens Used | {result.tokens_used:,} |

---

## Summary

{result.summary}

---

## Key Points

"""
        for i, point in enumerate(result.key_points, 1):
            md += f"{i}. {point}\n"

        md += "\n---\n\n## Topics Discussed\n\n"
        for topic in result.topics:
            md += f"- {topic}\n"

        if result.action_items:
            md += "\n---\n\n## Action Items\n\n"
            for item in result.action_items:
                md += f"- [ ] {item}\n"

        if result.questions_asked:
            md += "\n---\n\n## Questions Asked\n\n"
            for q in result.questions_asked:
                md += f"- {q}\n"

        if result.code_snippets:
            md += "\n---\n\n## Code Snippets\n\n"
            for i, code in enumerate(result.code_snippets, 1):
                md += f"### Snippet {i}\n\n```\n{code}\n```\n\n"

        md += "\n---\n\n## Full Verbatim Extraction\n\n"
        md += result.verbatim_extraction

        md += "\n\n---\n\n*Extracted using RLM ChatGPT Extractor*\n"

        return md

    def _generate_summary(self, result: ExtractionResult) -> str:
        """Generate key points summary."""
        md = f"""# Key Points: {result.conversation.title}

> Extracted: {result.extraction_time}

## Summary

{result.summary}

## Key Points

"""
        for i, point in enumerate(result.key_points, 1):
            md += f"{i}. {point}\n"

        if result.action_items:
            md += "\n## Action Items\n\n"
            for item in result.action_items:
                md += f"- [ ] {item}\n"

        return md

    def _generate_code_file(self, result: ExtractionResult) -> str:
        """Generate code snippets file."""
        md = f"# Code Snippets from: {result.conversation.title}\n\n"

        for i, code in enumerate(result.code_snippets, 1):
            md += f"## Snippet {i}\n\n```\n{code}\n```\n\n---\n\n"

        return md

    def _to_json(self, result: ExtractionResult) -> dict:
        """Convert result to JSON-serializable dict."""
        conv = result.conversation
        return {
            "title": conv.title,
            "url": conv.url,
            "fetched_at": conv.fetched_at,
            "extraction_time": result.extraction_time,
            "model_used": result.model_used,
            "cost_usd": result.cost_usd,
            "tokens_used": result.tokens_used,
            "statistics": conv.get_stats(),
            "summary": result.summary,
            "key_points": result.key_points,
            "topics": result.topics,
            "action_items": result.action_items,
            "questions_asked": result.questions_asked,
            "code_snippets": result.code_snippets,
            "messages": [
                {
                    "index": m.index,
                    "role": m.role,
                    "content": m.content
                }
                for m in conv.messages
            ]
        }


# ============================================================================
# MAIN CLI
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description=" ChatGPT Conversation Extractor with RLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract from a ChatGPT share link
  python chatgpt_extractor.py "https://chatgpt.com/share/abc123..."

  # Use a faster/cheaper model (Haiku)
  python chatgpt_extractor.py "https://chatgpt.com/share/..." --model haiku

  # Load from exported JSON file
  python chatgpt_extractor.py conversations.json --json

  # Custom output directory
  python chatgpt_extractor.py "https://chatgpt.com/share/..." --output my_extractions/

Models available:
  haiku     - Claude Haiku 4.5 (fastest, cheapest)
  sonnet    - Claude Sonnet 4.5 (balanced)
  opus      - Claude Opus 4.5 (most capable)
        """
    )

    parser.add_argument("source", help="ChatGPT share URL or path to JSON export")
    parser.add_argument("--json", action="store_true",
                        help="Source is a JSON file (ChatGPT export format)")
    parser.add_argument("--model", "-m", default="haiku",
                        help="Model to use (default: haiku)")
    parser.add_argument("--output", "-o", default="extractions",
                        help="Output directory (default: extractions)")
    parser.add_argument("--max-iterations", type=int, default=10,
                        help="Max RLM iterations (default: 10)")
    parser.add_argument("--budget", type=float, default=0.50,
                        help="Max cost per extraction (default: $0.50)")
    parser.add_argument("--no-logging", action="store_true",
                        help="Disable detailed logging")
    parser.add_argument("--verbatim-only", action="store_true",
                        help="Only extract verbatim content (no synthesis)")

    args = parser.parse_args()

    # Fetch the conversation first (doesn't need API key)
    fetcher = ChatGPTFetcher()

    if args.json:
        conversation = fetcher.from_json(args.source)
    else:
        conversation = fetcher.fetch(args.source)

    # Handle verbatim-only mode
    if args.verbatim_only:
        print("\n Verbatim extraction mode (no RLM synthesis)")
        output = OutputGenerator(args.output)

        # Create a minimal result
        result = ExtractionResult(
            conversation=conversation,
            verbatim_extraction=conversation.to_text(),
            key_points=[],
            summary="Verbatim extraction only - no synthesis performed.",
            topics=[],
            action_items=[],
            code_snippets=[],
            questions_asked=[m.content for m in conversation.messages if m.role == 'user'],
            model_used="N/A (verbatim only)",
            cost_usd=0,
            tokens_used=0,
            extraction_time=datetime.now().isoformat()
        )

        files = output.save(result)
        return

    # Check API key (only needed for RLM extraction)
    if not os.getenv("ANTHROPIC_API_KEY"):
        print(" ERROR: ANTHROPIC_API_KEY not set!")
        print("   Run: export ANTHROPIC_API_KEY='sk-ant-...'")
        sys.exit(1)

    # Extract with RLM
    extractor = ConversationExtractor(
        model=args.model,
        max_iterations=args.max_iterations,
        budget_limit=args.budget,
        enable_logging=not args.no_logging,
    )

    result = extractor.extract(conversation)

    # Save results
    output = OutputGenerator(args.output)
    files = output.save(result)

    # Print final summary
    print("\n" + "=" * 60)
    print(" EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Title: {conversation.title}")
    print(f"Messages: {len(conversation.messages)}")
    print(f"Key Points: {len(result.key_points)}")
    print(f"Cost: ${result.cost_usd:.4f}")
    print(f"Output: {args.output}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
