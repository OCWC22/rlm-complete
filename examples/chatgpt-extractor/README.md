# ChatGPT Conversation Extractor

Extract and synthesize ChatGPT conversations using the RLM (Recursive Language Model) framework.

## Features

- **Fetch ChatGPT conversations** from shared links (`chatgpt.com/share/...`)
- **Verbatim extraction** of every message in the conversation
- **Key points synthesis** - distill the conversation into actionable insights
- **Topic identification** - automatically detect what was discussed
- **Code snippet extraction** - pull out all code shared in the conversation
- **Action item extraction** - identify todos and next steps
- **Multiple model support** - use Haiku (fast/cheap), Sonnet, or Opus

## Installation

```bash
cd examples/chatgpt-extractor
pip install -r requirements.txt

# Set your API key
export ANTHROPIC_API_KEY='sk-ant-...'
```

## Usage

### Basic Usage

```bash
# Extract from a ChatGPT share link
python chatgpt_extractor.py "https://chatgpt.com/share/abc123..."

# Results saved to extractions/[timestamp]_[title]/
```

### Model Selection

```bash
# Use Claude Haiku (fastest, cheapest - recommended for most use cases)
python chatgpt_extractor.py "https://chatgpt.com/share/..." --model haiku

# Use Claude Sonnet (balanced)
python chatgpt_extractor.py "https://chatgpt.com/share/..." --model sonnet

# Use Claude Opus (most capable, for complex analysis)
python chatgpt_extractor.py "https://chatgpt.com/share/..." --model opus
```

### From Exported JSON

If you've exported your ChatGPT data (Settings > Data Controls > Export):

```bash
python chatgpt_extractor.py conversations.json --json
```

### Verbatim Only (No AI Synthesis)

Just extract the raw conversation without RLM processing:

```bash
python chatgpt_extractor.py "https://chatgpt.com/share/..." --verbatim-only
```

### Custom Options

```bash
python chatgpt_extractor.py "https://chatgpt.com/share/..." \
  --model haiku \
  --output my_extractions/ \
  --max-iterations 15 \
  --budget 1.00
```

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

The extractor uses the RLM (Recursive Language Model) framework to handle conversations of any length:

1. **Fetch**: Downloads the ChatGPT conversation from the share URL
2. **Parse**: Extracts messages from the HTML/JSON structure
3. **Chunk**: RLM intelligently chunks the conversation
4. **Process**: Sub-LLMs analyze each chunk for key information
5. **Synthesize**: Results are aggregated into a structured extraction

This approach means even very long conversations can be fully processed, as the context is externalized and processed recursively.

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
