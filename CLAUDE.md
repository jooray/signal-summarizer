# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Run Commands

```bash
# Install dependencies
pip install poetry
poetry install

# Run the summarizer
poetry run summarize_signal_group --group GROUP_ID

# List available groups
poetry run summarize_signal_group --list-groups

# Common options
poetry run summarize_signal_group --group GROUP_ID --last-week
poetry run summarize_signal_group --group GROUP_ID --since 2024-01-01 --until 2024-01-31
poetry run summarize_signal_group --group GROUP_ID --resume-file resume.json
poetry run summarize_signal_group --group GROUP_ID --redo-themes
poetry run summarize_signal_group --group GROUP_ID --redo-merging

# Import messages from Signal export
python import_messages.py <source_db> <destination_db>
```

## Architecture Overview

This tool summarizes Signal group messages using LLMs. The pipeline:

1. **Message fetching** (`group_summarizer.py:fetch_messages`) - Reads messages from SQLite database
2. **Attachment processing** (`group_summarizer.py:process_attachments`) - Describes images via vision model, transcribes audio via Whisper
3. **Link processing** (`group_summarizer.py:process_links`) - Summarizes shared URLs and YouTube videos
4. **Theme extraction** (`group_summarizer.py:generate_themes`) - Splits conversation into chunks, extracts themes from each
5. **Theme recombination** (`group_summarizer.py:recombine_themes`) - Merges similar themes using either:
   - Embedding-based DBSCAN clustering (`cluster.py`) followed by LLM similarity verification
   - Iterative random-batch merging with LLM similarity prompts
6. **Translation** - Translates final summary if target language differs from English
7. **Output** - Writes Markdown summary

### Key Components

- **`summarize_signal_group.py`** - CLI entry point, argument parsing, orchestrates the summarization
- **`group_summarizer.py`** - Core summarization logic, theme extraction/merging, link handling
- **`llm_util.py`** - LLM abstraction layer supporting Ollama, OpenAI, and Venice providers. Contains Pydantic models for structured output (`ConversationTheme`, `SimilarityGroup`, etc.)
- **`cluster.py`** - DBSCAN clustering of themes using Ollama embeddings
- **`vision_util.py`** - Image description via Ollama vision models
- **`ollama_client.py`** - Low-level Ollama API wrapper for vision tasks
- **`resume_util.py`** - Resume file persistence for long-running jobs

### Configuration

All settings in `config.json` (copy from `config.json.sample`). Key structure:
- `defaults` - Global settings (database path, models, prompts)
- `groups` - Per-group overrides (group_description, language, output_file)
- `models` - LLM provider configs with `provider` field (`ollama`/`openai`/`venice`)
- `embedding_clustering` - Optional DBSCAN settings (`enabled`, `model`, `eps`, `min_samples`)

### Database Schema

Messages table: `id, source, sourceName, timestamp, message, groupId, groupName, attachmentPaths, attachmentDescriptions, quoteText`

Links table: `url, group_id, timestamp, title, summary, error`
