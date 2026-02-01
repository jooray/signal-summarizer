# Signal Group Summarizer

A tool to summarize messages from Signal groups using large language models (LLMs). The script processes messages stored in an SQLite database, handles attachments (images and audio), extracts and summarizes links, and generates a summary in Markdown format.

## It is a bit clever 😅

It can use fully local models. For understanding images, it uses multimodal models through [ollama](https://ollama.com/).

For voice recongition (voice messages), it uses whisper and then LLM.

It generates topics from LLMs, downloads webpages to describe them, then uses embedding clusters
to detect similar themes, merge them (we can be way beyond context window and even when we're not,
long texts would usually not work due to problems with keeping attention on many topics at the
same time).

It also uses the LLM for the final translation step. It will understand topics in any language,
but it will internally generate English text, which it can then translate back to a specified
language.

The app can resume operation if stopped.

Use [signal-message-processor](https://github.com/jooray/signal-message-processor) to collect the
messages, it can be run on a different computer than the one that does the summaries.

*Please do not send private chats to corporate AI clouds* - there is a reason why it can run fully
locally.

### But also does not work super well...

If you have improvements of the algorithm, or even the prompts, let me know. The problem with this
approach is it is extremely sensitive to error - either generating too many topics that are too
similar, or grouping many unrelated themes to a few grand themes ("Geopolitic" or
"cryptocurrency").

## Features

- **Summarizes Conversations**: Extracts main themes, recombines them, and translates summaries as needed.
- **Multiple LLM Providers**: Supports Ollama, OpenAI, and Venice providers.
- **Model Configuration**: Easily switch between different models for various tasks.
- **Processes Attachments**:
  - **Images**: Describes images using a vision model via Ollama.
  - **Audio**: Transcribes audio attachments using `pywhispercpp`.
- **Link Summarization**: Extracts and summarizes links, including YouTube videos.
- **Language Translation**: Translates summaries into the desired language.
- **Configurable via JSON**: All settings are easily adjustable through a `config.json` file.
- **Handles Large Prompts**: Splits and recombines messages for efficient processing using LangChain's text splitter.
- **Modular Design**: Utilizes separate components for LLM interactions, vision processing, and speech-to-text.
- **Resume Functionality**: Supports resuming the summarization process from the last successful step using a resume file.
- **Customizable Similarity Threshold**: Allows configuration of the similarity threshold for merging themes during recombination.
- **New Embedding-Based Clustering**: Optionally group themes first with embeddings + clustering before verifying with the existing similarity prompt (reduces the number of LLM calls).
- **Selective Redo Options**: Provides command-line options to redo specific parts of the summarization process.

## Requirements

- **Python**: 3.9 or higher
- **Dependencies**:
  - `pywhispercpp` for audio transcription
  - `langchain` and `langchain_community` for interfacing with LLMs
  - `langchain_openai` for OpenAI interactions
  - `langchain_ollama` for Ollama interactions
  - An instance of Ollama running with the necessary models (if using Ollama)
  - OpenAI-compatible API key and endpoint (if using OpenAI or Venice providers)
  - `requests` and `beautifulsoup4` for fetching and parsing webpage content
  - `colorlog` for colored logging output
  - `scikit-learn` for DBSCAN clustering (if using embedding-based clustering)
  - `numpy` for numeric operations

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/signal-group-summarizer.git
   cd signal-group-summarizer
   ```

2. **Install dependencies using Poetry:**

   ```bash
   pip install poetry
   poetry install
   ```

3. **Configure the tool:**

   - Copy `config.json.sample` to `config.json` and adjust settings as needed.

     ```bash
     cp config.json.sample config.json
     ```

## Usage

### As a Command-Line Tool

#### List Available Groups

```bash
poetry run summarize_signal_group --list-groups
```

#### Generate Summary for Specific Group(s)

To generate a summary for specific group(s), use:

```bash
poetry run summarize_signal_group --group GROUP_ID1 GROUP_ID2
```

If no `--group` is specified, the script will summarize all groups defined in the `config.json`.

#### Specify Output File

```bash
poetry run summarize_signal_group --group GROUP_ID --output summary.md
```

#### Specify Date Range

```bash
poetry run summarize_signal_group --group GROUP_ID --since YYYY-MM-DD --until YYYY-MM-DD
```

#### Summarize Last Week or Month

```bash
poetry run summarize_signal_group --group GROUP_ID --last-week
```

or

```bash
poetry run summarize_signal_group --group GROUP_ID --last-month
```

#### Regenerate Attachment Descriptions

If you want to force regeneration of attachment descriptions (e.g., image descriptions, audio transcriptions), use:

```bash
poetry run summarize_signal_group --group GROUP_ID --regenerate-attachment-descriptions
```

#### Regenerate Link Summaries

If you want to force regeneration of link summaries, use:

```bash
poetry run summarize_signal_group --group GROUP_ID --regenerate-link-summaries
```

#### Redo Specific Steps

To redo specific parts of the summarization process even if they are present in the resume file, use:

- **Redo Themes Generation** (implies `--redo-merging`):

  ```bash
  poetry run summarize_signal_group --group GROUP_ID --redo-themes
  ```

- **Redo Merging of Themes**:

  ```bash
  poetry run summarize_signal_group --group GROUP_ID --redo-merging
  ```

- **Redo Link Processing**:

  ```bash
  poetry run summarize_signal_group --group GROUP_ID --redo-links
  ```

#### Keep Resume File After Successful Processing

By default, the resume file is deleted after successful processing. To keep the resume file, use:

```bash
poetry run summarize_signal_group --group GROUP_ID --keep-resume-file
```

#### Resuming from a Previous Run

If you are processing large groups or expect the summarization process to take a significant amount of time, you can use a resume file to save progress and resume in case of interruptions or failures.

##### Using a Resume File

Specify a path for the resume file using the `--resume-file` command-line argument. The summarizer will save its progress to this file after each major step. If the summarizer is interrupted, you can rerun the command with the same resume file to continue from where it left off.

```bash
poetry run summarize_signal_group --group GROUP_ID --resume-file resume.json
```

##### Resume File Configuration

You can also specify a default resume file in the `config.json` under the `"resume_file"` key in the `"defaults"` section. If a resume file is specified both in the command line and in the config file, the command line argument takes precedence.

```json
{
  "defaults": {
    "resume_file": "resume.json",
    // ... other defaults ...
  },
  // ... rest of the config ...
}
```

##### Behavior of the Resume File

- **Automatic Deletion**: When the summarization process completes successfully for all specified groups, the resume file will be automatically deleted unless `--keep-resume-file` is used.
- **Group-Specific Progress**: The resume file tracks progress per group. If you are summarizing multiple groups and some are fully processed while others are not, the summarizer will skip the fully processed groups on the next run.
- **No Resume File by Default**: If you do not specify a resume file, the summarizer will not save progress between runs.

### Embedding-Based Clustering of Themes

Optionally, you can enable an **embedding-based clustering** step to reduce the number of LLM calls and better group similar themes before verifying them with the existing similarity prompt.

The tool supports three clustering algorithms for grouping similar themes:

| Algorithm | Best For | Tuning Required |
|-----------|----------|-----------------|
| **HDBSCAN** (default) | General use | None |
| **DBSCAN** | Known density | Yes (`eps`) |
| **Louvain** | Large datasets | Minimal |

**Recommendation**: Use HDBSCAN (default) - it automatically handles varying density without parameter tuning and produces better semantic groupings.

Example `config.json` snippet:

```json
"embedding_clustering": {
  "enabled": true,
  "model": "mxbai-embed-large",
  "method": "hdbscan",
  "min_cluster_size": 2,
  "hdbscan_min_samples": 1
}
```

#### Algorithm-Specific Parameters

**HDBSCAN** (recommended):
- `min_cluster_size`: 2-3 for small datasets, 3-5 for large
- `hdbscan_min_samples`: 1 for sensitivity, 2 for stricter

**DBSCAN**:
- `eps`: 0.2-0.5 (requires tuning per dataset)
- `min_samples`: typically 2

**Louvain**:
- `similarity_threshold`: 0.4-0.6 (higher = tighter clusters)
- `resolution`: <1.0 fewer clusters, >1.0 more clusters

## Configuration

The tool is configured via a `config.json` file. You can specify defaults and override settings for specific groups. Key configuration options include:

- **models**: Defines LLM models with their providers and configurations.
- **themes**: Configuration for extracting main themes from conversations.
- **themes_recombination**: Configuration for recombining extracted themes.
- **translation**: Configuration for translating summaries into the desired language.
- **link_summarizer**: Configuration for summarizing links shared in the conversation.
- **vision_config**: Settings for the vision model used for image descriptions.
- **speech_to_text**: Settings for the speech-to-text model.
- **database**: Path to the SQLite database containing messages.
- **attachment_root**: Path to the directory containing attachments.
- **output_file**: Default output file for summaries.
- **language**: The language to ensure in the summaries.
- **resume_file**: Path to the resume file for saving progress (optional).
- **embedding_clustering**: (Optional) Settings for embedding-based clustering of themes (supports HDBSCAN, DBSCAN, and Louvain methods).

### Configuration Breakdown

- **themes_recombination**: Contains settings for the recombination of themes using an LLM-based similarity prompt.

  - **similarity_threshold**: Specifies the minimum similarity rating required to merge themes during the recombination phase. The default value is 4. The similarity rating ranges from 1 to 6, where:

    - **1** - No similarity  
    - **2** - Small similarity  
    - **3** - Broad subset similarity  
    - **4** - Narrow subset similarity  
    - **5** - Topics are very similar  
    - **6** - Topics are basically the same  

    Only themes with a similarity rating equal to or above the `similarity_threshold` will be merged.

- **embedding_clustering**:
  - **enabled**: If set to true, the tool will first cluster extracted themes with an embedding model.
  - **model**: The Ollama embedding model used for obtaining embeddings.
  - **method**: Clustering algorithm to use: `hdbscan` (default), `dbscan`, or `louvain`.
  - **min_cluster_size**: (HDBSCAN) Minimum cluster size (default: 2).
  - **hdbscan_min_samples**: (HDBSCAN) Minimum samples for core points (default: 1).
  - **eps**: (DBSCAN) Maximum distance between samples in the same neighborhood.
  - **min_samples**: (DBSCAN) Minimum samples for a core point.
  - **similarity_threshold**: (Louvain) Minimum similarity for edge creation (default: 0.5).
  - **resolution**: (Louvain) Resolution parameter for community detection (default: 1.0).

- **models**: Define all LLM models you intend to use, specifying their providers (`ollama`, `venice`, or `openai`), endpoints, and necessary credentials.

- **Other configurations**: As described in the initial documentation.

## Notes

- **LLM Providers**:
  - **Ollama**: Ensure that the Ollama server is running and accessible at the endpoint specified in the configuration. Make sure that the necessary models are installed in Ollama.
  - **OpenAI**: Ensure that you have a valid OpenAI API key and that the `apiBase` is correctly set if using a custom endpoint. But even better - don't use a 3rd party API.
  - **Venice**: Uses OpenAI-compatible API of venice.ai, but removes the "n" parameter which is not supported by Venice. But even better - don't use a 3rd party API.

- **Attachments**: Ensure that attachments (images, audio files) are stored and accessible as per the database entries.

- **Language Support**: The tool supports summarizing conversations in multiple languages. Ensure that the `language` field in each group configuration is set appropriately.

- **Security**: Keep your `config.json` secure, especially if it contains sensitive information like API keys. Avoid committing it to version control systems.

- **Error Handling**: The tool logs warnings and errors during processing. Ensure you monitor the logs to address any issues that arise.

## Support and value4value

If you like this project, I would appreciate if you contributed time, talent or treasure.

Time and talent can be used in testing it out, fixing bugs or improving the algorithm, or sources
(like SimpleX groups, even the ugly Discord and Telegram can be supported, even though those are
shitcoins of messengers).

Treasure can be [sent back through here](https://juraj.bednar.io/en/support-me/).
