# File: group_summarizer.py

import datetime
import json
import logging
import re
import sqlite3
import warnings
from pathlib import Path
from typing import List

import requests
from langchain_unstructured import UnstructuredLoader

from llm_util import LLMUtil, ConversationThemes, ConversationTheme, SimilarityGroups
import random
from vision_util import VisionUtil
import yt_dlp
from resume_util import load_resume_file, save_resume_file, delete_resume_file

warnings.filterwarnings("ignore", category=UserWarning, module='bs4')  # Suppress BeautifulSoup warnings

def setup_logging(log_level):
    # Configure logging with colors
    import sys
    from colorlog import ColoredFormatter

    log_colors = {
        'DEBUG': 'cyan',
        'INFO': 'white',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }
    formatter = ColoredFormatter(
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        log_colors=log_colors
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_level = getattr(logging, log_level.upper(), None)
    root_logger.setLevel(root_level)
    root_logger.handlers = []  # Clear existing handlers
    root_logger.addHandler(handler)

    for module_logger_name in ['llm_util', 'vision_util', 'ollama_client', 'resume_util']:
        module_logger = logging.getLogger(module_logger_name)
        module_logger.setLevel(root_level)
        module_logger.propagate = True

def load_config(config_file_path):
    with open(config_file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_group_config(config, group_id):
    from collections.abc import Mapping

    def deep_merge(source, overrides):
        for key, value in overrides.items():
            if isinstance(value, Mapping) and key in source:
                source[key] = deep_merge(source.get(key, {}), value)
            else:
                source[key] = value
        return source

    defaults = config.get('defaults', {})
    group_config = config.get('groups', {}).get(group_id, {})
    merged_config = deep_merge(defaults.copy(), group_config)
    return merged_config

def get_date_range(args):
    if args.since:
        since = datetime.datetime.strptime(args.since, '%Y-%m-%d')
    elif args.last_week:
        since = datetime.datetime.now() - datetime.timedelta(days=7)
    elif args.last_month:
        since = datetime.datetime.now() - datetime.timedelta(weeks=4)
    else:
        since = None  # No lower limit

    if args.until:
        until = datetime.datetime.strptime(args.until, '%Y-%m-%d')
    else:
        until = datetime.datetime.now()

    return since, until

def fetch_messages(database, group_id, since, until):
    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    query = '''SELECT id, source, sourceName, timestamp, message, attachmentPaths, attachmentDescriptions, quoteText FROM messages WHERE groupId = ?'''
    params = [group_id]
    if since:
        query += ' AND timestamp >= ?'
        params.append(int(since.timestamp() * 1000))  # Assuming timestamp is in milliseconds
    if until:
        query += ' AND timestamp <= ?'
        params.append(int(until.timestamp() * 1000))

    cursor.execute(query, params)
    rows = cursor.fetchall()
    messages = []
    for row in rows:
        msg = {
            'id': row[0],
            'source': row[1],
            'sourceName': row[2],
            'timestamp': row[3],
            'message': row[4],
            'attachmentPaths': json.loads(row[5]) if row[5] else [],
            'attachmentDescriptions': json.loads(row[6]) if row[6] else [],
            'quoteText': row[7]
        }
        messages.append(msg)
    conn.close()
    return messages

def process_attachments(messages, group_config, regenerate, database, attachment_root, vision_util, stt_model):
    conn = sqlite3.connect(database)
    cursor = conn.cursor()

    for msg in messages:
        attachment_descriptions = msg.get('attachmentDescriptions', [])
        if attachment_descriptions and not regenerate:
            continue  # Skip if already processed and not regenerating

        attachment_descriptions = []
        for attachment_path_str in msg['attachmentPaths']:
            attachment_path = attachment_root.joinpath(attachment_path_str)
            if not attachment_path.exists():
                logging.warning(f"Attachment file not found: {attachment_path}")
                continue

            extension = attachment_path.suffix.lower()
            if extension in ['.png', '.jpg', '.jpeg', '.gif', '.webp'] and vision_util is not None:
                vision_prompt_template = group_config['vision_config'].get('prompt')
                prompt = vision_prompt_template.format(
                    group_description=group_config.get('group_description', ''),
                    common_topics=group_config.get('common_topics', ''),
                    language=group_config.get('language', 'English')
                )
                logging.debug(f"Describing image with prompt:\n{prompt}")
                description = vision_util.describe_image(attachment_path, prompt)
                logging.debug(f"Image description:\n{description}")
                attachment_descriptions.append(description)
            elif extension in ['.aac', '.mp3', '.m4a'] and stt_model is not None:
                transcription = transcribe_audio(attachment_path, stt_model)
                logging.debug(f"Audio transcription:\n{transcription}")
                attachment_descriptions.append(transcription)
            else:
                logging.warning(f"Unknown or disabled attachment type: {attachment_path}")
        # Update message with attachment descriptions
        if attachment_descriptions:
            msg['attachmentDescriptions'] = attachment_descriptions
            # Save to database
            cursor.execute('UPDATE messages SET attachmentDescriptions=? WHERE id=?',
                           (json.dumps(attachment_descriptions), msg['id']))
            conn.commit()
    conn.close()

def transcribe_audio(audio_path, stt_model):
    segments = stt_model.transcribe(str(audio_path))
    transcribed_text = ''.join([segment.text for segment in segments]).strip()
    return transcribed_text

def process_links(messages, group_config, regenerate, database, group_id, llm):
    # Create 'links' table if it doesn't exist
    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS links (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT,
            group_id TEXT,
            timestamp INTEGER,
            title TEXT,
            summary TEXT,
            error TEXT,
            UNIQUE(url, group_id, timestamp)
        )
    ''')
    conn.commit()

    link_pattern = re.compile(r'https?://\S+')

    link_summarizer_config = group_config.get('link_summarizer', {})
    prompt_template = link_summarizer_config.get('prompt', '')
    max_title_length = link_summarizer_config.get('max_title_length', 300)
    max_summary_length = link_summarizer_config.get('max_summary_length', 300)
    max_chunk_size = link_summarizer_config.get('max_chunk_size', 5000)

    for msg in messages:
        if not msg.get('message'):
            continue
        urls = link_pattern.findall(msg['message'])
        for url in urls:
            # Check if the link is already summarized unless regenerating
            cursor.execute('SELECT summary FROM links WHERE url = ? AND group_id = ? AND timestamp = ?', (url, group_id, msg['timestamp']))
            result = cursor.fetchone()
            if result and not regenerate:
                continue  # Already summarized

            # Summarize the link
            title, summary, error = summarize_link(url, llm, group_config)

            # Save to 'links' table
            cursor.execute('INSERT OR REPLACE INTO links (url, group_id, timestamp, title, summary, error) VALUES (?, ?, ?, ?, ?, ?)',
                           (url, group_id, msg['timestamp'], title, summary, error))
            conn.commit()
    conn.close()

def summarize_link(url, llm, group_config):
    # Check if YouTube
    if 'youtube.com' in url or 'youtu.be' in url:
        return summarize_youtube_video(url, llm, group_config)
    else:
        return summarize_webpage(url, llm, group_config)

def summarize_webpage(url, llm, group_config):
    link_summarizer_config = group_config.get('link_summarizer', {})
    prompt_template = link_summarizer_config.get('prompt', '')
    max_title_length = link_summarizer_config.get('max_title_length', 300)
    max_summary_length = link_summarizer_config.get('max_summary_length', 300)
    max_chunk_size = link_summarizer_config.get('max_chunk_size', 5000)

    try:
        loader = UnstructuredLoader(
            web_url=url,
            chunking_strategy="basic",
            max_characters=max_chunk_size,
            include_orig_elements=False
        )
        docs = loader.load()

        if not docs or not docs[0].page_content.strip():
            logging.warning(f"Empty content for URL: {url}")
            error = "Empty webpage content"
            return None, None, error

        # Use llm.summarize_link()
        result = llm.summarize_link(
            context=docs[0].page_content,
            prompt_template=prompt_template,
            max_title_length=max_title_length,
            max_summary_length=max_summary_length
        )

        if not result.title:
            result.title = docs[0].metadata.get('filename', url)

        return result.title, result.summary, result.error

    except Exception as e:
        logging.error(f"Failed to fetch webpage {url}: {e}")
        error = f"Failed to fetch webpage: {str(e)}"
        return None, None, error

def summarize_youtube_video(url, llm, group_config):
    # Use yt-dlp to extract video title and description
    link_summarizer_config = group_config.get('link_summarizer', {})
    prompt_template = link_summarizer_config.get('prompt', '')
    max_title_length = link_summarizer_config.get('max_title_length', 300)
    max_summary_length = link_summarizer_config.get('max_summary_length', 300)

    try:
        ydl_opts = {'quiet': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
        title = info.get('title', '')
        description = info.get('description', '')

        if not description.strip():
            logging.warning(f"No description available for YouTube video: {url}")
            error = "No description available"
            return title, None, error

        result = llm.summarize_link(
            context=f"Video: {title}\n{description}",
            prompt_template=prompt_template,
            max_title_length=max_title_length,
            max_summary_length=max_summary_length
        )

        # Use the video's title if available
        if not result.title and title:
            result.title = title

        return result.title, result.summary, result.error

    except Exception as e:
        logging.error(f"Failed to load YouTube video {url}: {e}")
        error = f"Failed to load YouTube video: {str(e)}"
        return None, None, error

def build_links_section(database, group_config, group_id, since, until):
    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    params = [group_id]
    query = 'SELECT url, title, summary, error FROM links WHERE group_id = ?'

    if since:
        query += ' AND timestamp >= ?'
        params.append(int(since.timestamp() * 1000))
    if until:
        query += ' AND timestamp <= ?'
        params.append(int(until.timestamp() * 1000))

    cursor.execute(query, params)
    links = cursor.fetchall()
    conn.close()

    if not links:
        return ""

    links_section = "## Links shared with the group\n"
    for url, title, summary, error in links:
        if not title:
            title = url
        if error:
            links_section += f"- [{url}]({url}): ({error})\n"
        elif summary:
            links_section += f"- [{title}]({url}): {summary}\n"
        else:
            links_section += f"- [{title}]({url})\n"
    return links_section

def build_conversation_chunks(messages, group_config, llm_dict):
    themes_config = group_config.get('themes', {})
    max_chunk_size = themes_config.get('max_chunk_size', 5000)

    messages_texts = []
    for msg in messages:
        timestamp = datetime.datetime.fromtimestamp(msg['timestamp'] / 1000).strftime('%Y-%m-%d %H:%M')
        text_parts = [f"[{timestamp}] {msg['sourceName']}: {msg['message']}"]
        if msg.get('quoteText'):
            text_parts.append(f"(In reply to: \"{msg['quoteText']}\")")
        if msg.get('attachmentDescriptions'):
            text_parts.extend(msg['attachmentDescriptions'])
        messages_texts.append('\n'.join(text_parts))

    combined_text = '\n'.join(messages_texts)
    themes_model_name = themes_config.get('model')
    themes_llm = llm_dict.get(themes_model_name)
    return themes_llm.split_text(combined_text, max_chunk_size)

def generate_themes(chunk, group_config, llm):
    themes_config = group_config.get('themes', {})
    themes_prompt_template = themes_config.get('prompt', '')
    # Generate structured themes
    try:
        logging.debug("Generating themes with prompt:")
        logging.debug(f"Prompt Template:\n{themes_prompt_template}")
        logging.debug(f"Context:\n{chunk}")
        themes = llm.generate_structured_output(
            prompt_template=themes_prompt_template,
            context=chunk,
            pydantic_class=ConversationThemes
        )
        logging.debug(f"Generated themes:\n{themes}")
        return themes
    except Exception as e:
        logging.error(f"Failed to generate themes: {e}")
        return None

def embedding_cluster_and_merge(themes_list, group_config, similarity_prompt_template, merging_prompt_template, llm):
    """
    First cluster the themes using embeddings and DBSCAN, then run the standard
    similarity-based merging on each cluster.
    """
    from cluster import cluster_themes_with_embeddings

    # Flatten all themes
    all_themes = []
    for tset in themes_list:
        if tset and tset.themes:
            all_themes.extend(tset.themes)

    if not all_themes:
        return None

    # Perform embedding-based clustering
    clustering_config = group_config.get("embedding_clustering", {})
    model_name = clustering_config.get("model", "mxbai-embed-large")
    method = clustering_config.get("method", "hdbscan")

    # Get method-specific parameters
    if method == "hdbscan":
        kwargs = {
            "min_cluster_size": clustering_config.get("min_cluster_size", 2),
            "min_samples": clustering_config.get("hdbscan_min_samples", 1)
        }
    elif method == "louvain":
        kwargs = {
            "threshold": clustering_config.get("similarity_threshold", 0.5),
            "resolution": clustering_config.get("resolution", 1.0)
        }
    else:  # dbscan
        kwargs = {
            "eps": clustering_config.get("eps", 0.3),
            "min_samples": clustering_config.get("min_samples", 2)
        }

    logging.info(f"Clustering themes using {method} with params: {kwargs}")

    clusters_indices = cluster_themes_with_embeddings(
        themes=[f"{theme.name}: {theme.summary}" for theme in all_themes],
        model_name=model_name,
        method=method,
        **kwargs
    )
    logging.info(f"Embeddings returned these clusters: {clusters_indices}")


    final_themes = []
    for cluster in clusters_indices:
        # For each cluster, extract the relevant themes
        cluster_themes = [all_themes[idx] for idx in cluster]
        # Then apply the existing similarity-based merges to just this cluster
        merged_cluster_themes = process_theme_batch_with_llm(
            cluster_themes,
            similarity_prompt_template,
            merging_prompt_template,
            llm,
            group_config
        )
        final_themes.extend(merged_cluster_themes)

    # If anything is left out because it's in noise (-1) cluster, add them individually
    # The DBSCAN step in cluster.py already excludes label==-1, but let's do a sanity check:
    used_indices = set()
    for cluster in clusters_indices:
        for idx in cluster:
            used_indices.add(idx)
    for idx, theme in enumerate(all_themes):
        if idx not in used_indices:
            # This is presumably noise or unclustered
            final_themes.append(theme)

    # Return them as a single ConversationThemes object
    return ConversationThemes(themes=final_themes)

def process_theme_batch_with_llm(cluster_themes, similarity_prompt_template, merging_prompt_template, llm, group_config):
    """
    Runs the typical similarity_prompt-based merges on a single cluster of themes.
    Note: This is basically a streamlined version of the batch merging used in the
    older code, but for just one cluster at a time.
    """
    if len(cluster_themes) <= 1:
        return cluster_themes

    # 1. Build a text block
    numbered_themes = [(idx + 1, t) for idx, t in enumerate(cluster_themes)]
    themes_text = ""
    for idx, theme in numbered_themes:
        themes_text += f"{idx}. {capitalize_theme_name(theme.name)}\nSummary: {theme.summary}\n"
        if theme.dissenting_opinions:
            themes_text += f"Dissenting opinions: {theme.dissenting_opinions}\n"
        themes_text += "\n"

    # 2. Generate similarity groups for this cluster
    try:
        similarity_groups = llm.generate_structured_output(
            prompt_template=similarity_prompt_template,
            context=themes_text,
            pydantic_class=SimilarityGroups
        )
        logging.debug(f"Similarity groups for cluster: {similarity_groups.groups}")
    except Exception as e:
        logging.error(f"Failed to identify similar themes in cluster: {e}")
        return cluster_themes  # Return original if error

    # 3. Merge themes based on similarity groups
    merged_cluster_themes, _ = merge_themes_in_cluster(
        cluster_themes, similarity_groups, group_config, llm, merging_prompt_template
    )
    return merged_cluster_themes

def merge_themes_in_cluster(
    cluster_themes: List[ConversationTheme],
    similarity_groups: SimilarityGroups,
    group_config,
    llm,
    merging_prompt_template: str
):
    similarity_threshold = group_config["themes_recombination"].get("similarity_threshold", 4)
    batch_changed = False
    processed_indices = set()
    merged_themes = []

    # For convenience, keep them 1-based
    for group in similarity_groups.groups:
        indices = group.theme_numbers
        if any(idx in processed_indices for idx in indices):
            continue

        indices = [idx for idx in indices if 0 <= idx - 1 < len(cluster_themes)]
        if not indices:
            logging.error("No valid indices found in the similarity group.")
            continue

        # If there's only one theme or similarity rating is below threshold, do not merge
        if (len(indices) <= 1) or (group.similarity_rating < similarity_threshold):
            continue
        # Merge these themes
        themes_to_merge = [cluster_themes[idx - 1] for idx in indices]
        merged_theme = merge_themes_with_prompt(themes_to_merge, llm, merging_prompt_template)
        merged_themes.append(merged_theme)
        processed_indices.update(indices)
        batch_changed = True

    # Add leftover themes not mentioned in similarity groups
    for i, theme in enumerate(cluster_themes, start=1):
        if i not in processed_indices:
            merged_themes.append(theme)

    return merged_themes, batch_changed

def merge_themes_with_prompt(themes_to_merge: List[ConversationTheme], llm, merging_prompt_template: str) -> ConversationTheme:
    # Prepare the context for merging
    themes_text = ""
    for theme in themes_to_merge:
        themes_text += f"Theme Name: {capitalize_theme_name(theme.name)}\nSummary: {theme.summary}\n"
        if theme.dissenting_opinions:
            themes_text += f"Dissenting opinions: {theme.dissenting_opinions}\n"
        themes_text += "\n"

    # Generate merged theme using the LLM
    try:
        merged_theme = llm.generate_structured_output(
            prompt_template=merging_prompt_template,
            context=themes_text,
            pydantic_class=ConversationTheme
        )
        logging.debug(f"Merged theme: {merged_theme}")
        return merged_theme
    except Exception as e:
        logging.error(f"Failed to merge themes: {e}")
        # If merging fails, fallback to a naive concatenation
        merged_name = " / ".join([theme.name for theme in themes_to_merge])
        merged_summary = "\n".join([f"{theme.name}: {theme.summary}" for theme in themes_to_merge])
        merged_dissenting_opinions = "\n".join([theme.dissenting_opinions for theme in themes_to_merge if theme.dissenting_opinions])
        return ConversationTheme(
            name=merged_name,
            summary=merged_summary,
            dissenting_opinions=merged_dissenting_opinions
        )

def recombine_themes(themes_list, group_config, llm):
    """
    This is the main entry point for recombining themes.
    If embedding_clustering is enabled, we first do an embedding-based
    clustering + partial merges, then we can optionally refine further
    (if you want more merges across clusters, you can do that, but by default,
    we only do merges within each cluster).
    """
    recombination_config = group_config.get('themes_recombination', {})
    similarity_prompt_template = recombination_config.get('similarity_prompt', '')
    merging_prompt_template = recombination_config.get('merging_prompt', '')

    embedding_clustering_config = group_config.get('embedding_clustering', {})
    if embedding_clustering_config.get('enabled', False):
        # Use the embedding-based approach
        logging.info('Using embeddings-based clustering and merging')
        final_themes = embedding_cluster_and_merge(
            themes_list,
            group_config,
            similarity_prompt_template,
            merging_prompt_template,
            llm
        )
        return final_themes
    else:
        # Fall back to the original iterative merging approach
        logging.info('Using iterative clustering and merging approach')
        return iterative_recombine_themes(themes_list, group_config, llm)

def iterative_recombine_themes(themes_list, group_config, llm):
    """
    The older iterative approach using random batch merges with the similarity prompt
    until no further merges occur or max iterations are reached.
    """
    recombination_config = group_config.get('themes_recombination', {})
    batch_size = recombination_config.get('batch_size', 5)
    configured_max_iterations = recombination_config.get('max_iterations', 30)
    configured_convergence_threshold = recombination_config.get('convergence_threshold', 3)
    similarity_threshold = recombination_config.get('similarity_threshold', 4)
    similarity_prompt_template = recombination_config.get('similarity_prompt', '')
    merging_prompt_template = recombination_config.get('merging_prompt', '')

    # Collect all themes from the chunks
    all_themes = []
    for themes in themes_list:
        if themes and themes.themes:
            all_themes.extend(themes.themes)

    if not all_themes:
        return None

    N = len(all_themes)
    B = batch_size

    # Calculate expected iterations for pairing any two themes
    if N > 1 and B > 1:
        expected_iterations = (N - 1) / (B - 1)
    else:
        expected_iterations = 1

    # Set k (multiplier for expected_iterations)
    k = 4
    calculated_max_iterations = int(k * expected_iterations)
    max_iterations = min(configured_max_iterations, calculated_max_iterations)

    # Calculate convergence_threshold
    calculated_convergence_threshold = max(1, int(max_iterations / 10))
    convergence_threshold = min(configured_convergence_threshold, calculated_convergence_threshold)

    logging.debug(f"Calculated max_iterations: {max_iterations}")
    logging.debug(f"Calculated convergence_threshold: {convergence_threshold}")

    iteration = 0
    previous_theme_count = N
    convergence_counter = 0

    while iteration < max_iterations:
        iteration += 1
        logging.debug(f"Recombination iteration {iteration} with {len(all_themes)} themes")

        # Randomly shuffle themes to maximize pairing chances
        random.shuffle(all_themes)

        # Split themes into batches
        batches = [all_themes[i:i + batch_size] for i in range(0, len(all_themes), batch_size)]
        new_all_themes = []
        for batch in batches:
            merged_batch, _ = process_theme_batch(
                batch,
                group_config,
                llm,
                similarity_prompt_template,
                merging_prompt_template,
                similarity_threshold
            )
            new_all_themes.extend(merged_batch)

        current_theme_count = len(new_all_themes)
        if current_theme_count == previous_theme_count:
            convergence_counter += 1
            logging.debug(f"No reduction in theme count. Convergence counter: {convergence_counter}")
        else:
            convergence_counter = 0  # Reset counter if reduction occurs

        if convergence_counter >= convergence_threshold:
            logging.debug("No reduction in theme count over multiple iterations. Terminating recombination.")
            break

        all_themes = new_all_themes
        previous_theme_count = current_theme_count

    # Return the final merged themes
    final_themes = ConversationThemes(themes=all_themes)
    return final_themes

def process_theme_batch(themes_batch: List[ConversationTheme], group_config, llm,
                        similarity_prompt_template: str, merging_prompt_template: str,
                        similarity_threshold: float):
    batch_changed = False  # Flag to indicate if any merges occurred in this batch

    if len(themes_batch) == 1:   # If there is only one theme in this batch, return it as-is
        return themes_batch, False

    # Number the themes starting from 1
    numbered_themes = [(idx + 1, theme) for idx, theme in enumerate(themes_batch)]
    themes_text = ""
    for idx, theme in numbered_themes:
        themes_text += f"{idx}. {capitalize_theme_name(theme.name)}\nSummary: {theme.summary}\n"
        if theme.dissenting_opinions:
            themes_text += f"Dissenting opinions: {theme.dissenting_opinions}\n"
        themes_text += "\n"

    # Generate similarity groups using the LLM
    try:
        similarity_groups = llm.generate_structured_output(
            prompt_template=similarity_prompt_template,
            context=themes_text,
            pydantic_class=SimilarityGroups
        )
        logging.debug(f"Similarity groups: {similarity_groups.groups}")
    except Exception as e:
        logging.error(f"Failed to identify similar themes: {e}")
        return themes_batch, False  # Return original batch if error occurs

    # Merge themes based on similarity groups
    merged_themes = []
    processed_indices = set()
    for group in similarity_groups.groups:
        indices = group.theme_numbers
        if any(idx in processed_indices for idx in indices):
            continue  # Skip if any theme in the group has already been processed
        if len(indices) == 1:
            idx = indices[0] - 1
            merged_themes.append(themes_batch[idx])
            processed_indices.add(indices[0])
        else:
            if group.similarity_rating < similarity_threshold:
                # Do not merge themes below the similarity threshold
                for idx in indices:
                    if idx not in processed_indices:
                        merged_themes.append(themes_batch[idx - 1])
                        processed_indices.add(idx)
                continue
            themes_to_merge = [themes_batch[idx - 1] for idx in indices]
            merged_theme = merge_themes_with_prompt(themes_to_merge, llm, merging_prompt_template)
            merged_themes.append(merged_theme)
            processed_indices.update(indices)
            batch_changed = True  # A merge occurred

    # Append themes that were not mentioned in any similarity groups
    for idx, theme in enumerate(themes_batch):
        if idx + 1 not in processed_indices:
            merged_themes.append(theme)

    return merged_themes, batch_changed

def build_summary_from_themes(themes, group_config, database, group_id, since, until):
    if not themes or not themes.themes:
        return "No significant themes were discussed."

    summary_lines = []
    for idx, theme in enumerate(themes.themes, 1):
        summary_lines.append(f"## **{capitalize_theme_name(theme.name)}**\n{theme.summary}")
        if theme.dissenting_opinions:
            summary_lines.append(f"\n**Dissenting opinions:** {theme.dissenting_opinions}")

    # Add links section
    links_section = build_links_section(
        database=database,
        group_config=group_config,
        group_id=group_id,
        since=since,
        until=until
    )
    if links_section:
        summary_lines.append(links_section)

    final_summary = "\n\n".join(summary_lines)
    return final_summary

def get_current_date_formatted():
    now = datetime.datetime.now()
    day = now.day
    suffix = 'th' if 11 <= day <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')
    current_date = now.strftime(f'%A, %B {day}{suffix} %Y')
    return current_date

def summarize_group(config, args, group_id, llm_dict, vision_util, stt_model, resume_data, resume_file):
    group_config = get_group_config(config, group_id)
    group_config['group_id'] = group_id  # Ensure group_id is in the group_config
    attachment_root = Path(group_config.get('attachment_root', './'))

    if args.output:
        output_file = args.output
    else:
        output_file = group_config.get('output_file', 'summary.md')

    output_file_orig = group_config.get('output_file_orig', 'summary_orig.md')

    database = group_config.get('database', 'messages.db')

    # Determine date range
    since, until = get_date_range(args)

    # Load group resume data
    group_resume_data = resume_data.get('groups', {}).get(group_id, {})
    if 'groups' not in resume_data:
        resume_data['groups'] = {}
    resume_data['groups'][group_id] = group_resume_data

    # Fetch messages
    if 'messages' in group_resume_data and not args.redo_themes:
        messages = group_resume_data['messages']
        logging.info(f"Loaded messages from resume data for group {group_id}")
    else:
        messages = fetch_messages(database, group_id, since, until)
        group_resume_data['messages'] = messages
        if resume_file:
            save_resume_file(resume_file, resume_data)

    if not messages:
        print(f"No messages found for group {group_id} in the specified date range.")
        # Remove group from resume data if present
        if group_id in resume_data.get('groups', {}):
            del resume_data['groups'][group_id]
            if resume_file:  # Only save if resume file is provided
                save_resume_file(resume_file, resume_data)
        return

    # Process attachments (images and audio)
    if group_resume_data.get('attachments_processed'):
        logging.info(f"Attachments already processed for group {group_id}")
    else:
        process_attachments(messages, group_config, args.regenerate_attachment_descriptions, database, attachment_root, vision_util, stt_model)
        group_resume_data['attachments_processed'] = True
        if resume_file:  # Only save if resume file is provided
            save_resume_file(resume_file, resume_data)

    # Build prompts for summarization
    if 'conversation_chunks' in group_resume_data and not args.redo_themes:
        conversation_chunks = group_resume_data['conversation_chunks']
        logging.info(f"Loaded conversation chunks from resume data for group {group_id}")
    else:
        conversation_chunks = build_conversation_chunks(messages, group_config, llm_dict)
        group_resume_data['conversation_chunks'] = conversation_chunks
        if resume_file:  # Only save if resume file is provided
            save_resume_file(resume_file, resume_data)

    # Initialize LLMs for themes, recombination, translation
    themes_model_name = group_config.get('themes', {}).get('model')
    themes_llm = llm_dict.get(themes_model_name)

    recombination_model_name = group_config.get('themes_recombination', {}).get('model')
    recombination_llm = llm_dict.get(recombination_model_name)

    translation_model_name = group_config.get('translation', {}).get('model')
    translation_llm = llm_dict.get(translation_model_name)

    link_summarizer_model_name = group_config.get('link_summarizer', {}).get('model')
    link_summarizer_llm = llm_dict.get(link_summarizer_model_name)

    # Process links
    if group_resume_data.get('links_processed') and not args.redo_links:
        logging.info(f"Links already processed for group {group_id}")
    else:
        process_links(messages, group_config, args.regenerate_link_summaries, database, group_id, link_summarizer_llm)
        group_resume_data['links_processed'] = True
        if resume_file:  # Only save if resume file is provided
            save_resume_file(resume_file, resume_data)

    # Generate summaries (themes) per chunk
    if 'themes' in group_resume_data and not args.redo_themes:
        summaries_dicts = group_resume_data['themes']
        summaries = [ConversationThemes(**theme_dict) for theme_dict in summaries_dicts]
        logging.info(f"Loaded themes from resume data for group {group_id}")
    else:
        summaries = []
        for conversation_chunk in conversation_chunks:
            themes = generate_themes(conversation_chunk, group_config, themes_llm)
            if themes:
                summaries.append(themes)
        # Convert to serializable format
        summaries_dicts = [t.dict() for t in summaries]
        group_resume_data['themes'] = summaries_dicts
        if resume_file:  # Only save if resume file is provided
            save_resume_file(resume_file, resume_data)

    # Recombine summaries
    if 'final_themes' in group_resume_data and not args.redo_merging:
        final_themes_dict = group_resume_data['final_themes']
        final_themes = ConversationThemes(**final_themes_dict)
        logging.info(f"Loaded final themes from resume data for group {group_id}")
    else:
        final_themes = recombine_themes(summaries, group_config, recombination_llm)
        # Convert to serializable format
        final_themes_dict = final_themes.dict() if final_themes else {}
        group_resume_data['final_themes'] = final_themes_dict
        if resume_file:  # Only save if resume file is provided
            save_resume_file(resume_file, resume_data)

    # Build final summary from themes
    if 'final_summary' in group_resume_data and not args.redo_merging:
        final_summary = group_resume_data['final_summary']
        logging.info(f"Loaded final summary from resume data for group {group_id}")
    else:
        final_summary = build_summary_from_themes(final_themes, group_config, database, group_id, since, until)
        group_resume_data['final_summary'] = final_summary
        if resume_file:  # Only save if resume file is provided
            save_resume_file(resume_file, resume_data)

    # Write the pre-translation summary to output_file_orig
    with open(output_file_orig, 'w', encoding='utf-8') as f:
        f.write(final_summary)

    # Translate the final summary if the desired language is not English
    target_language = group_config.get('language', 'English')
    if target_language.lower() != 'english':
        try:
            translation_prompt = group_config.get('translation', {}).get('prompt', '')
            max_chunk_size = group_config.get('translation', {}).get('max_chunk_size', 2000)
            if 'translated_summary' in group_resume_data:
                translated_summary = group_resume_data['translated_summary']
                logging.info(f"Loaded translated summary from resume data for group {group_id}")
            else:
                translated_summary = translation_llm.translate_text(final_summary, target_language, translation_prompt, max_chunk_size)
                group_resume_data['translated_summary'] = translated_summary
                if resume_file:  # Only save if resume file is provided
                    save_resume_file(resume_file, resume_data)
            final_summary = translated_summary
            logging.info(f"Translated summary to {target_language}.")
        except Exception as e:
            logging.error(f"Translation failed: {e}")
            logging.info("Proceeding with the original summary.")
    else:
        logging.info("Translation not required as target language is English.")

    # Write final summary to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(final_summary)

    print(f"Summary for group {group_id} written to {output_file}")

    # Remove group data from resume_data and save
    if 'groups' in resume_data:
        if group_id in resume_data['groups']:
            del resume_data['groups'][group_id]
            if resume_file:  # Only save if resume file is provided
                save_resume_file(resume_file, resume_data)

def capitalize_theme_name(name):
    return name[0].upper() + name[1:] # not using .capitalize() because we don't want to touch capitalization of other letters
