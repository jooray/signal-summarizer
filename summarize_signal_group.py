#!/usr/bin/env python3

import argparse
import sys
import sqlite3
from group_summarizer import (
    setup_logging,
    load_config,
    summarize_group,
)
from vision_util import VisionUtil
from llm_util import LLMUtil
from resume_util import load_resume_file, save_resume_file, delete_resume_file
import logging


def parse_args():
    parser = argparse.ArgumentParser(description='Signal Group Summarizer')
    parser.add_argument('--config', default='config.json', help='Path to the configuration file (default: config.json)')
    parser.add_argument('--log-level', default='INFO', help='Logging level')
    parser.add_argument('--group', nargs='*', help='Group ID(s) to generate summary for. If not specified, summarize all groups.')
    parser.add_argument('--output', help='Output file for the summary')
    parser.add_argument('--since', help='Start date (inclusive) in format YYYY-MM-DD')
    parser.add_argument('--until', help='End date (inclusive) in format YYYY-MM-DD')
    parser.add_argument('--last-week', action='store_true', help='Summarize messages from the last 7 days')
    parser.add_argument('--last-month', action='store_true', help='Summarize messages from the last 4 weeks')
    parser.add_argument('--list-groups', action='store_true', help='List all groups (with ID and name)')
    parser.add_argument('--regenerate-attachment-descriptions', action='store_true',
                        help='Regenerate attachment descriptions even if they already exist')
    parser.add_argument('--regenerate-link-summaries', action='store_true',
                        help='Regenerate link summaries even if they already exist')
    parser.add_argument('--resume-file', help='Path to the resume file')
    parser.add_argument('--keep-resume-file', action='store_true', help='Keep the resume file after successful processing')
    parser.add_argument('--redo-merging', action='store_true', help='Redo the merging step even if data is present in resume file')
    parser.add_argument('--redo-links', action='store_true', help='Redo the link processing step even if data is present in resume file')
    parser.add_argument('--redo-themes', action='store_true', help='Redo the themes generation step even if data is present in resume file (implies --redo-merging)')
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log_level)

    # If --redo-themes is specified, it implies --redo-merging
    if args.redo_themes:
        args.redo_merging = True

    config = load_config(args.config)

    if args.resume_file:
        resume_file = args.resume_file
    else:
        resume_file = config.get('defaults', {}).get('resume_file', None)

    if resume_file:
        resume_data = load_resume_file(resume_file)
        logging.info(f"Using resume file: {resume_file}")
    else:
        resume_data = {'groups': {}}

    # Handle --list-groups
    if args.list_groups:
        list_groups(config)
        sys.exit(0)

    # If no group is specified, summarize all groups
    if not args.group:
        group_ids = list(config['groups'].keys())
    else:
        group_ids = args.group

    # Initialize VisionUtil if needed
    vision_config = config.get('defaults', {}).get('vision_config', {})
    if vision_config.get('enabled', True):
        vision_util = VisionUtil(vision_config)
    else:
        vision_util = None

    # Initialize speech-to-text model if needed
    speech_config = config.get('defaults', {}).get('speech_to_text', {})
    enable_voice_processing = speech_config.get('enabled', True)
    if enable_voice_processing:
        from pywhispercpp.model import Model as WhisperModel
        whisper_model_name = speech_config.get('model', 'medium')
        whisper_n_threads = speech_config.get('n_threads', 1)
        stt_model = WhisperModel(whisper_model_name, n_threads=whisper_n_threads)
    else:
        stt_model = None

    # Instantiate all models. Decision models (Venice's Jev) share the
    # dictionary but are DecisionClient instances, not chat LLMs.
    from decision_util import DECISION_PROVIDERS, DecisionClient

    models_config = config.get('defaults', {}).get('models', {})
    llm_dict = {}
    for model_name, model_config in models_config.items():
        if model_config.get('provider') in DECISION_PROVIDERS:
            llm_dict[model_name] = DecisionClient(model_config)
        else:
            llm_dict[model_name] = LLMUtil(model_config)

    for group_id in group_ids:
        summarize_group(config, args, group_id, llm_dict, vision_util, stt_model, resume_data, resume_file)

    # After processing all groups, check if we can delete the resume file
    if resume_file and 'groups' in resume_data and not resume_data['groups']:
        if args.keep_resume_file:
            logging.info(f"All groups processed. Keeping resume file {resume_file} as per --keep-resume-file.")
        else:
            delete_resume_file(resume_file)
            logging.info(f"All groups processed. Deleted resume file {resume_file}.")


def list_groups(config):
    database = config.get('defaults', {}).get('database', 'messages.db')
    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    cursor.execute('SELECT DISTINCT groupId, groupName FROM messages')
    groups = cursor.fetchall()
    print("Available groups:")
    for group in groups:
        groupId, groupName = group
        print(f"ID: {groupId}, Name: {groupName}")
    conn.close()


if __name__ == '__main__':
    main()
