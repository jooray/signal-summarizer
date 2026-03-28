#!/bin/bash
set -euo pipefail

REMOTE_HOST="username@server"
REMOTE_ROOT="signal-message-processor"
LOCAL_DB="messages-new.db"
LOCAL_ATTACHMENTS="attachments"

scp -r "${REMOTE_HOST}:${REMOTE_ROOT}/messages.db" "${LOCAL_DB}"
mkdir -p "${LOCAL_ATTACHMENTS}"
rsync -ralp "${REMOTE_HOST}:${REMOTE_ROOT}/attachments/" "${LOCAL_ATTACHMENTS}/"
poetry run python ./import_messages.py "${LOCAL_DB}" messages.db
