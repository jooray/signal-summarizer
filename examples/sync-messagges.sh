#!/bin/bash
scp -r username@server:signal-message-processor/messages.db messages-new.db
rsync -ralp username@server:signal-message-processor/attachments/* attachments/ &
poetry run python3 ./import_messages.py messages-new.db messages.db
