#!/usr/bin/env bash
# Launch the Telegram RAG chatbot via the CLI
# Usage: ./run_chatbot.sh [extra python args]
set -e
python3 -m app.cli "$@"
