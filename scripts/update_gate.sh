#!/bin/bash
# Script to update GATE.md with current project state
# Usage: ./scripts/update_gate.sh

set -e
TODAY=$(date +%Y-%m-%d)
echo "Updating GATE.md timestamp to: $TODAY"
sed -i '' "s/Last Updated: .*/Last Updated: $TODAY/" GATE.md
echo "Done! Remember to update sections manually if needed."
