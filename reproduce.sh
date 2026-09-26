#!/bin/bash
# Reproduce the Quantum Advantage Boundary Hunter campaign
set -e

# Check Python
python3 --version || { echo "Python 3.10+ required"; exit 1; }

# Install dependencies
pip install -r requirements.txt

# Check for BlueQubit token
if [ -z "$BLUEQUBIT_API_TOKEN" ]; then
    echo "WARNING: BLUEQUBIT_API_TOKEN not set. Running with local simulation only."
    echo "Set it with: export BLUEQUBIT_API_TOKEN='your_token'"
    USE_BLUEQUBIT="--no-bluequbit"
else
    echo "BlueQubit token found. Using BlueQubit compute."
    USE_BLUEQUBIT="--max-cost-usd 5.0"
fi

# Run the campaign
python -m src.main --budget 25 $USE_BLUEQUBIT

echo ""
echo "Campaign complete. See:"
echo "  reports/PROTOTYPE_REPORT.md"
echo "  reports/bluequbit_application_evidence.md"
echo "  results/figures/"
echo "  state/decision_log.jsonl"
