#!/bin/bash
set -e

# Ensure script runs from its containing directory
cd "$(dirname "$0")" || exit 1

VENV_DIR="venv"

# Create virtualenv if missing
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtualenv in $VENV_DIR..."
  python3 -m venv "$VENV_DIR"
fi

# Activate virtualenv
source "$VENV_DIR/bin/activate"

# Upgrade pip and install requirements (PyTorch commented out in requirements.txt)
python3 -m pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "✅ Backend ready. Starting on http://localhost:3001 ..."
echo "   (PyTorch optional – using heuristic mode if not installed)"
echo ""

# Run backend
exec python3 app.py
