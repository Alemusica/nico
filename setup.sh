#!/bin/bash
# =============================================================================
# Automatic Setup Script for NICO Streamlit Gates
# =============================================================================
# This script makes the project portable - run once on any machine to set up.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#
# What it does:
# 1. Detects Python 3.12+
# 2. Creates virtual environment
# 3. Installs dependencies
# 4. Checks for CMEMS credentials
# 5. Creates portable start script
# =============================================================================

set -e  # Exit on error

echo "🚀 NICO Streamlit Gates - Automatic Setup"
echo "=========================================="
echo ""

# -----------------------------------------------------------------------------
# 1. DETECT PYTHON
# -----------------------------------------------------------------------------
echo "📍 Step 1/5: Detecting Python..."

PYTHON_CMD=""
for cmd in python3.12 python3.13 python3.11 python3; do
    if command -v $cmd &> /dev/null; then
        VERSION=$($cmd --version 2>&1 | awk '{print $2}')
        MAJOR=$(echo $VERSION | cut -d. -f1)
        MINOR=$(echo $VERSION | cut -d. -f2)
        
        if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 11 ]; then
            PYTHON_CMD=$cmd
            echo "✅ Found $cmd (version $VERSION)"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "❌ ERROR: Python 3.11+ not found!"
    echo "   Please install Python 3.12 or newer:"
    echo "   - macOS: brew install python@3.12"
    echo "   - Linux: apt install python3.12 or yum install python312"
    exit 1
fi

# -----------------------------------------------------------------------------
# 2. CREATE VIRTUAL ENVIRONMENT
# -----------------------------------------------------------------------------
echo ""
echo "📦 Step 2/5: Creating virtual environment..."

if [ -d ".venv" ]; then
    echo "⚠️  Virtual environment already exists. Skipping creation."
else
    $PYTHON_CMD -m venv .venv
    echo "✅ Virtual environment created in .venv/"
fi

# Activate venv
source .venv/bin/activate

# -----------------------------------------------------------------------------
# 3. INSTALL DEPENDENCIES
# -----------------------------------------------------------------------------
echo ""
echo "📥 Step 3/5: Installing dependencies..."

# Upgrade pip first
pip install --upgrade pip > /dev/null 2>&1

# Install from requirements.txt
if [ -f "requirements.txt" ]; then
    echo "   Installing from requirements.txt..."
    pip install -r requirements.txt > /dev/null 2>&1
    echo "✅ Dependencies installed"
else
    echo "❌ ERROR: requirements.txt not found!"
    exit 1
fi

# Check for copernicusmarine (critical for CMEMS L4)
if ! python -c "import copernicusmarine" 2>/dev/null; then
    echo "⚠️  copernicusmarine not found, installing..."
    pip install copernicusmarine > /dev/null 2>&1
    echo "✅ copernicusmarine installed"
fi

# -----------------------------------------------------------------------------
# 4. CHECK CREDENTIALS
# -----------------------------------------------------------------------------
echo ""
echo "🔑 Step 4/5: Checking CMEMS credentials..."

CREDS_FILE="config/credentials.yaml"
CREDS_TEMPLATE="config/credentials.yaml.template"
CREDS_CONFIGURED=false

# Copy template if credentials file doesn't exist
if [ ! -f "$CREDS_FILE" ]; then
    if [ -f "$CREDS_TEMPLATE" ]; then
        cp "$CREDS_TEMPLATE" "$CREDS_FILE"
        echo "✅ Created $CREDS_FILE from template"
    fi
fi

if [ -f "$CREDS_FILE" ]; then
    # Check if credentials are filled in
    if grep -q "your_copernicus_username_here" "$CREDS_FILE" 2>/dev/null; then
        CREDS_CONFIGURED=false
    else
        CREDS_CONFIGURED=true
    fi
fi

if [ "$CREDS_CONFIGURED" = false ]; then
    echo "⚠️  CMEMS credentials not configured"
    echo ""
    echo "   To download CMEMS L4 data, you need a free Copernicus Marine account:"
    echo "   1. Register at: https://marine.copernicus.eu/register"
    echo "   2. Edit config/credentials.yaml with your username/password"
    echo "   3. Or set environment variables:"
    echo "      export COPERNICUS_USERNAME='your_username'"
    echo "      export COPERNICUS_PASSWORD='your_password'"
    echo ""
    echo "   (You can still run the app, but CMEMS L4 downloads will fail)"
else
    echo "✅ Credentials configured in $CREDS_FILE"
fi

# -----------------------------------------------------------------------------
# 5. CREATE PORTABLE START SCRIPT
# -----------------------------------------------------------------------------
echo ""
echo "🛠️  Step 5/5: Creating portable start script..."

# Get absolute path to current directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cat > run_streamlit.sh << EOF
#!/bin/bash
# Auto-generated portable start script
# Created by setup.sh on $(date)

PROJECT_DIR="$PROJECT_DIR"

cd "\$PROJECT_DIR"
source .venv/bin/activate
streamlit run streamlit_app.py --server.headless true
EOF

chmod +x run_streamlit.sh

echo "✅ Portable start script created: run_streamlit.sh"

# -----------------------------------------------------------------------------
# DONE
# -----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "To start Streamlit:"
echo "   ./run_streamlit.sh"
echo ""
echo "Or manually:"
echo "   source .venv/bin/activate"
echo "   streamlit run streamlit_app.py"
echo ""
echo "The app will be available at: http://localhost:8501"
echo ""

# Test import
echo "🧪 Testing imports..."
python -c "
import streamlit as st
import pandas as pd
import numpy as np
from src.services.cmems_l4_service import CMEMSL4Service
print('✅ All critical imports working!')
" 2>/dev/null

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Ready to go! Run: ./run_streamlit.sh"
else
    echo ""
    echo "⚠️  Import test failed. Check dependencies."
fi
