#!/bin/bash

# Exit on error
set -e

# Install system dependencies
apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    git

# Upgrade pip
python3 -m pip install --upgrade pip setuptools wheel

# Install Python dependencies
pip install --no-cache-dir -r requirements.txt

# Download minimal spaCy model
python3 -m spacy download en_core_web_sm

# Create health check endpoint
mkdir -p public
cat > public/health <<EOF
OK
EOF

echo "Build completed successfully!"