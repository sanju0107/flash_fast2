#!/bin/bash
set -e

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install --no-cache-dir fastapi==0.104.1 uvicorn==0.24.0 pydantic==2.5.0

echo "Build completed successfully!"