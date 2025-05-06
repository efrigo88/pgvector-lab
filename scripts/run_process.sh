#!/bin/bash

# Exit on error
set -e

echo "🚀 docker compose up -d --build..."
docker compose up -d --build

echo "⏳ Waiting for Ollama container to be ready..."
sleep 10

# Pull the model
echo "📥 Pulling nomic-embed-text model..."
docker exec ollama ollama pull nomic-embed-text

echo "🚀 Starting the process..."
docker exec -it app python -m src.main

echo "✅ Process finished successfully!"