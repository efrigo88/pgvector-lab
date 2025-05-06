#!/bin/bash

# Exit on error
set -e

echo "🚀 docker compose up -d --build..."
docker compose down -v
docker compose up -d --build

echo "⏳ Waiting for Ollama container to be ready..."
sleep 10

# Pull the model
echo "📥 Pulling nomic-embed-text model..."
docker exec ollama ollama pull nomic-embed-text

echo "🚀 Starting the process..."
docker exec -it app python -m src.main

echo "✅ Process finished successfully!"
echo "ℹ️  If you want to run the process again, use: docker exec -it app python -m src.main"
echo "ℹ️  If you're done, run: docker compose down -v"
