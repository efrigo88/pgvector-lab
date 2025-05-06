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

# Ask user if they want to run docker compose down
read -p "Do you want to run docker compose down? (y/N): " response

if [[ "$response" =~ ^[Yy]$ ]]; then
    echo "🔽 Running docker compose down..."
    docker compose down -v
    echo "✅ Docker compose down completed"
else
    echo "ℹ️  If you want to run the process again, use: docker exec -it app python -m src.main"
fi
