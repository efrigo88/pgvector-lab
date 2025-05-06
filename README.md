# PDF Document Processing with pgvector

This project processes PDF documents, generates embeddings using Ollama, and stores them in PostgreSQL with pgvector for semantic search capabilities.

## Prerequisites

- Docker and Docker Compose
- Python 3.10 or higher
- Git

## Project Structure

```
.
├── data/
│   ├── input/          # Place your PDF files here
│   ├── output/         # Generated Iceberg tables and JSONL files
│   └── answers/        # Query results
├── scripts/
│   └── run_process.sh  # Script to run the entire process
├── src/
│   ├── helpers.py      # Helper functions
│   ├── main.py         # Main processing logic
│   └── queries.py      # Predefined queries
├── Dockerfile
├── docker-compose.yml
├── .env.example        # Example environment variables
└── pyproject.toml
```

## Environment Variables

1. Copy the example environment file:

   ```bash
   cp .env.example .env
   ```

2. Review and adjust the variables in `.env` as needed for your environment.

## Running the Project

### Quick Start

1. Place your PDF file in the `data/input` directory
2. Make the run script executable:
   ```bash
   chmod +x scripts/run_process.sh
   ```
3. Run the process:
   ```bash
   ./scripts/run_process.sh
   ```

The script will:

1. Build and start Docker containers
2. Pull the required Ollama model
3. Process the PDF and store embeddings in PostgreSQL
4. Run predefined queries
5. Save results in `data/answers/answers.jsonl`

### Manual Steps

If you prefer to run steps manually:

1. Start the containers:

   ```bash
   docker compose up -d --build
   ```

2. Pull the Ollama model:

   ```bash
   docker exec ollama ollama pull nomic-embed-text
   ```

3. Run the process:

   ```bash
   docker exec -it app python -m src.main
   ```

4. Stop the containers when done:
   ```bash
   docker compose down
   ```

## Output Files

- `data/output/table/`: Spark Iceberg table with processed data
- `data/output/jsonl_file/`: JSONL file with processed data
- `data/answers/answers.jsonl`: Query results with similarity scores

## Querying

The project includes predefined queries in `src/queries.py`. Each query:

1. Is converted to an embedding using the same model as documents
2. Is compared against document embeddings using cosine similarity
3. Returns the top 3 most similar document chunks

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
