import os
from datetime import datetime
from typing import List, Dict, Any, Tuple

from langchain_ollama import OllamaEmbeddings
import pyspark.sql.functions as F
from pyspark.sql import DataFrame
import psycopg2
from psycopg2.extras import execute_values

from .helpers import (
    parse_pdf,
    get_text_content,
    get_chunks,
    get_ids,
    get_metadata,
    get_embeddings,
    save_json_data,
    spark,
    schema,
)

from .queries import QUERIES

INPUT_PATH = "./data/input/Example_DCL.pdf"
OUTPUT_PATH = "./data/output/"
JSONL_PATH = "./data/output/jsonl_file"
ANSWERS_PATH = "./data/answers/answers.jsonl"
CHUNK_SIZE = 750
CHUNK_OVERLAP = 100


def get_db_connection():
    """Create a connection to PostgreSQL."""
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )


def init_db():
    """Initialize the database with pgvector extension and required tables."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Enable pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # Create documents table
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    chunk TEXT,
                    metadata JSONB,
                    processed_at TIMESTAMP,
                    processed_dt TEXT,
                    embedding vector(1536)
                );
            """
            )

            # Create index for similarity search
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS documents_embedding_idx 
                ON documents 
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """
            )

            conn.commit()


def process_document() -> Tuple[
    List[str],
    List[str],
    List[Dict[str, Any]],
    List[List[float]],
    OllamaEmbeddings,
]:
    """Process PDF and generate embeddings."""
    doc = parse_pdf(INPUT_PATH)
    text_content = get_text_content(doc)
    print("✅ Text content generated.")

    chunks = get_chunks(text_content, CHUNK_SIZE)
    ids = get_ids(chunks, INPUT_PATH)
    metadatas = get_metadata(chunks, doc, INPUT_PATH)
    print("✅ Chunks, IDs and Metadatas generated.")

    model = OllamaEmbeddings(
        model="nomic-embed-text", base_url=os.getenv("OLLAMA_HOST")
    )
    embeddings = get_embeddings(chunks, model)
    print("✅ Embeddings generated.")
    return ids, chunks, metadatas, embeddings, model


def create_dataframe(
    ids: List[str],
    chunks: List[str],
    metadatas: List[Dict[str, Any]],
    embeddings: List[List[float]],
) -> DataFrame:
    """Create and save DataFrame with processed data."""
    df = spark.createDataFrame(
        [
            {
                "id": id_val,
                "chunk": chunk,
                "metadata": metadata,
                "processed_at": datetime.now(),
                "processed_dt": datetime.now().strftime("%Y-%m-%d"),
                "embeddings": embedding,
            }
            for id_val, chunk, metadata, embedding in zip(
                ids, chunks, metadatas, embeddings
            )
        ],
        schema=schema,
    )
    return df


def deduplicate_data(df: DataFrame) -> DataFrame:
    """Deduplicate data and return processed DataFrame."""
    df = (
        df.orderBy(F.col("processed_at").desc())
        .groupBy("id")
        .agg(
            F.first("processed_at").alias("processed_at"),
            F.first("processed_dt").alias("processed_dt"),
            F.first("chunk").alias("chunk"),
            F.first("metadata").alias("metadata"),
            F.first("embeddings").alias("embeddings"),
        )
    )
    return df


def store_in_postgres(df_loaded: DataFrame) -> None:
    """Store data in PostgreSQL with pgvector."""
    init_db()

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            rows = df_loaded.select(
                "id",
                "chunk",
                "metadata",
                "embeddings",
                "processed_at",
                "processed_dt",
            ).collect()

            # Prepare data for batch insert
            data = [
                (
                    row.id,
                    row.chunk,
                    row.metadata.asDict(),
                    row.embeddings,
                    row.processed_at,
                    row.processed_dt,
                )
                for row in rows
            ]

            # Upsert data
            execute_values(
                cur,
                """
                INSERT INTO documents (
                    id, chunk, metadata, embedding, processed_at, processed_dt
                )
                VALUES %s
                ON CONFLICT (id) DO UPDATE SET
                    chunk = EXCLUDED.chunk,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding,
                    processed_at = EXCLUDED.processed_at,
                    processed_dt = EXCLUDED.processed_dt;
                """,
                data,
            )

            conn.commit()

            # Get total count
            cur.execute("SELECT COUNT(*) FROM documents;")
            total_docs = cur.fetchone()[0]
            print(f"📊 Total documents in PostgreSQL: {total_docs}")


def prepare_queries(
    model: OllamaEmbeddings, queries: List[str]
) -> List[Dict[str, Any]]:
    """Run queries and prepare results in json format."""
    all_results = []

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            for query in queries:
                query_embedding = model.embed_documents([query])[0]

                # Perform similarity search
                cur.execute(
                    """
                    SELECT chunk, metadata, 1 - (embedding <=> %s) as similarity
                    FROM documents
                    ORDER BY embedding <=> %s
                    LIMIT 3;
                """,
                    (query_embedding, query_embedding),
                )

                results = cur.fetchall()

                query_result = {
                    "processed_at": datetime.now().isoformat(),
                    "query": query,
                    "results": [
                        {
                            "text": doc,
                            "metadata": meta,
                            "similarity": sim,
                        }
                        for doc, meta, sim in results
                    ],
                }
                all_results.append(query_result)

    return all_results


def main() -> None:
    """Process PDF, transform data, store in PostgreSQL, and run queries."""
    # Process document and generate embeddings
    ids, chunks, metadatas, embeddings, model = process_document()

    df = create_dataframe(ids, chunks, metadatas, embeddings)

    # Save DataFrame to Delta table
    (
        df.write.format("delta")
        .mode("append")
        .partitionBy("processed_dt")
        .save(OUTPUT_PATH)
    )
    print(f"✅ Saved Delta table in {OUTPUT_PATH}")

    # Load DataFrame from Delta table
    df_loaded = spark.read.format("delta").load(OUTPUT_PATH)

    df_deduplicated = deduplicate_data(df_loaded)
    print(f"✅ Deduplicated DataFrame in {OUTPUT_PATH}")

    # Save DataFrame as JSONL file for development purposes
    (
        df_deduplicated.repartition(1)
        .write.format("json")
        .mode("overwrite")
        .save(JSONL_PATH)
    )
    print(f"✅ Saved JSONL file in {JSONL_PATH}")

    store_in_postgres(df_deduplicated)

    # Run queries and save answers
    answers = prepare_queries(model, QUERIES)
    save_json_data(answers, ANSWERS_PATH)
    print(f"✅ Saved answers in {ANSWERS_PATH}")
    print("✅ Process completed!")

    spark.stop()
    print("✅ Spark session stopped.")


if __name__ == "__main__":
    main()
