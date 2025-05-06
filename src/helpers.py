import os
import json
from datetime import datetime
from typing import List, Dict, Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
import psycopg2
from psycopg2.extras import execute_values
from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.types as T
from docling.datamodel.document import InputDocument
from docling.document_converter import DocumentConverter
from langchain_ollama import OllamaEmbeddings


# Define schema
schema = T.StructType(
    [
        T.StructField("id", T.StringType(), True),
        T.StructField("chunk", T.StringType(), True),
        T.StructField(
            "metadata",
            T.StructType(
                [
                    T.StructField("source", T.StringType(), True),
                    T.StructField("chunk_index", T.IntegerType(), True),
                    T.StructField("title", T.StringType(), True),
                    T.StructField("chunk_size", T.IntegerType(), True),
                ]
            ),
            True,
        ),
        T.StructField("processed_at", T.TimestampType(), True),
        T.StructField("processed_dt", T.StringType(), True),
        T.StructField("embeddings", T.ArrayType(T.FloatType()), True),
    ]
)

# Create Spark session
spark = (
    SparkSession.builder.appName("TestSpark")
    .master(os.getenv("THREADS"))
    .config("spark.driver.memory", os.getenv("DRIVER_MEMORY"))
    .config("spark.sql.shuffle.partitions", os.getenv("SHUFFLE_PARTITIONS"))
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config(
        "spark.sql.catalog.spark_catalog",
        "org.apache.spark.sql.delta.catalog.DeltaCatalog",
    )
    .config(
        "spark.jars.packages",
        "io.delta:delta-spark_2.12:3.2.0,"
        "org.apache.hadoop:hadoop-aws:3.3.4,"
        "com.amazonaws:aws-java-sdk-bundle:1.12.262",
    )
    .config(
        "spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem"
    )
    .config(
        "spark.hadoop.fs.s3a.aws.credentials.provider",
        "com.amazonaws.auth.DefaultAWSCredentialsProviderChain",
    )
    .config("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com")
    .config("spark.hadoop.fs.s3a.path.style.access", "false")
    .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "true")
    .getOrCreate()
)


def parse_pdf(source_path: str) -> InputDocument:
    """Parse the PDF document using DocumentConverter."""
    converter = DocumentConverter()
    result = converter.convert(source_path)
    return result.document


def get_text_content(doc: InputDocument) -> List[str]:
    """Extract text content from the document."""
    return [
        text_item.text.strip()
        for text_item in doc.texts
        if text_item.text.strip() and text_item.label == "text"
    ]


def get_chunks(
    text_content: List[str], chunk_size: int = 750, chunk_overlap: int = 100
) -> List[str]:
    """Split text content into semantically meaningful chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
    )
    chunks = []
    for text in text_content:
        chunks.extend(splitter.split_text(text))
    if not chunks:
        raise ValueError("No text chunks found in the document.")
    return chunks


def get_ids(chunks: List[str], source_path: str) -> List[str]:
    """Generate unique IDs for each chunk."""
    filename = source_path.split("/")[-1]
    return [f"{filename}_chunk_{i}" for i in range(len(chunks))]


def get_metadata(
    chunks: List[str],
    doc: InputDocument,
    source_path: str,
) -> List[Dict[str, Any]]:
    """Generate metadata for each chunk."""
    filename = source_path.split("/")[-1]
    return [
        {
            "source": filename,
            "chunk_index": i,
            "title": doc.name,
            "chunk_size": len(chunk),
        }
        for i, chunk in enumerate(chunks)
    ]


def get_embeddings(
    chunks: List[str],
    model: OllamaEmbeddings,
) -> List[List[float]]:
    """Get embeddings for a list of chunks using Ollama embeddings."""
    return model.embed_documents(chunks)


def save_json_data(
    data: List[Dict[str, Any]], file_path: str, overwrite: bool = True
) -> None:
    """Save data to a JSONL file (one JSON object per line)."""
    if not overwrite and os.path.exists(file_path):
        raise FileExistsError(
            f"File {file_path} already exists and overwrite=False"
        )
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")


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
                    embedding vector(768)
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


def store_in_postgres(df: DataFrame) -> None:
    """Store data in PostgreSQL with pgvector."""
    init_db()

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            rows = df.select(
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
                    json.dumps(row.metadata.asDict()),  # Convert dict to JSON string
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
    queries: List[str],
    embeddings: List[List[float]],
) -> List[Dict[str, Any]]:
    """Run queries and prepare results in json format."""
    all_results = []

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            for query, embedding in zip(queries, embeddings):
                # Perform similarity search using pgvector
                cur.execute(
                    """
                    SELECT
                        chunk as text,
                        metadata,
                        1 - (embedding <=> %s) as similarity
                    FROM documents
                    ORDER BY embedding <=> %s
                    LIMIT 3;
                """,
                    (embedding, embedding),
                )

                results = cur.fetchall()

                query_result = {
                    "processed_at": datetime.now().isoformat(),
                    "query": query,
                    "results": [
                        {
                            "text": text,
                            "metadata": metadata,
                            "similarity": similarity,
                        }
                        for text, metadata, similarity in results
                    ],
                }
                all_results.append(query_result)

    return all_results
