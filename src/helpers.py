import os
import json
from datetime import datetime
from typing import List, Dict, Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_postgres import PGVector
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


def get_db_connection_string() -> str:
    """Create a connection string for PostgreSQL."""
    return (
        f"postgresql+psycopg://{os.getenv('POSTGRES_USER')}:"
        f"{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}:"
        f"{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
    )


def init_vector_store(embeddings: OllamaEmbeddings) -> PGVector:
    """Initialize the vector store with LangChain's PGVector."""
    connection_string = get_db_connection_string()
    collection_name = "documents"

    return PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=connection_string,
        use_jsonb=True,
    )


def store_in_postgres(df: DataFrame, embeddings: OllamaEmbeddings) -> None:
    """Store data in PostgreSQL using LangChain's PGVector."""
    vector_store = init_vector_store(embeddings)

    # Convert DataFrame rows to LangChain Documents
    documents = []
    for row in df.collect():
        doc = Document(
            page_content=row.chunk,
            metadata={
                "id": row.id,
                "source": row.metadata["source"],
                "chunk_index": row.metadata["chunk_index"],
                "title": row.metadata["title"],
                "chunk_size": row.metadata["chunk_size"],
                "processed_at": row.processed_at.isoformat(),
                "processed_dt": row.processed_dt,
            },
        )
        documents.append(doc)

    # Add documents to vector store
    vector_store.add_documents(documents)
    print(f"📊 Total documents in PostgreSQL: {len(documents)}")


def prepare_queries(
    queries: List[str],
    embeddings: OllamaEmbeddings,
) -> List[Dict[str, Any]]:
    """Run queries and prepare results in json format using LangChain's PGVector."""
    vector_store = init_vector_store(embeddings)
    all_results = []

    for query in queries:
        # Perform similarity search using LangChain's PGVector
        results = vector_store.similarity_search(
            query,
            k=3,
        )

        query_result = {
            "processed_at": datetime.now().isoformat(),
            "query": query,
            "results": [
                {
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity": 1.0,  # LangChain doesn't provide similarity scores directly
                }
                for doc in results
            ],
        }
        all_results.append(query_result)

    return all_results
