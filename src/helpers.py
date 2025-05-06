import os
import json
from datetime import datetime
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter

import boto3
from pyspark.sql import SparkSession
import pyspark.sql.types as T
from docling.datamodel.document import InputDocument
from docling.document_converter import DocumentConverter
from langchain_ollama import OllamaEmbeddings


# Spark configs
THREADS = "local[4]"
DRIVER_MEMORY = "8g"
SHUFFLE_PARTITIONS = "4"

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
    .master(THREADS)
    .config("spark.driver.memory", DRIVER_MEMORY)
    .config("spark.sql.shuffle.partitions", SHUFFLE_PARTITIONS)
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
    if source_path.startswith("s3://"):
        # Read from S3 and save to temporary file
        s3_file = read_from_s3(source_path)
        temp_file = f"/tmp/{source_path.split('/')[-1]}"
        with open(temp_file, "wb") as f:
            f.write(s3_file.read())
        try:
            result = converter.convert(temp_file)
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)
    else:
        # Read from local file
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


def prepare_queries(
    collection: chromadb.Collection,
    model: OllamaEmbeddings,
    queries: List[str],
) -> List[Dict[str, Any]]:
    """Run queries and prepare results in json format."""
    all_results = []

    for query in queries:
        query_embedding = model.embed_documents([query])[0]
        results = collection.query(
            query_embeddings=[query_embedding], n_results=3
        )
        query_result = {
            "processed_at": datetime.now().isoformat(),
            "query": query,
            "results": [
                {
                    "text": doc,
                    "similarity": sim,
                }
                for doc, sim in zip(
                    results["documents"][0], results["distances"][0]
                )
            ],
        }
        all_results.append(query_result)

    return all_results


def save_json_data(
    data: List[Dict[str, Any]], file_path: str, overwrite: bool = True
) -> None:
    """Save data to a JSONL file (one JSON object per line)."""
    if not overwrite and os.path.exists(file_path):
        raise FileExistsError(
            f"File {file_path} already exists and overwrite=False"
        )

    # Create a temporary file
    temp_file = f"/tmp/{os.path.basename(file_path)}"

    # Write to temporary file
    with open(temp_file, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

    # If it's an S3 path, upload the file
    if file_path.startswith(("s3://", "s3a://")):
        write_to_s3(temp_file, file_path)
        # Clean up temporary file
        os.remove(temp_file)
    else:
        # Move the temporary file to the final location
        os.rename(temp_file, file_path)
