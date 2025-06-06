import os
import json
from datetime import datetime
from typing import List, Dict, Any

import pyspark.sql.functions as F
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_postgres import PGEngine, PGVectorStore
from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.types as T
from docling.datamodel.document import InputDocument
from docling.document_converter import DocumentConverter
from langchain_ollama import OllamaEmbeddings


SCHEMA = T.StructType(
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
    .config(
        "spark.sql.extensions",
        "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
    )
    .config(
        "spark.sql.catalog.spark_catalog",
        "org.apache.iceberg.spark.SparkSessionCatalog",
    )
    .config("spark.sql.catalog.spark_catalog.type", "hadoop")
    .config("spark.sql.catalog.spark_catalog.warehouse", "./data/warehouse")
    .config(
        "spark.jars.packages",
        "org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.5.0,"
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


def get_collection_name(path: str) -> str:
    """Get the collection name from the source path."""
    return path.split("/")[-2].split("=")[-1]


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
        schema=SCHEMA,
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


def create_iceberg_table(df: DataFrame, table_name: str) -> None:
    """Create Iceberg table if it doesn't exist."""
    # Register DataFrame as a temporary view
    df.createOrReplaceTempView("temp_df")

    spark.sql(
        f"CREATE TABLE IF NOT EXISTS {table_name} "
        "USING iceberg AS SELECT * FROM temp_df LIMIT 0"
    )

    # Save DataFrame to Iceberg table
    df.write.format("iceberg").mode("overwrite").saveAsTable(table_name)
    print(f"✅ Saved Iceberg table {table_name}")


def save_json_data(
    data: List[Dict[str, Any]], file_path: str, overwrite: bool = True
) -> None:
    """Save data to a JSONL file (one JSON object per line)."""
    if not overwrite and os.path.exists(file_path):
        raise FileExistsError(
            f"File {file_path} already exists and overwrite=False"
        )

    # Create directory structure if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")


def get_db_connection_string() -> str:
    """Create a connection string for PostgreSQL."""
    return (
        f"postgresql+asyncpg://{os.getenv('POSTGRES_USER')}:"
        f"{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}:"
        f"{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
    )


def init_vector_store(
    collection_name: str, embeddings: OllamaEmbeddings
) -> PGEngine:
    """Initialize the vector store with LangChain's PGEngine."""
    connection_string = get_db_connection_string()
    engine = PGEngine.from_connection_string(url=connection_string)

    try:
        # Try to initialize the table, it will fail if it already exists
        engine.init_vectorstore_table(
            table_name=collection_name,
            vector_size=len(embeddings.embed_query("test")),
        )
    except Exception as e:
        if "already exists" not in str(e):
            raise
        print(f"ℹ️ Table '{collection_name}' already exists, continuing...")

    return engine


def store_in_postgres(
    collection_name: str, df: DataFrame, embeddings: OllamaEmbeddings
) -> None:
    """Store data in PostgreSQL using LangChain's PGVectorStore."""
    engine = init_vector_store(collection_name, embeddings)

    # Create vector store instance
    vector_store = PGVectorStore.create_sync(
        engine,
        embedding_service=embeddings,
        table_name=collection_name,
    )

    # Convert DataFrame rows to LangChain Documents
    documents = []
    for row in df.collect():
        doc = Document(
            page_content=row.chunk,
            metadata={
                "source": row.metadata["source"],
                "chunk_index": row.metadata["chunk_index"],
                "title": row.metadata["title"],
                "chunk_size": row.metadata["chunk_size"],
                "processed_at": row.processed_at.isoformat(),
                "processed_dt": row.processed_dt,
            },
        )
        documents.append(doc)

    vector_store.add_documents(documents)
    print(
        f"📊 Total documents in PostgreSQL collection '{collection_name}': {len(documents)}"
    )


def prepare_queries(
    collection_name: str,
    queries: List[str],
    embeddings: OllamaEmbeddings,
) -> List[Dict[str, Any]]:
    """Run queries and prepare results in json format using LangChain's PGVectorStore."""
    engine = init_vector_store(collection_name, embeddings)

    # Create vector store instance
    vector_store = PGVectorStore.create_sync(
        engine,
        embedding_service=embeddings,
        table_name=collection_name,
    )

    all_results = []

    for query in queries:
        # Perform similarity search using LangChain's PGVectorStore
        results = vector_store.similarity_search(
            query,
            k=3,
        )
        query_result = {
            "collection": collection_name,
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
