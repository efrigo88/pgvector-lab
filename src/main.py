from datetime import datetime
from typing import List, Dict, Any, Tuple

from chromadb import Collection
from langchain_ollama import OllamaEmbeddings
import pyspark.sql.functions as F
from pyspark.sql import DataFrame

from .helpers import (
    get_client,
    get_collection,
    parse_pdf,
    get_text_content,
    get_chunks,
    get_ids,
    get_metadata,
    get_embeddings,
    prepare_queries,
    save_json_data,
    spark,
    schema,
)

from .queries import QUERIES

# Paths
ENV = "dev"
BUCKET_NAME = (
    f"s3://docker-pipeline-ml-ec2-lab-{ENV}-"
    f"{datetime.now().strftime('%Y%m%d')}"
)
SPARK_BUCKET_NAME = BUCKET_NAME.replace("s3://", "s3a://")
INPUT_PATH = f"{BUCKET_NAME}/data/input/Example_DCL.pdf"
OUTPUT_PATH = f"{SPARK_BUCKET_NAME}/data/output/delta_table"
JSONL_PATH = f"{SPARK_BUCKET_NAME}/data/output/jsonl_file"
ANSWERS_PATH = f"{SPARK_BUCKET_NAME}/data/answers/answers.jsonl"

CHUNK_SIZE = 750
CHUNK_OVERLAP = 100
OLLAMA_HOST = "http://ollama:11434"


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

    chunks = get_chunks(text_content, CHUNK_SIZE, CHUNK_OVERLAP)
    ids = get_ids(chunks, INPUT_PATH)
    metadatas = get_metadata(chunks, doc, INPUT_PATH)
    print("✅ Chunks, IDs and Metadatas generated.")

    model = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_HOST)
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


def store_in_chromadb(df_loaded: DataFrame) -> Collection:
    """Store data in ChromaDB"""
    client = get_client()
    collection = get_collection(client)

    rows = df_loaded.select("id", "chunk", "metadata", "embeddings").collect()
    id_list = [row.id for row in rows]
    doc_list = [row.chunk for row in rows]
    meta_list = [row.metadata.asDict() for row in rows]
    embed_list = [row.embeddings for row in rows]

    collection.upsert(
        ids=id_list,
        documents=doc_list,
        metadatas=meta_list,
        embeddings=embed_list,
    )
    # Only for development purposes
    print(f"✅ Upserted {df_loaded.count()} chunks in ChromaDB.")

    all_data = collection.get()
    total_docs = len(all_data["ids"])
    print(f"📊 Total documents in ChromaDB: {total_docs}")
    return collection


def main() -> None:
    """Process PDF, transform data, store in ChromaDB, and run queries."""
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

    collection = store_in_chromadb(df_deduplicated)

    # Run queries and save answers
    answers = prepare_queries(collection, model, QUERIES)
    save_json_data(answers, ANSWERS_PATH)
    print(f"✅ Saved answers in {ANSWERS_PATH}")
    print("✅ Process completed!")

    spark.stop()
    print("✅ Spark session stopped.")


if __name__ == "__main__":
    main()
