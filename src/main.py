import os
from typing import List, Dict, Any, Tuple

from langchain_ollama import OllamaEmbeddings

from .helpers import (
    get_collection_name,
    parse_pdf,
    get_text_content,
    get_chunks,
    get_ids,
    get_metadata,
    get_embeddings,
    deduplicate_data,
    create_dataframe,
    create_iceberg_table,
    save_json_data,
    prepare_queries,
    store_in_postgres,
    spark,
)

from .queries import QUERIES

INPUT_PATH = "./data/input/client=client_name_1/Example_DCL.pdf"
COLLECTION_NAME = get_collection_name(INPUT_PATH)
ANSWERS_PATH = f"./data/answers/client={COLLECTION_NAME}/answers.jsonl"

CHUNK_SIZE = 200
CHUNK_OVERLAP = 20


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


def main() -> None:
    """Process PDF, transform data, store in PostgreSQL, and run queries."""
    # Process document and generate embeddings
    ids, chunks, metadatas, embeddings, model = process_document()

    df = create_dataframe(ids, chunks, metadatas, embeddings)

    iceberg_tbl_name = "documents"
    create_iceberg_table(df, iceberg_tbl_name)

    # Load DataFrame from Iceberg table
    df_loaded = spark.table(iceberg_tbl_name)

    df_deduplicated = deduplicate_data(df_loaded)
    print(f"✅ Deduplicated DataFrame in {iceberg_tbl_name}")

    # Store in PostgreSQL using LangChain's PGVectorStore
    store_in_postgres(COLLECTION_NAME, df_deduplicated, model)

    # Run queries and save answers
    answers = prepare_queries(COLLECTION_NAME, QUERIES, model)
    save_json_data(answers, ANSWERS_PATH)
    print(f"✅ Saved answers in {ANSWERS_PATH}")
    print("✅ Process completed!")

    spark.stop()
    print("✅ Spark session stopped.")


if __name__ == "__main__":
    main()
