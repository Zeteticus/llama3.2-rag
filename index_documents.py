import os
import logging
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Weaviate
from langchain_community.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm
import weaviate
from weaviate.embedded import EmbeddedOptions
from transformers import AutoTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def process_text_file(file_path):
    try:
        logger.info(f"Processing text file: {file_path}")
        loader = TextLoader(file_path, encoding='utf-8')
        document = loader.load()

        max_len = 500  # Reduced from 700 to stay within BERT's limit
        chunks = []

        for doc in document:
            tokens = tokenizer.encode(doc.page_content, add_special_tokens=False)
            
            # Split tokens into chunks of max_len
            for i in range(0, len(tokens), max_len):
                chunk_tokens = tokens[i:i + max_len]
                chunk_text = tokenizer.decode(chunk_tokens)
                chunks.append(chunk_text.strip())

        logger.info(f"Extracted {len(chunks)} chunks from {file_path}")
        return chunks
    except Exception as e:
        logger.error(f"Error processing text file {file_path}: {str(e)}")
        return []

def create_or_get_vector_store(chunks=None, sources=None):
    client = weaviate.Client(
        embedded_options=EmbeddedOptions(
            persistence_data_path="./weaviate_data"
        )
    )

    embeddings = HuggingFaceEmbeddings()

    # Check if the schema already exists
    try:
        client.schema.get("Document")
    except weaviate.exceptions.UnexpectedStatusCodeException:
        # If the schema doesn't exist, create it
        class_obj = {
            "class": "Document",
            "vectorizer": "none",  # We'll use HuggingFaceEmbeddings
            "vectorIndexType": "hnsw",
            "properties": [
                {
                    "name": "text",
                    "dataType": ["text"],
                },
                {
                    "name": "source",
                    "dataType": ["text"],
                }
            ]
        }
        client.schema.create_class(class_obj)

    vector_store = Weaviate(client, "Document", "text", embedding=embeddings, by_text=False)

    if chunks and sources:
        # Process chunks in batches
        batch_size = 500  # You can adjust this value
        for i in tqdm(range(0, len(chunks), batch_size), desc="Adding chunks to Weaviate"):
            batch_chunks = chunks[i:i+batch_size]
            batch_sources = sources[i:i+batch_size]
            vector_store.add_texts(texts=batch_chunks, metadatas=[{"source": source} for source in batch_sources])

    return vector_store

def main():
    text_directory = "content"
    all_chunks = []
    all_sources = []

    # Check if directory exists
    if not os.path.isdir(text_directory):
        logger.error(f"Directory not found: {text_directory}")
        return

    # Process all text files in the directory
    for filename in os.listdir(text_directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(text_directory, filename)
            chunks = process_text_file(file_path)
            all_chunks.extend(chunks)
            all_sources.extend([filename] * len(chunks))

    logger.info(f"Total chunks extracted: {len(all_chunks)}")

    if not all_chunks:
        logger.error("No chunks were extracted from any text file. Aborting.")
        return

    # Create or get vector store
    vector_store = create_or_get_vector_store(all_chunks, all_sources)
    if vector_store is None:
        return

    logger.info("Indexing completed successfully.")

if __name__ == "__main__":
    main()
