import os
import logging
from langchain_community.llms import Ollama
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_community.vectorstores import Weaviate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
import tiktoken
from tqdm import tqdm
import weaviate
from weaviate.embedded import EmbeddedOptions

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to load text file and split into chunks using TikToken
def process_text_file(file_path):
    try:
        logger.info(f"Processing text file: {file_path}")
        loader = TextLoader(file_path, encoding='utf-8')
        document = loader.load()

        # Initialize TikToken-based splitter
        tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
        text_splitter = TokenTextSplitter(
            chunk_size=500,  # Number of tokens per chunk
            chunk_overlap=50,  # Number of overlapping tokens
            encoding_name="cl100k_base"  # Specify the tokenizer
        )

        chunks = text_splitter.split_documents(document)
        logger.info(f"Extracted {len(chunks)} chunks from {file_path}")
        return chunks
    except Exception as e:
        logger.error(f"Error processing text file {file_path}: {str(e)}")
        return []

def create_or_get_vector_store(chunks=None):
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

    if chunks:
        # Process chunks in batches
        batch_size = 500  # You can adjust this value
        for i in tqdm(range(0, len(chunks), batch_size), desc="Adding chunks to Weaviate"):
            batch = chunks[i:i+batch_size]
            vector_store.add_documents(documents=batch)

    return vector_store

# Function to set up RAG
def setup_rag(vector_store):
    if vector_store is None:
        logger.error("Vector store is None. Cannot set up RAG.")
        return None

    llm = Ollama(model="llama3.1")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 5})
    )
    return qa_chain

# Main function
def main():
    text_directory = "content"
    all_chunks = []

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

    logger.info(f"Total chunks extracted: {len(all_chunks)}")

    if not all_chunks:
        logger.error("No chunks were extracted from any text file. Aborting.")
        return

    # Create or get vector store
    vector_store = create_or_get_vector_store(all_chunks)
    if vector_store is None:
        return

    # Set up RAG
    qa_chain = setup_rag(vector_store)
    if qa_chain is None:
        return

    # Interactive question-answering loop
    while True:
        question = input("Ask a question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        try:
            answer = qa_chain.invoke({"query": question})
            print(f"Answer: {answer['result']}\n")
        except Exception as e:
            logger.error(f"Error during question answering: {str(e)}")

if __name__ == "__main__":
    main()
