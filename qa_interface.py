import logging
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Weaviate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
import weaviate
from weaviate.embedded import EmbeddedOptions

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_vector_store():
    client = weaviate.Client(
        embedded_options=EmbeddedOptions(
            persistence_data_path="./weaviate_data"
        )
    )
    embeddings = HuggingFaceEmbeddings()
    return Weaviate(client, "Document", "text", embedding=embeddings, by_text=False)

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

def main():
    # Get vector store
    vector_store = get_vector_store()
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
