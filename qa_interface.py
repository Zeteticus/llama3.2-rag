import logging
import textwrap
from langchain_community.llms import Ollama
from langchain_weaviate import WeaviateVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import weaviate
from weaviate.embedded import EmbeddedOptions
import os
import ollama

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_vector_store():
    # For Weaviate v4 with embedded mode - only use embedded_options, not connection_params
    client = weaviate.WeaviateClient(
        embedded_options=EmbeddedOptions(
            persistence_data_path="./weaviate_data"
        )
    )
    
    # Connect the client (required in v4)
    client.connect()
    
    embeddings = HuggingFaceEmbeddings()
    return WeaviateVectorStore(client=client, index_name="Document", text_key="text", embedding=embeddings)

def setup_rag(vector_store):
    if vector_store is None:
        logger.error("Vector store is None. Cannot set up RAG.")
        return None
    
    llm = Ollama(model="llama3.2")
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 6}),
        memory=memory
    )
    return qa_chain

def print_separator():
    print("~" * 125)

def format_answer(answer, width=90, initial_indent="  ", subsequent_indent="    "):
    wrapped_lines = textwrap.wrap(answer, width=width, initial_indent=initial_indent,
                                  subsequent_indent=subsequent_indent, break_long_words=False,
                                  replace_whitespace=False)
    return "\n".join(wrapped_lines)

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
            print_separator()
            answer = qa_chain.invoke({"question": question})
            formatted_answer = format_answer(answer['answer'])
            print("Answer:")
            print(formatted_answer)
            print_separator()
            print()
        except Exception as e:
            logger.error(f"Error during question answering: {str(e)}")

if __name__ == "__main__":
    main()
