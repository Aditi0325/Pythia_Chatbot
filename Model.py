import numpy as np 
from langchain_community.llms import HuggingFaceHub
from langchain_community.document_loaders.hugging_face_dataset import (
    HuggingFaceDatasetLoader,
)
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)

chat_history = []
llm = None
llm_embeddings = None
conversation_retrieval_chain = None
db = None
db2 = None
db3 = None

# Initialize the language model and do the embedding stuff
def initialize_llm():
    global llm, llm_embeddings
    """Initializes the language model and embeddings.

    Returns:
        tuple: A tuple containing the initialized LLM and embeddings objects.
    """

   

    llm = HuggingFaceHub(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        model_kwargs={
            "max_new_tokens": 512,
            "top_k": 30,
            "temperature": 0.1,
            "repetition_penalty": 1.03,
        },
    )
    return llm

# Process a Document 
def process_hf_dataset(dataset_name, page_content_column, name, llm):
    global llm_embeddings, conversation_retrieval_chain, db, db2, db3
    """Processes a dataset from Hugging Face and creates a conversational retrieval chain.

    Args:
        dataset_name (str): The name of the Hugging Face dataset.
        page_content_column (str): The name of the column containing text content.
        name (str): The name of the dataset configuration.
        llm_embeddings (AwaEmbeddings): The initialized embeddings object.

    Returns:
        ConversationalRetrievalChain: The initialized conversational retrieval chain.
    """

    print(f"Dataset Name: {dataset_name}, Page Content Column: {page_content_column}, Name: {name}")

    loader = HuggingFaceDatasetLoader(dataset_name, page_content_column, name)
    documents = loader.load()
    filtered_documents =  filter_complex_metadata(documents)


    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(filtered_documents)

    # create the open-source embedding function
    llm_embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Initialize and save Chroma vector store
    db = Chroma.from_documents(texts, llm_embeddings)

    # Save to the disk
    db2 = Chroma.from_documents(texts, llm_embeddings, persist_directory="./chroma_db")

    # load from disk
    db3 = Chroma(persist_directory="./chroma_db", embedding_function=llm_embeddings)
    retriever = db3.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    conversation_retrieval_chain = ConversationalRetrievalChain.from_llm(llm, retriever)
    return conversation_retrieval_chain

# User Prompt
def process_prompt(prompt):
   """Processes a user prompt and returns the response from the conversational retrieval chain.

   Args:
       prompt (str): The user's prompt.

   Returns:
       str: The response from the conversational retrieval chain.
   """

   global conversation_retrieval_chain
   global chat_history

   result = conversation_retrieval_chain({"question": prompt, "chat_history": chat_history})
   chat_history.append((prompt, result['answer']))

   return result['answer']
