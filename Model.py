import numpy as np 
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
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import GPT4All

chat_history = []
llm = None
llm_embeddings = None
conversation_retrieval_chain = None
#db = None
db2 = None
db3 = None
is_trained = False


# Initialize the language model and do the embedding stuff
def initialize_llm():
    global llm
    """Initializes the language model and embeddings.

    Returns:
        tuple: A tuple containing the initialized LLM and embeddings objects.
    """

    local_path = (
        "./gpt4all-falcon.gguf"  # replace with your desired local file path
    )

    # Callbacks support token-wise streaming
    callbacks = [StreamingStdOutCallbackHandler()]

    # Verbose is required to pass to the callback manager
    llm = GPT4All(model=local_path, callbacks=callbacks, verbose=True)
    return llm

# Process a Document 
def process_hf_dataset(dataset_name, page_content_column, name, llm):
    global llm_embeddings, conversation_retrieval_chain, db2, db3, is_trained
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
    #db = Chroma.from_documents(texts, llm_embeddings)

    # Save to the disk
    db2 = Chroma.from_documents(texts, llm_embeddings, persist_directory="./chroma_db")

    # load from disk
    db3 = Chroma(persist_directory="./chroma_db", embedding_function=llm_embeddings)
    retriever = db3.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    conversation_retrieval_chain = ConversationalRetrievalChain.from_llm(llm, retriever)
    is_trained = True
    return conversation_retrieval_chain


# User Prompt
def process_prompt(prompt):
   global conversation_retrieval_chain, llm, chat_history

   if is_trained:
      result = conversation_retrieval_chain({"question": prompt, "chat_history": chat_history})
   else:
      result = llm(prompt)

   chat_history.append((prompt, result['answer']))
   return result['answer']






