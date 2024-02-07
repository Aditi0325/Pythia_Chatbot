import awadb
from langchain_community.llms import HuggingFaceHub
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import AwaEmbeddings

chat_history = []
llm = None
llm_embeddings = None
conversationalRetrievalChain = None

# Initialize the llm mocdel and do the embedding stuff
def initialize_llm():
   global llm, llm_embeddings
   """Initializes the language model and embeddings.

   Returns:
       tuple: A tuple containing the initialized LLM and embeddings objects.
   """

   llm_embeddings = AwaEmbeddings()
   llm_embeddings.set_model("all-mpnet-base-v2")

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

# Processing the pdf document
def process_pdf(pdf_path, llm):
    global llm_embeddings, conversation_retrieval_chain
    """Processes a PDF file and creates a conversational retrieval chain.

    Args:
        pdf_path (str): The path to the PDF file.
        llm_embeddings (AwaEmbeddings): The initialized embeddings object.

    Returns:
        ConversationalRetrievalChain: The initialized conversational retrieval chain.
    """

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    db = FAISS.from_documents(texts, llm_embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    conversation_retrieval_chain = ConversationalRetrievalChain.from_llm(llm, retriever)


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

# Initialize the language model
initialize_llm()

chain = process_pdf("data/DeepLearning.pdf", llm)



