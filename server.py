import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import the necessary methods from your Model module
from Model import initialize_llm, process_hf_dataset

# Initialize FastAPI app
app = FastAPI()

# Initialize the language model when the app starts
llm = initialize_llm()

@app.get("/")
def check():
    return {"message": "Hello World!"}

# Define request body models
class TrainModelRequest(BaseModel):
    dataset_name: str
    page_content_column: str
    name: str
    
class ProcessPromptRequest(BaseModel):
    prompt: str

# Class for processing prompts
class PromptProcessor:
    def __init__(self, llm):
        self.llm = llm
        self.conversation_retrieval_chain = None
        self.chat_history = []

    def process_prompt(self, prompt):
        if self.conversation_retrieval_chain:
            result = self.conversation_retrieval_chain({"question": prompt, "chat_history": self.chat_history})
        else:
            result = self.llm(prompt)

        self.chat_history.append((prompt, result['answer']))
        return result['answer']

# Create an instance of the PromptProcessor class
prompt_processor = PromptProcessor(llm)

# Define endpoint to train model with dataset name
@app.post("/train_model/")
async def train_model(request: TrainModelRequest):
    try:
        # Call method to process dataset
        global conversation_retrieval_chain  # Access global variable for chain
        conversation_retrieval_chain = process_hf_dataset(request.dataset_name, page_content_column=request.page_content_column, name=request.name, llm=llm)
        prompt_processor.conversation_retrieval_chain = conversation_retrieval_chain  # Update in prompt processor
        return {"message": f"Model trained with dataset: {request.dataset_name}, page_content_column: {request.page_content_column}, and name: {request.name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Define endpoint for processing user prompt using the class
@app.post("/process_prompt/")
async def process_user_prompt(request: ProcessPromptRequest):
    try:
        # Call method to process user prompt using the class
        response = prompt_processor.process_prompt(request.prompt)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
