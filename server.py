import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Import the necessary methods from your Model module
from Model import initialize_llm, load_data, process_hf_dataset, process_prompt

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

# Class for processing prompts  
class ProcessPromptRequest(BaseModel):
    prompt: str

# Define endpoint to train model with dataset name
@app.post("/train_model/")
async def train_model(request: TrainModelRequest):
    try:
        # Call method to process dataset
        global conversation_retrieval_chain  # Access global variable for chain
        name = ''
        if(request.name == "0"):
            name = None
        else:
            name = request.name

        conversation_retrieval_chain = process_hf_dataset(request.dataset_name, page_content_column=request.page_content_column, name=name, llm=llm)
        conversation_retrieval_chain = conversation_retrieval_chain  # Update in prompt processor
        return {"message": f"Model trained with dataset: {request.dataset_name}, page_content_column: {request.page_content_column}, and name: {request.name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


@app.get("/loadData")
async def process_user_prompt():
    try:
        # Call method to process user prompt using the class
        load_data()
        return {"success": 'Loaded data'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

    

# Define endpoint for processing user prompt using the class
@app.post("/process_prompt/")
async def process_user_prompt(request: ProcessPromptRequest):
    try:
        # Call method to process user prompt using the class
        print(request.prompt)
        response = process_prompt(request.prompt)
        return {"response": response,
                headers: {
  "Access-Control-Allow-Origin": "*", 
  "Access-Control-Allow-Credentials": true,
  "Access-Control-Allow-Headers": "Origin,Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token,locale",
  "Access-Control-Allow-Methods": "POST, OPTIONS"
},}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

app.add_middleware(
  CORSMiddleware,
  allow_origins = ["*"],
  allow_methods = ["*"],
  allow_headers = ["*"]
)


if __name__ == "__main__":
   uvicorn.run("server:app", host="0.0.0.0", port=8000, log_level="info")
