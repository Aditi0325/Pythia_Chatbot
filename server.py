from fastapi import FastAPI
import uvicorn
from Model import initialize_llm, process_pdf, process_prompt

app = FastAPI()

# Initialize the language model and processing chain
initialize_llm()

@app.get("/")
def check():
    return {"message": "Hello World!"}

# Define API endpoints
@app.post("/upload-pdf")
async def upload_pdf():
    # Specify the path to the PDF file
    pdf_path = "data/DeepLearning.pdf"

    # Process the PDF and build the retrieval chain
    process_pdf(pdf_path)

    return {"message": "PDF processed successfully"}

@app.post("/prompt")
async def process_user_prompt(prompt_data: dict):
    # Extract prompt from the request body
    prompt = prompt_data.get('prompt', '')

    # Process user prompt and return response
    response = process_prompt(prompt)
    return {"response": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
