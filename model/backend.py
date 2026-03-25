import os
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from prompts import prompt  # Ensure prompts.py is in the same director
from Doc_retriver import get_docs
from fastapi import FastAPI, HTTPException
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
app=FastAPI(
    title="Gemini API Wrapper",
    description="A FastAPI backend powered by Langchain and Google Gemini",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)
# 1. Configuration & Model Setup
dotenv_path = os.path.join(os.path.dirname("C:\Users\vaibh\OneDrive\Desktop\ML\Projects\ecommendation-system\Cancer-Care-AI\model\.env"), '.env')
load_dotenv(dotenv_path)
api_key = os.getenv("access_token")

HF_MED_MODEL_NAME = os.getenv("HF_MED_MODEL_NAME", "google/medgemma-4b-it")

# Initialize the model
model = ChatGoogleGenerativeAI(
    model="gemma-3-27b-it",
    temperature=1.0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=api_key,
)
chain = prompt | model

class Query_input(BaseModel):
    query:str
    cancer_type:str
    past_medicine:str
    Past_Techniques:str
    Current_Stage:str
    Other_Details:str
    Patient_Question:str
    retrieved_docs:str
    clinical_docs_analysis:str
    

class Query_output(BaseModel):
    response:str

@app.post("/api/generate",response_model=Query_output)
async def generate_response(request: Query_input):
    if not os.environ.get("GOOGLE_API_KEY"):
        raise HTTPException(
            status_code=500, 
            detail="GOOGLE_API_KEY is not set in the environment variables."
        )
    
    try:
        # Retrieve context documents based on the user's query
        wiki_docs, research_docs = get_docs(request.query)
        all_docs = wiki_docs + research_docs
        retrieved_context = "\n\n".join([doc.page_content for doc in all_docs])

        # Invoke the chain with the prompt variables
        response = chain.invoke({
            "cancer_type": request.cancer_type,
            "past_medicine": request.past_medicine,
            "Past_Techniques": request.Past_Techniques,
            "Current_Stage": request.Current_Stage,
            "Other_Details": request.Other_Details,
            "Patient_Question": request.Patient_Question,
            "retrieved_docs": retrieved_context,
            "clinical_docs_analysis": request.clinical_docs_analysis
        })
        
        return Query_output(response=response.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Welcome to the Gemini API Backend. Visit /docs to see the API documentation."}
