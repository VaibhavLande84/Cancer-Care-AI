from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os

load_dotenv()
api_key = os.getenv("access_token")

model = ChatGoogleGenerativeAI(
    model="gemma-3-27b-it",
    temperature=1.0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=api_key,
)

answer=model.invoke("what is bone cancer")