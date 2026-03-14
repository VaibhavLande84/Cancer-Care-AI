import os
from langchain_huggingface import HuggingFacePipeline
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Get the token
hf_token = os.getenv("access_token")

# Verify token is loaded
if not hf_token:
    raise ValueError("access_token not found in .env file")

print(f"✓ Token loaded successfully")

try:
    print("Initializing local HuggingFace Pipeline...")
    print("(Downloading model on first run - this may take a few minutes)")
    
    # Using HuggingFacePipeline which runs locally
    llm = HuggingFacePipeline.from_model_id(
        model_id="gpt2",  # Small, fast model for testing
        task="text-generation",
        pipeline_kwargs={
            "temperature": 0.0,
            "max_length": 100
        }
    )
    
    print("✓ LLM initialized")
    print("Invoking model...")
    result = llm.invoke("who are you")
    
    print("✓ Response received:")
    print(result)
    
except Exception as e:
    print(f"✗ Error: {type(e).__name__}")
    print(f"Message: {str(e)}")
    import traceback
    traceback.print_exc()