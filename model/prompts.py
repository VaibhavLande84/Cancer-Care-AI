from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
import os

llm = HuggingFaceEndpoint(
repo_id="zai-org/GLM-5",
task="text-generation",
max_new_tokens=512,
temperature=0.1,
huggingfacehub_api_token='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'
)
    

st.header('cancer helpbot')
user_input=st.text_input('enter ur prompt')
if st.button('summerize'):
    result=llm.invoke(user_input)
    st.text(result)