import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from prompts import prompt  # Ensure prompts.py is in the same directory

# 1. Configuration & Model Setup
load_dotenv()
api_key = os.getenv("access_token")

# Initialize the model
model = ChatGoogleGenerativeAI(
    model="gemma-3-27b-it",
    temperature=1.0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=api_key,
)

# Define the LangChain Expression Language (LCEL) chain
chain = prompt | model

def main():
    st.set_page_config(page_title="Oncology AI Assistant", layout="wide") # Changed to "wide"
    
    st.title("🩺 Patient Information Portal")
    st.subheader("Generate AI insights based on clinical data.")
    st.divider()

    # Create two main columns: Left for Inputs, Right for Results
    input_col, result_col = st.columns([1, 1], gap="large")

    # --- LEFT COLUMN: INPUTS ---
    with input_col:
        st.markdown("### Patient Details")
        cancer_type = st.text_input("Cancer Type", placeholder="e.g., Breast Cancer")
        
        past_medicine = st.text_area("Past Medicine", height=100)
        past_techniques = st.text_area("Past Techniques", height=100)

        current_stage = st.selectbox(
            "Current Stage", 
            ["Stage I", "Stage II", "Stage III", "Stage IV", "In Remission", "Unknown"]
        )
        
        other_details = st.text_area("Other Details", height=100)
        patient_question = st.text_input("Patient Question")
        
        generate_btn = st.button("Generate Result", use_container_width=True)

    # --- RIGHT COLUMN: RESULTS ---
    with result_col:
        st.markdown("### AI Analysis")
        if generate_btn:
            if not cancer_type or not patient_question:
                st.warning("Please fill in the Cancer Type and Patient Question.")
            else:
                with st.spinner("Analyzing data..."):
                    try:
                        input_data = {
                            "cancer_type": cancer_type,
                            "past_medicine": past_medicine,
                            "Past_Techniques": past_techniques,
                            "Current_Stage": current_stage,
                            "Other_Details": other_details,
                            "Patient_Question": patient_question
                        }
                        
                        result = chain.invoke(input_data)
                        
                        st.success("Analysis Complete")
                        st.markdown(result.content)
                    
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
        else:
            st.info("Fill in the details on the left and click 'Generate Result' to see the AI response here.")

if __name__ == "__main__":
    main()