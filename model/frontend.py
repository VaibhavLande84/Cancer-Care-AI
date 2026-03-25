import streamlit as st
import requests

# Set Streamlit page config
st.set_page_config(page_title="Cancer Care AI Assistant", page_icon="🎗️", layout="wide")

st.title("🎗️ Cancer Care AI Assistant")
st.markdown("This assistant provides information and emotional support to cancer patients using Gemini and medical literature.")

st.header("Patient Information")

col1, col2 = st.columns(2)

with col1:
    cancer_type = st.text_input("Cancer Type", placeholder="e.g., Stage III Non-Small Cell Lung Cancer")
    past_medicine = st.text_area("Past Medicine", placeholder="e.g., Chemotherapy, Tamoxifen")
    past_techniques = st.text_area("Past Techniques", placeholder="e.g., Surgery, Immunotherapy")
    current_stage = st.text_input("Current Stage", placeholder="e.g., Stage IV, Remission")

with col2:
    other_details = st.text_area("Other Details", placeholder="e.g., Experiencing severe fatigue...")
    clinical_docs_analysis = st.text_area("Clinical Docs Analysis", placeholder="Paste any doctor reports or prescription text analysis here...")
    
st.header("Ask the Assistant")
patient_question = st.text_input("Your Question", placeholder="What are my treatment options now?")

if st.button("Generate Response"):
    if not patient_question or not cancer_type:
        st.warning("Please provide at least the Cancer Type and your Question.")
    else:
        with st.spinner("Analyzing and retrieving relevant medical literature..."):
            # Prepare the payload for FastAPI
            payload = {
                "query": patient_question,
                "cancer_type": cancer_type,
                "past_medicine": past_medicine,
                "Past_Techniques": past_techniques,
                "Current_Stage": current_stage,
                "Other_Details": other_details,
                "Patient_Question": patient_question,
                "retrieved_docs": "",
                "clinical_docs_analysis": clinical_docs_analysis
            }
            
            try:
                # Assuming the FastAPI runs on localhost:8000
                response = requests.post("http://localhost:8000//api/generate", json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    st.success("Analysis Complete")
                    st.write("### AI Response")
                    st.write(data.get("response", "No response received."))
                else:
                    st.error(f"Error: Backend returned status code {response.status_code}")
                    st.error(response.text)
            except requests.exceptions.ConnectionError:
                st.error("Failed to connect to the backend server. Please ensure the FastAPI backend is running on http://127.0.0.1:8000/")
