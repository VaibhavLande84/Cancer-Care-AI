import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from prompts import prompt  # Ensure prompts.py is in the same directory
from Doc_retriver import get_docs

from typing import List, Optional

# 1. Configuration & Model Setup
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
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

# Define the LangChain Expression Language (LCEL) chain
chain = prompt | model

def _read_uploaded_text(files: Optional[List[st.runtime.uploaded_file_manager.UploadedFile]]) -> str:
    if not files:
        return ""

    chunks: List[str] = []
    for f in files:
        name = getattr(f, "name", "uploaded_file")
        file_bytes = f.getvalue()
        lower = name.lower()

        if lower.endswith(".pdf"):
            try:
                from pypdf import PdfReader  # type: ignore
                import io

                reader = PdfReader(io.BytesIO(file_bytes))
                text = "\n".join((page.extract_text() or "") for page in reader.pages)
                chunks.append(f"--- {name} (pdf) ---\n{text}".strip())
            except Exception as e:
                chunks.append(f"--- {name} (pdf) ---\n(Could not read PDF: {e})")
        else:
            # Default: try decode as text
            try:
                text = file_bytes.decode("utf-8", errors="replace")
            except Exception:
                text = str(file_bytes)
            chunks.append(f"--- {name} ---\n{text}".strip())

    return "\n\n".join(chunks).strip()

def _medgemma_analyze(prescription_text: str, report_text: str, patient_question: str) -> str:
    content = []
    if prescription_text.strip():
        content.append(f"Prescription (patient input):\n{prescription_text.strip()}")
    if report_text.strip():
        content.append(f"Doctor reports / uploads:\n{report_text.strip()}")
    docs_blob = "\n\n".join(content).strip() or "(No patient documents provided.)"

    analysis_prompt = f"""You are a clinical assistant. Summarize and analyze the patient's provided prescription and doctor reports.

Rules:
- Do NOT give direct medical advice or dosing instructions.
- If a detail is missing/unclear, say it is missing.
- Output concise structured text with headings.

Patient question (for focus): {patient_question}

Patient documents:
{docs_blob}

Return:
1) Key extracted facts (diagnoses, meds, dates, labs, imaging, procedures)
2) Potential concerns / red flags to discuss with oncologist (non-alarmist)
3) Clarifying questions to ask the patient/doctor
"""

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        import torch  # type: ignore

        @st.cache_resource(show_spinner=False)
        def _load_hf_med_model():
            tok = AutoTokenizer.from_pretrained(HF_MED_MODEL_NAME)
            mdl = AutoModelForCausalLM.from_pretrained(
                HF_MED_MODEL_NAME,
                torch_dtype=getattr(torch, "float16", None) or torch.float32,
                device_map="auto",
            )
            return tok, mdl

        tok, mdl = _load_hf_med_model()

        try:
            chat = [{"role": "user", "content": analysis_prompt}]
            prompt_text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt_text = analysis_prompt

        inputs = tok(prompt_text, return_tensors="pt")
        inputs = {k: v.to(mdl.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = mdl.generate(
                **inputs,
                max_new_tokens=600,
                do_sample=False,
                temperature=0.0,
            )
        decoded = tok.decode(out[0], skip_special_tokens=True)
        return decoded.strip()
    except Exception as e:
        return (
            "HF medical document analysis was skipped because the Hugging Face model "
            f"`{HF_MED_MODEL_NAME}` could not be loaded or run locally.\n\n"
            f"Error: {e}\n\n"
            "Fix: `pip install transformers torch accelerate`. "
            "If you don't have a GPU/VRAM, choose a smaller HF model via `HF_MED_MODEL_NAME` in `.env`."
        )

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

        st.markdown("### Current Prescription & Doctor Reports (optional)")
        current_prescription = st.text_area(
            "Current Medical Prescription",
            height=120,
            placeholder="Paste your current medicines (name, dose, frequency) exactly as written.",
        )
        uploaded_reports = st.file_uploader(
            "Upload doctor reports (PDF/TXT)",
            type=["pdf", "txt"],
            accept_multiple_files=True,
        )
        
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
                        # RAG: retrieve documents relevant to cancer type and patient question
                        rag_query = f"{cancer_type} {patient_question}"
                        with st.spinner("Retrieving relevant literature..."):
                            docs_wiki, docs_research = get_docs(rag_query)
                        wiki_text = "\n\n".join(d.page_content for d in docs_wiki) if docs_wiki else "(No Wikipedia results)"
                        research_text = "\n\n".join(d.page_content for d in docs_research) if docs_research else "(No research papers found)"
                        retrieved_docs = f"### Wikipedia\n{wiki_text}\n\n### Research (Arxiv)\n{research_text}"

                        # MedGemma: analyze patient-provided prescription + reports
                        reports_text = _read_uploaded_text(uploaded_reports)
                        with st.spinner("Analyzing prescription & reports..."):
                            clinical_docs_analysis = _medgemma_analyze(
                                prescription_text=current_prescription,
                                report_text=reports_text,
                                patient_question=patient_question,
                            )

                        input_data = {
                            "cancer_type": cancer_type,
                            "past_medicine": past_medicine,
                            "Past_Techniques": past_techniques,
                            "Current_Stage": current_stage,
                            "Other_Details": other_details,
                            "Patient_Question": patient_question,
                            "retrieved_docs": retrieved_docs,
                            "clinical_docs_analysis": clinical_docs_analysis,
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