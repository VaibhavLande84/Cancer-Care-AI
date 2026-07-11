import os
import json
from pydantic import BaseModel
from dotenv import load_dotenv
from prompts import prompt
from fastapi import FastAPI, HTTPException
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# New RAG Architecture Components
# ---------------------------------------------------------------------------
from entity_linker import link_entities
from patient_state import PatientState, ConditionEntry, MedicationEntry
from medlineplus_retriever import retrieve_medical_context
from dailymed_retriever import retrieve_drug_info
from Doc_retriver import get_docs

app = FastAPI(
    title="Cancer Care AI — RAG Backend",
    description="RAG-powered API using MedlinePlus, DailyMed/OpenFDA, and Entity Linking (NLM Clinical Tables)",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# 1. Configuration & Model Setup (provider-agnostic, reads from .env)
# ---------------------------------------------------------------------------
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

from model_factory import create_llm

model = create_llm()
chain = prompt | model

# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------

class Query_input(BaseModel):
    query: str
    cancer_type: str = ""
    past_medicine: str = ""
    Past_Techniques: str = ""
    Current_Stage: str = ""
    Other_Details: str = ""
    Patient_Question: str = ""
    clinical_docs_analysis: str = ""
    # Optional: pass a pre-built patient state JSON to continue a session
    patient_state_json: Optional[str] = None


class Query_output(BaseModel):
    response: str
    patient_state: dict = {}
    codes_found: list[str] = []


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/generate", response_model=Query_output)
async def generate_response(request: Query_input):
    # Ensure at least one API key is configured
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")
            or os.getenv("ANTHROPIC_API_KEY") or os.getenv("GROQ_API_KEY")
            or os.getenv("access_token")):
        raise HTTPException(
            status_code=500,
            detail="No API key found in .env. Set one for your provider (e.g. GEMINI_API_KEY, OPENAI_API_KEY).",
        )

    try:
        # ------------------------------------------------------------------
        # STEP 1: Restore or create patient state
        # ------------------------------------------------------------------
        if request.patient_state_json:
            ps = PatientState.from_dict(json.loads(request.patient_state_json))
        else:
            ps = PatientState(patient_id="api-user")

        ps.touch()
        if request.cancer_type:
            ps.cancer_type = request.cancer_type
        if request.past_medicine:
            ps.past_medicine = request.past_medicine
        if request.Past_Techniques:
            ps.past_techniques = request.Past_Techniques
        if request.Current_Stage:
            ps.current_stage = request.Current_Stage
        if request.Other_Details:
            ps.other_details = request.Other_Details

        # ------------------------------------------------------------------
        # STEP 2: Entity link the question
        # ------------------------------------------------------------------
        linking_result = link_entities(request.Patient_Question or request.query)
        for c in linking_result.conditions:
            if c.code:
                ps.add_condition(c.code, c.name)
        for m in linking_result.medications:
            if m.code:
                ps.add_medication(m.code, m.name)

        linking_context = linking_result.to_prompt_context()

        # ------------------------------------------------------------------
        # STEP 3: Retrieve MedlinePlus context
        # ------------------------------------------------------------------
        patient_codes = [c.icd10_code for c in ps.active_conditions]
        rag_query = f"{request.cancer_type} {request.Patient_Question or request.query}"
        medline_result = retrieve_medical_context(
            query=rag_query,
            patient_codes=patient_codes,
        )
        medlineplus_context = medline_result["context_blob"]

        # Supplementary: Wikipedia / Arxiv
        wiki_text = ""
        research_text = ""
        try:
            docs_wiki, docs_research = get_docs(rag_query)
            wiki_text = "\n\n".join(d.page_content for d in docs_wiki) if docs_wiki else ""
            research_text = "\n\n".join(d.page_content for d in docs_research) if docs_research else ""
        except Exception:
            wiki_text = ""
            research_text = ""
        if wiki_text or research_text:
            medlineplus_context += "\n\n### Supplementary\n"
            if wiki_text:
                medlineplus_context += f"\nWikipedia:\n{wiki_text}\n"
            if research_text:
                medlineplus_context += f"\nResearch (Arxiv):\n{research_text}\n"

        # ------------------------------------------------------------------
        # STEP 4: Retrieve drug labels
        # ------------------------------------------------------------------
        dailymed_context = ""
        for med in ps.current_medications:
            if med.rxnorm_id:
                drug_info = retrieve_drug_info(
                    drug_name=med.brand_name,
                    rxcui=med.rxnorm_id,
                )
                dailymed_context += drug_info["context_blob"] + "\n\n"
        if not dailymed_context:
            dailymed_context = "(No drug label information was retrieved.)"

        # ------------------------------------------------------------------
        # STEP 5: Build prompt inputs and invoke chain
        # ------------------------------------------------------------------
        patient_state_block = ps.to_prompt_block()

        input_data = {
            "patient_state_block": patient_state_block,
            "linking_context": linking_context,
            "medlineplus_context": medlineplus_context,
            "dailymed_context": dailymed_context,
            "clinical_docs_analysis": request.clinical_docs_analysis or "(No clinical documents analysis provided.)",
            "cancer_type": request.cancer_type,
            "past_medicine": request.past_medicine,
            "Past_Techniques": request.Past_Techniques,
            "Current_Stage": request.Current_Stage,
            "Other_Details": request.Other_Details,
            "Patient_Question": request.Patient_Question or request.query,
        }

        response = chain.invoke(input_data)

        # ------------------------------------------------------------------
        # STEP 6: Record interaction
        # ------------------------------------------------------------------
        ps.add_interaction(
            query=request.Patient_Question or request.query,
            response_summary=response.content[:200],
            codes=medline_result["codes_found"],
        )

        return Query_output(
            response=response.content,
            patient_state=ps.to_dict(),
            codes_found=medline_result["codes_found"],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def read_root():
    return {
        "message": "Cancer Care AI — RAG Backend v2.0",
        "docs": "/docs",
        "endpoints": {
            "POST /api/generate": "Generate response with RAG, entity linking & patient state",
        },
    }
