# Cancer Care AI — RAG-Powered Medical Information Assistant

A **Retrieval-Augmented Generation (RAG)** system that grounds AI responses in official NIH / NLM medical literature. Designed for cancer patients and caregivers — provides accurate, empathetic, citation-backed answers using MedlinePlus, DailyMed, and openFDA data.

---

## Tech Stack

### Frontend
| Technology | Purpose |
|---|---|
| **Streamlit** | Web UI framework |
| **Python** | Core application logic |

### Backend & LLM Orchestration
| Technology | Purpose |
|---|---|
| **LangChain** | LCEL prompt-to-model pipeline |
| **model_factory.py** | Provider-agnostic LLM factory (OpenAI, Anthropic, Gemini, Groq, DeepSeek, Together, Ollama, OpenAI-compatible) |
| **FastAPI** | REST API (`/api/generate`) |

### Retrieval & RAG
| Technology | Purpose |
|---|---|
| **ChromaDB** | Local vector store (semantic search over MedlinePlus) |
| **NLM Clinical Tables API** | ICD-10-CM / RxNorm autocomplete search |
| **MedlinePlus Connect** | Code-based article lookup |
| **NLM Web Search API** | Free-text fallback retrieval |
| **DailyMed v2** | FDA drug label data |
| **openFDA** | Drug label search API |

### Vision
| Technology | Purpose |
|---|---|
| **Gemini / GPT-4o / Claude** | Multimodal LLM for prescription OCR with handwriting ambiguity detection |

### Data
| Technology | Purpose |
|---|---|
| **Patient State JSON** | Per-session structured patient memory |
| **ChromaDB** | Persistent vector index on disk |

---

## Technical Details

### Architecture
```
User Query → Entity Linker (NLM Clinical Tables) → ICD-10 / RxNorm Codes
                                                          ↓
                                              RAG Pipeline (4 stages):
                                                1. ChromaDB Vector Search
                                                2. MedlinePlus Connect
                                                3. NLM Web Search (fallback)
                                                4. DailyMed / openFDA
                                                          ↓
                                              LLM (any provider) + Patient State
                                                          ↓
                                              Response with Source Citations
```

### RAG Pipeline Stages

1. **Entity Linking** — `GET /api/icd10cm/v3/search?terms=lung+cancer` → `C34.90` | `GET /api/rxterms/v3/search?terms=advil` → RxNorm `209387`
2. **Vector Search** — Semantic similarity over ChromaDB (enriched query: text + codes)
3. **Connect Lookup** — `medlineplus.gov/mplus/connect/retrieve?mainSearchCriteria=...`
4. **Web Search Fallback** — `wsearch.nlm.nih.gov/ws/query?db=healthTopics&term=...`

### Patient State Schema
```json
{
  "patient_id": "P-42",
  "active_conditions": [{"icd10_code": "C34.90", "name": "Lung cancer"}],
  "current_medications": [{"rxnorm_id": "209387", "brand_name": "Lisinopril"}],
  "interaction_history": [{"user_query": "...", "ai_response_summary": "..."}]
}
```

### Provider Support
| Provider | Env Value | Key Var | 
|---|---|---|
| OpenAI | `openai` | `OPENAI_API_KEY` |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` |
| Gemini | `google-gemini` | `GEMINI_API_KEY` |
| Groq | `groq` | `GROQ_API_KEY` |
| DeepSeek | `deepseek` | `DEEPSEEK_API_KEY` |
| Together | `together` | `TOGETHER_API_KEY` |
| OpenAI-Compatible | `openai-compatible` | `OPENAI_COMPATIBLE_API_KEY` |
| Ollama (local) | `ollama` | *(none)* |

---

## Getting Started

```bash
git clone https://github.com/VaibhavLande84/Cancer-Care-AI.git
cd Cancer-Care-AI
pip install -r env/requirements.txt

# Set API key in model/.env then:
streamlit run model/app.py
```

### Environment Variables
```env
LLM_PROVIDER=openai-compatible
OPENAI_COMPATIBLE_API_KEY=sk-...
OPENAI_COMPATIBLE_BASE_URL=https://api.aicredits.in/v1
OPENAI_COMPATIBLE_MODEL_NAME=gpt-4o
```

---

## Project Structure

```
Cancer-Care-AI/
├── model/
│   ├── app.py                    # Streamlit UI
│   ├── backend.py                # FastAPI backend
│   ├── prompts.py                # System prompt
│   ├── model_factory.py          # LLM factory
│   ├── entity_linker.py          # ICD-10 / RxNorm
│   ├── patient_state.py          # JSON state
│   ├── medlineplus_retriever.py  # RAG pipeline
│   ├── dailymed_retriever.py     # Drug labels
│   ├── prescription_reader.py    # Vision OCR
│   ├── ingestion_pipeline.py     # XML → ChromaDB
│   ├── Doc_retriver.py           # Wikipedia/Arxiv
│   └── .env                      # Config
├── env/requirements.txt
└── README.md
```

---

## Requirements

```
langchain-openai, langchain-anthropic, langchain-groq, langchain-ollama
langchain-google-genai, langchain-core, langchain-community
streamlit, fastapi, chromadb>=0.4.0, requests, pypdf
```

Full list in `env/requirements.txt`.

---

## Author

**Vaibhav Lande** — [GitHub](https://github.com/VaibhavLande84)

*Built for accessible, evidence-based cancer care information. Always consult a healthcare provider for medical decisions.*
