"""
entity_linker.py
──────────────────────────────────────────────
Path 3: NLM Clinical Tables API
Maps natural-language medical terms to standard
codes (ICD-10, RxNorm) so the retriever can
query MedlinePlus Connect and DailyMed.
"""

import requests
import json
import re
from typing import Optional
from dataclasses import dataclass, field, asdict


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class LinkedCode:
    code: str 
    name: str                
    source: str              
    score: float = 0.0       


@dataclass
class LinkingResult:
    """Full result of an entity-linking pass."""
    raw_query: str
    conditions: list[LinkedCode] = field(default_factory=list)
    medications: list[LinkedCode] = field(default_factory=list)
    error: Optional[str] = None

    def to_prompt_context(self) -> str:
        """Format for injection into the LLM prompt."""
        parts = []
        valid_conditions = [c for c in self.conditions if c.code]
        if valid_conditions:
            codes = ", ".join(f"{c.code} ({c.name})" for c in valid_conditions)
            parts.append(f"ICD-10 codes: {codes}")
        valid_medications = [m for m in self.medications if m.code]
        if valid_medications:
            codes = ", ".join(f"{m.code} ({m.name})" for m in valid_medications)
            parts.append(f"RxNorm IDs: {codes}")
        return " | ".join(parts) if parts else "(No codes identified)"

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# API clients
# ---------------------------------------------------------------------------

CLINICAL_TABLES_BASE = "https://clinicaltables.nlm.nih.gov/api"


def search_icd10cm(term: str, max_results: int = 3) -> list[LinkedCode]:
    """
    Search the ICD-10-CM autocomplete API.
    Returns up to *max_results* LinkedCode objects.
    """
    url = f"{CLINICAL_TABLES_BASE}/icd10cm/v3/search"
    params = {"terms": term, "maxList": max_results}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as e:
        return [LinkedCode(code="", name=f"ICD-10 lookup failed: {e}", source="icd10cm", score=0.0)]

    # Response format: [total, columns, rows]
    # columns = ["code", "name"]
    try:
        rows = data[2] if isinstance(data, (list, tuple)) and len(data) > 2 and data[2] is not None else []
    except (IndexError, TypeError):
        rows = []
    results = []
    for row in rows[:max_results]:
        try:
            code, name = row[0], row[1]
            results.append(LinkedCode(code=code, name=name, source="icd10cm"))
        except (IndexError, TypeError):
            continue
    return results if results else [LinkedCode(code="", name=f"No ICD-10 match for '{term}'", source="icd10cm", score=0.0)]


def search_rxnorm(term: str, max_results: int = 3) -> list[LinkedCode]:
    """
    Search the RxTerms (RxNorm) autocomplete API.
    Returns up to *max_results* LinkedCode objects.
    """
    url = f"{CLINICAL_TABLES_BASE}/rxterms/v3/search"
    params = {"terms": term, "maxList": max_results}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as e:
        return [LinkedCode(code="", name=f"RxNorm lookup failed: {e}", source="rxnorm", score=0.0)]

    # Response format: [total, columns, rows]
    # columns = ["rxcui", "name", "term_type", ...]
    try:
        rows = data[2] if isinstance(data, (list, tuple)) and len(data) > 2 and data[2] is not None else []
    except (IndexError, TypeError):
        rows = []
    results = []
    for row in rows[:max_results]:
        try:
            rxcui, name = row[0], row[1]
            results.append(LinkedCode(code=rxcui, name=name, source="rxnorm"))
        except (IndexError, TypeError):
            continue
    return results if results else [LinkedCode(code="", name=f"No RxNorm match for '{term}'", source="rxnorm", score=0.0)]


# ---------------------------------------------------------------------------
# Main linker function
# ---------------------------------------------------------------------------

def link_entities(query: str) -> LinkingResult:
    """
    Parse a free-text medical query and return standardised codes.

    Uses simple heuristics to decide whether the query mentions a
    condition / drug, then calls the appropriate Clinical Tables API.

    For production, replace the heuristics with a small classifier or
    the LLM-based Path 2 approach described in the architecture doc.
    """
    result = LinkingResult(raw_query=query)

    q_lower = query.lower().strip()

    # ---------- Simple heuristic: split on keywords ----------
    # If the query contains drug-like keywords, search RxNorm
    drug_keywords = [
        "mg", "tablet", "capsule", "dose", "prescription",
        "drug", "medication", "medicine", "lisinopril",
        "ibuprofen", "advil", "tylenol", "aspirin",
        " statin", "anti", "-blocker", " inhibitor",
    ]
    is_drug_query = any(kw in q_lower for kw in drug_keywords)

    # Condition keywords
    condition_keywords = [
        "cancer", "tumor", "diagnosis", "disease", "syndrome",
        "infection", "pain", "fracture", "diabetes", "hypertension",
        "asthma", "copd", "arthritis", "depression", "anxiety",
    ]
    is_condition_query = any(kw in q_lower for kw in condition_keywords)

    # If we can't guess, search both
    if not is_drug_query and not is_condition_query:
        is_condition_query = True
        is_drug_query = True

    # Extract the most salient noun phrase for each category
    # (Simple first-pass: just use the whole query)
    if is_condition_query:
        result.conditions = search_icd10cm(query)

    if is_drug_query:
        result.medications = search_rxnorm(query)

    return result


# ---------------------------------------------------------------------------
# Convenience: extract just the code strings
# ---------------------------------------------------------------------------

def extract_icd10_codes(query: str) -> list[str]:
    """Quick helper — return only ICD-10 code strings."""
    result = link_entities(query)
    return [c.code for c in result.conditions if c.code]


def extract_rxnorm_ids(query: str) -> list[str]:
    """Quick helper — return only RxNorm ID strings."""
    result = link_entities(query)
    return [c.code for c in result.medications if c.code]


# ---------------------------------------------------------------------------
# Demo (run directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_queries = [
        "lung cancer symptoms",
        "advil dosage",
        "type 2 diabetes treatment",
        "lisinopril side effects",
    ]
    for q in test_queries:
        res = link_entities(q)
        print(f"\nQuery: {q}")
        print(f"  Conditions: {[(c.code, c.name) for c in res.conditions]}")
        print(f"  Medications: {[(m.code, m.name) for m in res.medications]}")
