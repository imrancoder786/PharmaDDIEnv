import json
from pathlib import Path
from typing import Dict, Any
from .models import ToolResult

DATA_DIR = Path("PharmaEnv/data")

def load_interactions():
    return json.loads((DATA_DIR / "drug_interactions.json").read_text())

def load_metadata():
    return json.loads((DATA_DIR / "drug_metadata.json").read_text())

def lookup_drug(arguments: dict, session_state: dict) -> ToolResult:
    """
    Arguments: {"drug_name": "Warfarin"}
    Returns: drug class, metabolism pathways, narrow therapeutic index flag,
             common indications, renal clearance flag
    """
    drug_name = arguments.get("drug_name", "").strip()
    metadata = load_metadata()
    
    # Case-insensitive lookup
    found_key = next((k for k in metadata if k.lower() == drug_name.lower()), None)
    
    if not found_key:
        return ToolResult(
            tool_name="lookup_drug",
            success=False,
            result={},
            error=f"Drug '{drug_name}' not found in database."
        )
    
    return ToolResult(
        tool_name="lookup_drug",
        success=True,
        result=metadata[found_key]
    )

def check_interaction(arguments: dict, session_state: dict) -> ToolResult:
    """
    Arguments: {"drug_a": "Warfarin", "drug_b": "Aspirin"}
    Returns: interaction data if exists, or {"interaction": false} if no known interaction
    """
    drug_a = arguments.get("drug_a", "").strip().lower()
    drug_b = arguments.get("drug_b", "").strip().lower()
    
    if not drug_a or not drug_b:
        return ToolResult(
            tool_name="check_interaction",
            success=False,
            result={},
            error="Both drug_a and drug_b must be provided."
        )
    
    interactions = load_interactions()
    
    for inter in interactions:
        ia = inter["drug_a"].lower()
        ib = inter["drug_b"].lower()
        
        if (ia == drug_a and ib == drug_b) or (ia == drug_b and ib == drug_a):
            return ToolResult(
                tool_name="check_interaction",
                success=True,
                result=inter
            )
            
    return ToolResult(
        tool_name="check_interaction",
        success=True,
        result={"interaction": False, "message": f"No known major interaction found between {drug_a} and {drug_b} in our database."}
    )

def get_patient_labs(arguments: dict, session_state: dict) -> ToolResult:
    """
    Arguments: {"lab_type": "renal_function"}  
    lab_type options: "renal_function", "liver_function", "age", "weight", "allergies"
    """
    lab_type = arguments.get("lab_type", "")
    profile = session_state.get("task", {}).get("patient_profile", {})
    
    mapping = {
        "renal_function": "renal_function",
        "liver_function": "liver_function",
        "age": "age",
        "weight": "weight_kg",
        "allergies": "allergies"
    }
    
    if lab_type not in mapping:
        return ToolResult(
            tool_name="get_patient_labs",
            success=False,
            result={},
            error=f"Invalid lab_type '{lab_type}'. Available: {list(mapping.keys())}"
        )
    
    val = profile.get(mapping[lab_type])
    return ToolResult(
        tool_name="get_patient_labs",
        success=True,
        result={lab_type: val}
    )

def search_alternatives(arguments: dict, session_state: dict) -> ToolResult:
    """
    Arguments: {"drug_name": "Warfarin", "reason": "interacts with Aspirin"}
    Returns: list of drug alternatives from the interactions data
    """
    drug_name = arguments.get("drug_name", "").strip().lower()
    interactions = load_interactions()
    
    alternatives = []
    for inter in interactions:
        if inter["drug_a"].lower() == drug_name or inter["drug_b"].lower() == drug_name:
            alternatives.extend(inter.get("alternatives", []))
            
    # Remove duplicates
    alternatives = list(set(alternatives))
    
    return ToolResult(
        tool_name="search_alternatives",
        success=True,
        result={"drug_name": drug_name, "alternatives": alternatives}
    )

def get_dosing_guideline(arguments: dict, session_state: dict) -> ToolResult:
    """
    Arguments: {"drug_name": "Metformin", "condition": "renal_impairment"}
    """
    drug_name = arguments.get("drug_name", "").strip().lower()
    condition = arguments.get("condition", "").strip().lower()
    
    # Comprehensive hardcoded guidelines
    guidelines = {
        "metformin": {
            "renal_impairment": "eGFR 30-45: Max dose 1000mg/day. eGFR < 30: Contraindicated.",
            "liver_impairment": "Avoid due to increased risk of lactic acidosis."
        },
        "digoxin": {
            "renal_impairment": "Reduce dose by 50% or increase dosing interval. Monitor levels closely.",
            "elderly": "Use lower doses (e.g., 0.125mg daily or every other day) due to reduced renal clearance."
        },
        "warfarin": {
            "liver_impairment": "Enhanced anticoagulant effect. Monitor INR more frequently.",
            "elderly": "Lower initial doses recommended due to increased sensitivity."
        },
        "lithium": {
            "renal_impairment": "Contraindicated or require major dose reduction with extreme caution and frequent level monitoring."
        },
        "enoxaparin": {
            "renal_impairment": "CrCl < 30 mL/min: Reduce dose to 30mg once daily for DVT prophylaxis."
        },
        "allopurinol": {
            "renal_impairment": "Start at 100mg/day or less. Adjust based on CrCl."
        },
        "gabapentin": {
            "renal_impairment": "CrCl 30-59: 400-1400mg/day. CrCl 15-29: 200-700mg/day. CrCl 15: 100-300mg/day."
        },
        "rivaroxaban": {
            "renal_impairment": "CrCl 15-50: 15mg once daily for AFib. CrCl < 15: Avoid."
        },
        "spironolactone": {
            "renal_impairment": "CrCl < 30: Avoid due to risk of hyperkalemia."
        },
        "lisinopril": {
            "renal_impairment": "Initial dose 5mg in patients with CrCl 10-30 mL/min."
        },
        "methotrexate": {
            "renal_impairment": "CrCl 10-50: 50% dose. CrCl < 10: Avoid."
        },
        "gentamicin": {
            "renal_impairment": "Major adjustment required. Dose based on serum levels and CrCl."
        },
        "naproxen": {
            "renal_impairment": "Not recommended in advanced renal disease (CrCl < 30)."
        },
        "ibuprofen": {
            "renal_impairment": "Avoid in patients with severe renal impairment."
        },
        "clarithromycin": {
            "renal_impairment": "CrCl < 30: Halve the dose."
        }
    }
    
    # Try to find matching drug and condition
    drug_match = next((d for d in guidelines if d in drug_name), None)
    if not drug_match:
        return ToolResult(
            tool_name="get_dosing_guideline",
            success=False,
            result={},
            error=f"No specific dosing guidelines found for drug '{drug_name}'."
        )
        
    cond_match = next((c for c in guidelines[drug_match] if c in condition), None)
    if not cond_match:
        return ToolResult(
            tool_name="get_dosing_guideline",
            success=False,
            result={},
            error=f"No specific dosing guidelines found for drug '{drug_name}' under condition '{condition}'."
        )
        
    return ToolResult(
        tool_name="get_dosing_guideline",
        success=True,
        result={"drug": drug_name, "condition": condition, "guideline": guidelines[drug_match][cond_match]}
    )

def submit_answer(arguments: dict, session_state: dict) -> ToolResult:
    """
    Arguments: full DrugInteractionAnswer as dict
    This is the terminal action — triggers grading
    """
    session_state["answer_submitted"] = True
    session_state["final_answer"] = arguments
    
    return ToolResult(
        tool_name="submit_answer",
        success=True,
        result={"submitted": True, "message": "Answer received, calculating score..."}
    )

TOOLS = {
    "lookup_drug": lookup_drug,
    "check_interaction": check_interaction,
    "get_patient_labs": get_patient_labs,
    "search_alternatives": search_alternatives,
    "get_dosing_guideline": get_dosing_guideline,
    "submit_answer": submit_answer,
}

def execute_tool(tool_name: str, arguments: dict, session_state: dict) -> ToolResult:
    """Router — calls the right tool, handles unknown tool names gracefully"""
    if tool_name not in TOOLS:
        return ToolResult(
            tool_name=tool_name,
            success=False,
            result={},
            error=f"Unknown tool '{tool_name}'. Available: {list(TOOLS.keys())}"
        )
    return TOOLS[tool_name](arguments, session_state)
