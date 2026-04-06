from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from enum import Enum

class SeverityLevel(str, Enum):
    contraindicated = "contraindicated"
    major = "major"
    moderate = "moderate"
    minor = "minor"
    none = "none"

class TaskLevel(str, Enum):
    easy = "easy"
    medium = "medium"
    hard = "hard"

# ── Observation (what the agent sees after reset/step) ──────────────────────

class DrugInteractionObservation(BaseModel):
    task_id: str
    task_level: TaskLevel
    patient_profile: dict
    proposed_drug: str
    current_medications: List[str]
    task_description: str
    available_tools: List[str]
    step_count: int
    max_steps: int
    done: bool = False
    message: str = ""

# ── Actions (what the agent sends to step()) ────────────────────────────────

class ToolCallAction(BaseModel):
    """Agent calls one of the environment's MCP tools"""
    action_type: Literal["tool_call"] = "tool_call"
    tool_name: str = Field(..., description="One of: lookup_drug, check_interaction, get_patient_labs, search_alternatives, get_dosing_guideline, submit_answer")
    arguments: dict = Field(default_factory=dict)

# ── Tool Results (returned inside observation after tool call) ───────────────

class ToolResult(BaseModel):
    tool_name: str
    success: bool
    result: dict
    error: Optional[str] = None

# ── Final Answer (agent submits this to get graded) ─────────────────────────

class DrugInteractionAnswer(BaseModel):
    """The agent's final clinical assessment — submitted via submit_answer tool"""
    interactions_found: List[dict] = Field(
        description="List of {drug_a, drug_b, severity, mechanism, clinical_effect}"
    )
    overall_safety_verdict: Literal["safe", "use_with_caution", "avoid", "contraindicated"]
    recommendations: List[str]
    alternatives: List[str]
    monitoring_plan: str
    clinical_reasoning: str

# ── OpenEnv Standard Response ────────────────────────────────────────────────

class StepResult(BaseModel):
    observation: DrugInteractionObservation
    reward: float = Field(ge=0.0, le=1.0)
    done: bool
    info: dict = Field(default_factory=dict)

class ResetResult(BaseModel):
    observation: DrugInteractionObservation
    info: dict = Field(default_factory=dict)

class StateResult(BaseModel):
    observation: DrugInteractionObservation
    step_count: int
    done: bool
