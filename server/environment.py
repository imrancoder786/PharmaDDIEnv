import uuid
import random
from typing import Optional, Dict
from .models import *
from .task_generator import TaskGenerator
from .tools import execute_tool
from .graders import grade_answer

# In-memory session store
sessions: Dict[str, dict] = {}

task_generator = TaskGenerator()

def env_reset(task_level: Optional[str] = None, seed: Optional[int] = None) -> ResetResult:
    """Creates a new episode."""
    try:
        level = TaskLevel(task_level) if task_level else random.choice(list(TaskLevel))
    except ValueError:
        level = random.choice(list(TaskLevel))
        
    task = task_generator.generate_task(level, seed)
    session_id = str(uuid.uuid4())
    
    sessions[session_id] = {
        "session_id": session_id,
        "task": task,
        "step_count": 0,
        "max_steps": 20 if level == TaskLevel.hard else 15,
        "done": False,
        "tool_call_history": [],
        "found_interactions": set(), # Track unique interactions found for progress rewards
        "unique_drugs_looked_up": set(),
        "accumulated_progress_reward": 0.0,
        "final_answer": None
    }
    
    obs = DrugInteractionObservation(
        task_id=task["task_id"],
        task_level=task["task_level"],
        patient_profile=task["patient_profile"],
        proposed_drug=task["proposed_drug"],
        current_medications=task["current_medications"],
        task_description=task["task_description"],
        available_tools=["lookup_drug", "check_interaction", "get_patient_labs",
                        "search_alternatives", "get_dosing_guideline", "submit_answer"],
        step_count=0,
        max_steps=sessions[session_id]["max_steps"],
        done=False,
        message="Episode started. Use available tools to assess drug interaction safety."
    )
    
    return ResetResult(observation=obs, info={"session_id": session_id})

def env_step(session_id: str, action: ToolCallAction) -> StepResult:
    """Processes one agent action with partial reward signals."""
    session = sessions.get(session_id)
    if not session:
        raise ValueError(f"Session {session_id} not found")
    
    if session["done"]:
        return StepResult(observation=env_state(session_id).observation, reward=0.0, done=True)
        
    session["step_count"] += 1
    tool_name = action.tool_name
    arguments = action.arguments
    
    # Execute tool
    tool_result = execute_tool(tool_name, arguments, session)
    
    reward = 0.0
    done = False
    info = {}
    
    # --- Partial Reward Logic (Progress Signals) ---
    # Max progress reward is capped at 0.3, remaining 0.7 comes from final submission
    
    if tool_name == "lookup_drug" and tool_result.success:
        drug = arguments.get("drug_name", "").lower()
        if drug not in session["unique_drugs_looked_up"]:
            session["unique_drugs_looked_up"].add(drug)
            reward = 0.05
            
    elif tool_name == "check_interaction" and tool_result.success:
        res = tool_result.result
        if res.get("severity") and res["severity"] != "none":
            pair = tuple(sorted([res["drug_a"].lower(), res["drug_b"].lower()]))
            if pair not in session["found_interactions"]:
                session["found_interactions"].add(pair)
                reward = 0.1
                
    # Cap accumulated progress reward
    session["accumulated_progress_reward"] += reward
    if session["accumulated_progress_reward"] > 0.3:
        reward = 0.0 # No more progress reward after 0.3
    
    # --- Terminal Conditions ---
    if tool_name == "submit_answer" and tool_result.success:
        # Final grade scale 0.0 - 0.7
        final_grade, breakdown = grade_answer(arguments, session["task"])
        reward = (final_grade * 0.7) + min(session["accumulated_progress_reward"], 0.3)
        done = True
        session["done"] = True
        info["grading_breakdown"] = breakdown
        
    elif session["step_count"] >= session["max_steps"]:
        done = True
        session["done"] = True
        reward = 0.0 # Penalty for timeout
        info["message"] = "Max steps reached."

    # Build updated observation
    task = session["task"]
    obs = DrugInteractionObservation(
        task_id=task["task_id"],
        task_level=task["task_level"],
        patient_profile=task["patient_profile"],
        proposed_drug=task["proposed_drug"],
        current_medications=task["current_medications"],
        task_description=task["task_description"],
        available_tools=["lookup_drug", "check_interaction", "get_patient_labs",
                        "search_alternatives", "get_dosing_guideline", "submit_answer"],
        step_count=session["step_count"],
        max_steps=session["max_steps"],
        done=done,
        message=f"Tool '{tool_name}' result: {tool_result.result if tool_result.success else tool_result.error}"
    )
    
    return StepResult(observation=obs, reward=round(reward, 3), done=done, info=info)

def env_state(session_id: str) -> StateResult:
    session = sessions.get(session_id)
    if not session: raise ValueError("Not found")
    task = session["task"]
    obs = DrugInteractionObservation(
        task_id=task["task_id"], task_level=task["task_level"],
        patient_profile=task["patient_profile"], proposed_drug=task["proposed_drug"],
        current_medications=task["current_medications"], task_description=task["task_description"],
        available_tools=["lookup_drug", "check_interaction", "get_patient_labs", "submit_answer"],
        step_count=session["step_count"], max_steps=session["max_steps"], done=session["done"],
        message="State requested"
    )
    return StateResult(observation=obs, step_count=session["step_count"], done=session["done"])
