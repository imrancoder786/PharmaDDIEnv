from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models import ResetResult, StepResult, StateResult, ToolCallAction, ToolResult
from .PharmaEnv_environment import env_reset, env_step, env_state

app = FastAPI(
    title="Pharma Interaction Gym",
    description="OpenEnv environment for training LLMs on pharmaceutical drug interaction reasoning",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/health")
def health():
    return {"status": "ok", "environment": "pharma-interaction-gym"}

@app.post("/reset", response_model=ResetResult)
def reset(task_level: Optional[str] = None, seed: Optional[int] = None):
    """Start a new episode. Returns initial observation + session_id in info."""
    return env_reset(task_level, seed)

@app.post("/step/{session_id}", response_model=StepResult)
def step(session_id: str, action: ToolCallAction):
    """Take one action (tool call) in the environment."""
    try:
        return env_step(session_id, action)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/state/{session_id}", response_model=StateResult)
def state(session_id: str):
    """Get current state without advancing the episode."""
    try:
        return env_state(session_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/tasks")
def list_tasks():
    """List available task levels and descriptions."""
    return {
        "tasks": [
            {
                "id": "drug_interaction_easy",
                "level": "easy",
                "description": "Detect interaction between 2 drugs and classify severity",
                "reward_range": [0.0, 1.0]
            },
            {
                "id": "drug_interaction_medium", 
                "level": "medium",
                "description": "Screen a 5-drug regimen for all interactions with a proposed new drug",
                "reward_range": [0.0, 1.0]
            },
            {
                "id": "drug_interaction_hard",
                "level": "hard", 
                "description": "Complex patient with organ impairment: recommend safest drug option",
                "reward_range": [0.0, 1.0]
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
