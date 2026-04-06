"""
Baseline inference script for pharma-interaction-gym.
Strictly follows the Meta x HuggingFace OpenEnv Hackathon [START], [STEP], [END] format.
"""

import os
import json
import asyncio
import re
from typing import List, Optional
from openai import OpenAI
import requests

# ── Config from environment variables ────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

# Logging Helpers
def log_start(task: str, env: str, model: str):
    print(json.dumps({"type": "[START]", "task": task, "env": env, "model": model}), flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None):
    print(json.dumps({"type": "[STEP]", "step": step, "action": action, "reward": reward, "done": done, "error": error}), flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    print(json.dumps({"type": "[END]", "success": success, "steps": steps, "score": score, "rewards": rewards}), flush=True)

SYSTEM_PROMPT = """You are an expert clinical pharmacist AI.
Use the available tools to assess drug interaction safety.
Check EVERY current medication against the proposed drug.
Respond with EXACTLY this JSON format for tool calls:
{"tool_name": "name", "arguments": {"arg": "val"}}
When finished, use the 'submit_answer' tool.
"""

async def get_model_message(client: OpenAI, step: int, last_obs: str, last_reward: float, history: List[str]) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"History: {history}\nLast Observation: {last_obs}\nLast Reward: {last_reward}\nYour tool call (JSON):"}
    ]
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.1,
            max_tokens=500
        )
        text = (completion.choices[0].message.content or "").strip()
        # Extract JSON
        match = re.search(r'\{.*\}', text, re.DOTALL)
        return match.group() if match else text
    except Exception as e:
        return json.dumps({"tool_name": "lookup_drug", "arguments": {"drug_name": "Error"}})

async def run_task(task_level: str):
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy")
    
    # Environment interactions (using REST for simplicity in baseline)
    reset_resp = requests.post(f"{ENV_BASE_URL}/reset", params={"task_level": task_level})
    reset_data = reset_resp.json()
    session_id = reset_data["info"]["session_id"]
    obs = reset_data["observation"]
    
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    max_steps = obs["max_steps"]

    log_start(task=task_level, env="pharma-interaction-gym", model=MODEL_NAME)

    last_obs = obs["task_description"]
    last_reward = 0.0
    done = False

    try:
        for step in range(1, max_steps + 1):
            if done: break

            action_json_str = await get_model_message(client, step, last_obs, last_reward, history)
            
            try:
                action = json.loads(action_json_str)
                tool_name = action.get("tool_name", "lookup_drug")
                args = action.get("arguments", {})
            except:
                tool_name = "lookup_drug"
                args = {"drug_name": "Invalid JSON"}

            step_resp = requests.post(
                f"{ENV_BASE_URL}/step/{session_id}",
                json={"action_type": "tool_call", "tool_name": tool_name, "arguments": args}
            )
            step_data = step_resp.json()
            
            obs = step_data["observation"]
            reward = step_data.get("reward", 0.0)
            done = step_data.get("done", False)
            
            rewards.append(reward)
            steps_taken = step
            last_obs = obs.get("message", "")
            last_reward = reward

            log_step(step=step, action=action_json_str, reward=reward, done=done)
            history.append(f"Step {step}: {tool_name} -> {reward}")

            if done: break

        score = sum(rewards)
        score = min(max(score, 0.0), 1.0)
        success = score >= 0.7

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

async def main():
    for level in ["easy", "medium", "hard"]:
        await run_task(level)

if __name__ == "__main__":
    asyncio.run(main())
