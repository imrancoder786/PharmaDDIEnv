---
title: Pharma Interaction Gym
emoji: 💊
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Pharma Interaction Gym

An OpenEnv reinforcement learning environment that trains LLMs to reason 
like clinical pharmacists — detecting dangerous drug interactions, explaining 
pharmacological mechanisms, and recommending safe alternatives.

## Why This Matters

Drug interactions cause 30% of hospital admissions in India and millions 
of preventable deaths worldwide. LLMs trained on this environment can power 
real clinical decision support tools.

## Tasks

| Level | Description | Reward |
|-------|-------------|--------|
| Easy | Detect interaction between 2 drugs | 0.0–1.0 |
| Medium | Screen full polypharmacy regimen | 0.0–1.0 |
| Hard | Recommend safest option for complex patient | 0.0–1.0 |

## Available MCP Tools

- `lookup_drug` — drug class, metabolism, CYP pathways
- `check_interaction` — known interaction between two drugs
- `get_patient_labs` — patient organ function values
- `search_alternatives` — safer drug alternatives
- `get_dosing_guideline` — dose adjustments for organ impairment
- `submit_answer` — submit final assessment for scoring

## Quick Start

```bash
pip install requests openai
python inference.py
```

## Environment Variables

```
API_BASE_URL   LLM API endpoint
MODEL_NAME     Model identifier
HF_TOKEN       Hugging Face token
ENV_BASE_URL   This Space URL (default: localhost:7860)
```

## Reward Breakdown

- **Easy**: interaction detection (40%) + severity (30%) + mechanism (20%) + recommendation (10%)
- **Medium**: recall (40%) + precision (30%) + severity accuracy (20%) + recommendation (10%)
- **Hard**: safety completeness (35%) + alternatives (25%) + dose adjustment (20%) + reasoning (20%)
