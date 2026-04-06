import os
import json
from openai import OpenAI
from ..models import TaskLevel, SeverityLevel

def grade_answer(answer: dict, task: dict) -> tuple[float, dict]:
    """Main grader router."""
    level = task["task_level"]
    
    if level == "easy":
        return grade_easy(answer, task)
    elif level == "medium":
        return grade_medium(answer, task)
    else:
        return grade_hard(answer, task)

def grade_easy(answer: dict, task: dict) -> tuple[float, dict]:
    """Single interaction detection grading."""
    score = 0.0
    breakdown = {}
    
    interactions_found = answer.get("interactions_found", [])
    ground_truth = task["ground_truth"]
    gt_pair = {ground_truth["drug_a"].lower(), ground_truth["drug_b"].lower()}
    
    # 1. Interaction Detection (0.4)
    detected = False
    for found in interactions_found:
        found_pair = {found.get("drug_a", "").lower(), found.get("drug_b", "").lower()}
        if found_pair == gt_pair:
            detected = True
            break
    
    if detected:
        score += 0.4
        breakdown["interaction_detection"] = 0.4
    else:
        breakdown["interaction_detection"] = 0.0
        
    # Penalize hallucinations
    hallucinations = max(0, len(interactions_found) - (1 if detected else 0))
    penalty = min(0.2, hallucinations * 0.1)
    score -= penalty
    breakdown["hallucination_penalty"] = -penalty
    
    # 2. Severity (0.3)
    if detected:
        severity_order = ["none", "minor", "moderate", "major", "contraindicated"]
        truth_sev = ground_truth["severity"].lower()
        found_sev = next(f.get("severity", "").lower() for f in interactions_found if {f.get("drug_a", "").lower(), f.get("drug_b", "").lower()} == gt_pair)
        
        if found_sev == truth_sev:
            score += 0.3
            breakdown["severity"] = 0.3
        elif found_sev in severity_order and truth_sev in severity_order:
            if abs(severity_order.index(found_sev) - severity_order.index(truth_sev)) == 1:
                score += 0.15
                breakdown["severity"] = 0.15
            else:
                breakdown["severity"] = 0.0
        else:
            breakdown["severity"] = 0.0
    else:
        breakdown["severity"] = 0.0
        
    # 3. Mechanism (0.2)
    reasoning = answer.get("clinical_reasoning", "").lower()
    if detected and ground_truth["mechanism"].lower() in reasoning:
        score += 0.1
        breakdown["mechanism_type"] = 0.1
    else:
        breakdown["mechanism_type"] = 0.0
        
    cyp = ground_truth.get("cyp_enzyme")
    if detected and (not cyp or cyp.lower() in reasoning):
        score += 0.1
        breakdown["mechanism_detail"] = 0.1
    else:
        breakdown["mechanism_detail"] = 0.0
        
    # 4. Recommendation (0.1)
    verdict = answer.get("overall_safety_verdict", "").lower()
    truth_rec = ground_truth["recommendation"].lower()
    rec_map = {
        "contraindicated": "contraindicated",
        "avoid_combination": "avoid",
        "avoid_or_adjust": "avoid",
        "hold_before_procedure": "use_with_caution"
    }
    if verdict == rec_map.get(truth_rec, truth_rec):
        score += 0.1
        breakdown["recommendation"] = 0.1
    else:
        breakdown["recommendation"] = 0.0
        
    final_score = max(0.0, min(1.0, score))
    breakdown["total"] = final_score
    return final_score, breakdown

def grade_medium(answer: dict, task: dict) -> tuple[float, dict]:
    """Polypharmacy grading."""
    score = 0.0
    breakdown = {}
    
    interactions_found = answer.get("interactions_found", [])
    gt_interactions = task["ground_truth"]["interactions"]
    
    # Normalize pairs
    gt_pairs = [set([i["drug_a"].lower(), i["drug_b"].lower()]) for i in gt_interactions]
    found_pairs = [set([i.get("drug_a", "").lower(), i.get("drug_b", "").lower()]) for i in interactions_found]
    
    found_real = 0
    correct_severities = 0
    
    for i, gt_pair in enumerate(gt_pairs):
        for j, f_pair in enumerate(found_pairs):
            if gt_pair == f_pair:
                found_real += 1
                if interactions_found[j].get("severity", "").lower() == gt_interactions[i]["severity"].lower():
                    correct_severities += 1
                break
                
    # 1. Recall (0.4)
    recall = found_real / len(gt_pairs) if gt_pairs else 1.0
    score += 0.4 * recall
    breakdown["recall"] = 0.4 * recall
    
    # 2. Precision (0.3)
    precision = found_real / len(interactions_found) if interactions_found else 0.0
    score += 0.3 * precision
    breakdown["precision"] = 0.3 * precision
    
    # 3. Severity Accuracy (0.2)
    sev_acc = correct_severities / found_real if found_real else 0.0
    score += 0.2 * sev_acc
    breakdown["severity_accuracy"] = 0.2 * sev_acc
    
    # 4. Recommendation (0.1) - Check if verdict is 'avoid' or 'contraindicated' if any major/contra exists
    has_serious = any(i["severity"] in ["major", "contraindicated"] for i in gt_interactions)
    verdict = answer.get("overall_safety_verdict", "").lower()
    if has_serious and verdict in ["avoid", "contraindicated"]:
        score += 0.1
        breakdown["recommendation"] = 0.1
    elif not has_serious and verdict in ["safe", "use_with_caution"]:
        score += 0.1
        breakdown["recommendation"] = 0.1
    else:
        breakdown["recommendation"] = 0.0
        
    final_score = max(0.0, min(1.0, score))
    breakdown["total"] = final_score
    return final_score, breakdown

def grade_hard(answer: dict, task: dict) -> tuple[float, dict]:
    """LLM-as-Judge grading."""
    api_key = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    
    if not api_key:
        # Fallback for evaluation if no key provided
        return 0.5, {"error": "No API key for judge LLM, using default score"}
        
    client = OpenAI(base_url=base_url, api_key=api_key)
    
    prompt = f"""You are a senior clinical pharmacist evaluating an AI agent's drug safety assessment.

Patient context: {json.dumps(task['patient_profile'])}
Clinical question: {task['task_description']}

Agent's answer: {json.dumps(answer)}

Ground truth key points: {json.dumps(task['ground_truth']['key_points'])}

Score the answer on these dimensions (each 0.0-1.0):
1. safety_completeness: Did it identify all major safety concerns? 
2. alternative_quality: Are the suggested alternatives clinically appropriate and safe?
3. dose_adjustment: Did it correctly account for organ function in dosing?
4. clinical_reasoning: Is the reasoning mechanistically correct and coherent?

Return ONLY valid JSON: {{"safety_completeness": X, "alternative_quality": X, "dose_adjustment": X, "clinical_reasoning": X}}"""

    try:
        response = client.chat.completions.create(
            model=os.environ.get("MODEL_NAME", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        scores = json.loads(response.choices[0].message.content)
        
        s = scores.get("safety_completeness", 0) * 0.35
        a = scores.get("alternative_quality", 0) * 0.25
        d = scores.get("dose_adjustment", 0) * 0.20
        r = scores.get("clinical_reasoning", 0) * 0.20
        
        final_score = max(0.0, min(1.0, s + a + d + r))
        scores["total"] = final_score
        return final_score, scores
    except Exception as e:
        return 0.4, {"error": str(e), "message": "Judge LLM failed"}
