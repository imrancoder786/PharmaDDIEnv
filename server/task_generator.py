import random
import json
from pathlib import Path
from ..models import TaskLevel

class TaskGenerator:
    def __init__(self):
        self.data_dir = Path(__file__).parent.parent / "data"
        self.interactions = json.loads((self.data_dir / "drug_interactions.json").read_text())
        self.patients = json.loads((self.data_dir / "patient_profiles.json").read_text())
        self.metadata = json.loads((self.data_dir / "drug_metadata.json").read_text())

    def generate_task(self, level: TaskLevel, seed: int = None) -> dict:
        if seed is not None:
            random.seed(seed)
        
        if level == TaskLevel.easy:
            return self._easy_task()
        elif level == TaskLevel.medium:
            return self._medium_task()
        else:
            return self._hard_task()

    def _easy_task(self) -> dict:
        """Single serious interaction detection."""
        serious = [i for i in self.interactions if i["severity"] in ["major", "contraindicated"]]
        pair = random.choice(serious)
        
        return {
            "task_id": f"easy_{random.randint(1000,9999)}",
            "task_level": "easy",
            "patient_profile": {
                "age": random.randint(35, 70),
                "weight_kg": random.randint(55, 95),
                "renal_function": "normal",
                "liver_function": "normal",
                "current_medications": [pair["drug_a"]],
                "diagnoses": ["Requires medication review"],
                "allergies": []
            },
            "proposed_drug": pair["drug_b"],
            "current_medications": [pair["drug_a"]],
            "task_description": (
                f"A patient is currently taking {pair['drug_a']}. "
                f"The physician wants to prescribe {pair['drug_b']}. "
                f"Assess the safety of this combination."
            ),
            "ground_truth": pair
        }

    def _medium_task(self) -> dict:
        """Polypharmacy with multiple interactions."""
        # Pick a patient with several meds
        patient = random.choice([p for p in self.patients if len(p["current_medications"]) >= 3])
        
        # Find a drug that has at least 2 interactions with the patient's meds
        candidate_drugs = list(self.metadata.keys())
        random.shuffle(candidate_drugs)
        
        target_drug = None
        found_interactions = []
        
        for drug in candidate_drugs:
            if drug in patient["current_medications"]:
                continue
                
            ints = []
            for p_med in patient["current_medications"]:
                for inter in self.interactions:
                    if (inter["drug_a"] == drug and inter["drug_b"] == p_med) or \
                       (inter["drug_b"] == drug and inter["drug_a"] == p_med):
                        ints.append(inter)
            
            if len(ints) >= 2:
                target_drug = drug
                found_interactions = ints
                break
        
        # Fallback if no drug found (unlikely with 200+ interactions)
        if not target_drug:
            return self._easy_task() 

        return {
            "task_id": f"medium_{random.randint(1000,9999)}",
            "task_level": "medium",
            "patient_profile": patient,
            "proposed_drug": target_drug,
            "current_medications": patient["current_medications"],
            "task_description": (
                f"Patient ({patient['id']}) is taking {', '.join(patient['current_medications'])}. "
                f"Physician wants to add {target_drug}. Identify all interactions and provide a safety verdict."
            ),
            "ground_truth": {
                "interactions": found_interactions,
                "patient_meds": patient["current_medications"]
            }
        }

    def _hard_task(self) -> dict:
        """Complex patient with organ impairment and clinical choice."""
        # Pick a patient with impairment
        impaired_patients = [p for p in self.patients if p["renal_function"] != "normal" or p["liver_function"] != "normal"]
        patient = random.choice(impaired_patients) if impaired_patients else self.patients[0]
        
        indications = ["Severe Pain", "Hypertension", "Atrial Fibrillation", "Bacterial Infection"]
        indication = random.choice(indications)
        
        # Define some candidate drug options for the indication
        indication_map = {
            "Severe Pain": ["Tramadol", "Ibuprofen", "Acetaminophen", "Oxycodone"],
            "Hypertension": ["Lisinopril", "Spironolactone", "Amlodipine", "Hydrochlorothiazide"],
            "Atrial Fibrillation": ["Warfarin", "Rivaroxaban", "Digoxin", "Amiodarone"],
            "Bacterial Infection": ["Clarithromycin", "Ciprofloxacin", "Azithromycin", "Amoxicillin"]
        }
        
        candidates = indication_map[indication]
        
        return {
            "task_id": f"hard_{random.randint(1000,9999)}",
            "task_level": "hard",
            "patient_profile": patient,
            "proposed_drug": f"One of {', '.join(candidates)}",
            "current_medications": patient["current_medications"],
            "task_description": (
                f"Patient has {indication} and {', '.join(patient['diagnoses'])}. "
                f"Current meds: {', '.join(patient['current_medications'])}. "
                f"Renal: {patient['renal_function']}, Liver: {patient['liver_function']}. "
                f"Which of these is safest: {', '.join(candidates)}? Provide full reasoning and dose adjustments."
            ),
            "ground_truth": {
                "indication": indication,
                "candidates": candidates,
                "patient_profile": patient,
                "key_points": [
                    "Must check interactions for all 4 candidates",
                    "Must account for renal/liver impairment in dosing",
                    "Must avoid contraindicated combinations",
                    "Must rank by safety and effectiveness"
                ]
            }
        }
