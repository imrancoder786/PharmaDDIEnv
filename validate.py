import requests
import json
import sys

BASE_URL = "http://localhost:7860"

def test_health():
    print("Testing /health endpoint...")
    try:
        resp = requests.get(f"{BASE_URL}/health")
        print(f"Response: {resp.json()}")
        return resp.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_tasks():
    print("\nTesting /tasks endpoint...")
    try:
        resp = requests.get(f"{BASE_URL}/tasks")
        print(f"Response: {json.dumps(resp.json(), indent=2)}")
        return resp.status_code == 200
    except Exception as e:
        print(f"Tasks check failed: {e}")
        return False

def test_reset():
    print("\nTesting /reset endpoint (easy level)...")
    try:
        resp = requests.post(f"{BASE_URL}/reset", params={"task_level": "easy"})
        data = resp.json()
        print(f"Observation: {data['observation']['task_description']}")
        print(f"Session ID: {data['info']['session_id']}")
        return resp.status_code == 200, data['info']['session_id']
    except Exception as e:
        print(f"Reset check failed: {e}")
        return False, None

def test_step(session_id):
    print(f"\nTesting /step endpoint for session {session_id}...")
    try:
        action = {
            "action_type": "tool_call",
            "tool_name": "lookup_drug",
            "arguments": {"drug_name": "Warfarin"}
        }
        resp = requests.post(f"{BASE_URL}/step/{session_id}", json=action)
        data = resp.json()
        print(f"Tool Result: {data['observation']['message']}")
        return resp.status_code == 200
    except Exception as e:
        print(f"Step check failed: {e}")
        return False

def main():
    print("=== PHARMA INTERACTION GYM VALIDATOR ===\n")
    
    if not test_health():
        print("FAIL: Server not reachable. Make sure it is running on port 7860.")
        sys.exit(1)
        
    if not test_tasks():
        print("FAIL: Tasks endpoint error.")
        
    ok, session_id = test_reset()
    if not ok:
        print("FAIL: Reset endpoint error.")
        sys.exit(1)
        
    if not test_step(session_id):
        print("FAIL: Step endpoint error.")
        sys.exit(1)
        
    print("\nSUCCESS: Environment passed basic validation!")

if __name__ == "__main__":
    main()
