import requests
import json
import time

INSTRUCTION = "Respond to the following situation shared with you by a person in no more than 150 words. Situation:"

MODEL_NAME = "olmo-3:7b-instruct-q8_0" 

scenarios = [
    
    {
        "id": "S01_I",
        "condition": "implicit",
        "emotion": "guilty",
        "text": f"{INSTRUCTION} I saw a little turtle about to get hit on the road the other day and I was in a hurry and didn't stop and get him. I hope he made it across the road, poor little guy. "
    },
    {
        "id": "S01_E",
        "condition": "explicit",
        "emotion": "guilty",
        "text": f"{INSTRUCTION} I saw a little turtle about to get hit on the road the other day and I was in a hurry and didn't stop and get him. I hope he made it across the road, poor little guy. I felt really guilty."
    },

]

def prompt_olmo(model_name, scenario_text):
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": scenario_text}
        ],
        "stream": False,
        "options": {
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 42,
            "num_predict": 250
        }
    }

    try:
        response = requests.post(
            "http://localhost:11434/api/chat", 
            json=payload,
            timeout=600
        )
        response.raise_for_status()
        return response.json()["message"]["content"].strip()

    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to Ollama. Is it running?")
        raise
    except requests.exceptions.ReadTimeout:
        print("ERROR: Ollama took too long to respond. Try increasing timeout or use a smaller/faster model.")
        raise
    except requests.exceptions.HTTPError as e:
        print(f"ERROR: HTTP {e.response.status_code} — {e.response.text}")
        raise
    except KeyError:
        print(f"ERROR: Unexpected response format: {response.json()}")
        raise


results = []

for s in scenarios:
    print(f"Running {s['id']} ({s['condition']}) ...", end=" ", flush=True)

    start = time.time()
    response_text = prompt_olmo(MODEL_NAME, s["text"])
    duration = round(time.time() - start, 2)

    results.append({
        **s,
        "model": MODEL_NAME,
        "response": response_text,
        "duration_seconds": duration
    })

    print(f"done in {duration}s")

with open("olmo3_responses.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\nSaved to olmo3_responses.json")
