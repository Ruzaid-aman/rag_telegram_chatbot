import requests
import json

import json
import requests

def test_ollama():
    url = "http://127.0.0.1:11434/api/chat"  # Try the chat endpoint instead
    headers = {'Content-Type': 'application/json'}
    payload = {
        "model": "gemma3:1b",
        "messages": [
            {"role": "user", "content": "Hello, is Ollama working?"}
        ]
    }
    
    try:
        print(f"Sending request to {url}")
        print(f"Payload: {json.dumps(payload, indent=2)}")
        response = requests.post(url, json=payload, headers=headers)
        print(f"Status code: {response.status_code}")
        print(f"Response headers: {response.headers}")
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)}")
        else:
            print(f"Error response: {response.text}")
    except Exception as e:
        print(f"Error: {str(e)}")
if __name__ == "__main__":
    test_ollama()
