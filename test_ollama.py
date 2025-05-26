import requests

def test_ollama():
    url = "http://127.0.0.1:11434/api/generate"
    payload = {
        "model": "gemma:3b",
        "prompt": "Hello, are you working?",
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        print("Ollama response:", result.get("response", "No response"))
        return True
    except Exception as e:
        print(f"Error testing Ollama: {str(e)}")
        return False

if __name__ == "__main__":
    test_ollama()
