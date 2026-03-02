#!/usr/bin/env python3
import requests
import sys

def test_api():
    base_url = "http://127.0.0.1:8000/v1"
    
    # Test models endpoint
    try:
        resp = requests.get(f"{base_url}/models")
        print(f"Models endpoint status: {resp.status_code}")
        print(f"Models response: {resp.json()}")
    except Exception as e:
        print(f"Failed to query models: {e}")

    # Test completions endpoint
    try:
        resp = requests.post(
            f"{base_url}/chat/completions", 
            json={
                "model": "test", 
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 0.0
            }
        )
        print(f"Chat completions status: {resp.status_code}")
        print(f"Chat completions response: {resp.json()}")
    except Exception as e:
         print(f"Failed to query chat completions: {e}")

if __name__ == '__main__':
    test_api()