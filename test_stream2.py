import requests
import json

response = requests.post(
    "http://localhost:8000/api/chat/stream",
    json={"query": "hello", "session_id": "test_session", "uploaded_file_path": None},
    stream=True
)

print(f"Status Code: {response.status_code}")
for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))
