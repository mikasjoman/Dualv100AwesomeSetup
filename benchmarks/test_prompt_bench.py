import urllib.request
import json
import time

SERVER_IP = "192.168.1.240"
PORT = 8086

# Generate a 4000-token prompt
prompt_text = "Summarize and analyze the mathematical and architectural trade-offs of large language models: " * 150

payload = {
    "model": "qwen3.8-flash-next-ud-q4_k_xl",
    "messages": [
        {"role": "user", "content": prompt_text}
    ],
    "max_tokens": 50,
    "temperature": 0.0,
    "stream": False
}

req = urllib.request.Request(
    f"http://{SERVER_IP}:{PORT}/v1/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

t0 = time.time()
try:
    with urllib.request.urlopen(req, timeout=120) as resp:
        res = json.loads(resp.read().decode())
        t1 = time.time()
        print(f"Total time: {t1 - t0:.2f}s")
        print("Usage:", res.get("usage"))
except Exception as e:
    print("Error:", e)
