import sys

with open("/home/user/strata/serve/server.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

start_idx = None
for i, line in enumerate(lines):
    if "def _openai(self, req):" in line:
        start_idx = i
        break

sse_idx = None
for i in range(start_idx, start_idx + 100):
    if "self._sse()" in lines[i]:
        sse_idx = i
        break

except_idx = None
for i in range(sse_idx, sse_idx + 40):
    if "except OSError:" in lines[i]:
        except_idx = i
        break

replacement = """            self._sse()
            try:
                last_chunk = None
                for c in chunks:
                    if c is None:
                        self.wfile.write(b": keep-alive\\n\\n")      # an SSE comment: clients ignore it
                    else:
                        last_chunk = c
                        self.wfile.write(b"data: " + json.dumps(c, ensure_ascii=False).encode() + b"\\n\\n")
                    self.wfile.flush()
                if last_chunk is not None and not last_chunk.get("choices", [{}])[0].get("finish_reason"):
                    finish_chunk = {
                        "id": last_chunk.get("id"),
                        "object": "chat.completion.chunk",
                        "created": last_chunk.get("created"),
                        "model": last_chunk.get("model"),
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                    }
                    self.wfile.write(b"data: " + json.dumps(finish_chunk, ensure_ascii=False).encode() + b"\\n\\n")
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\\n\\n")
"""

lines[sse_idx:except_idx] = [replacement]

with open("/home/user/strata/serve/server.py", "w", encoding="utf-8") as f:
    f.writelines(lines)

print("SUCCESS")
